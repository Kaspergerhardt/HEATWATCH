"""Google Takeout.

Used:
  live_pace_*.csv      per s, only during a recorded exercise: steps, distance (mm)
  gps_location_*.csv   per s while GPS is on: latitude, longitude
  heart_rate_*.csv     all-day HR every 1-5 s
  steps_*.csv          steps per minute
  HRV                  Daily_..._Summary (rmssd per night); 5-min RMSSD files only for nights without a summary
  No age / gender in the export.

Trainings: UserExercises_*.csv, names with 'Run' only; left out: < 2 min and MANUAL entries that repeat
  a recorded run (start within duplicate_tolerance). No UserExercises file: found from the data
  (block without a gap > 10 min, >= 5 min, 1.8-6.5 m/s, >= 120 steps/min).
"""
import re

import numpy as np
import pandas as pd

from ...settings import PROC
from .common import DAILY_COLS, SESSION_COLS, STREAM_COLS, empty, file_hash, haversine, progress

SOURCE = "fitbit"
TZ = PROC["local_timezone"]
GAP = pd.Timedelta(minutes=10)
MIN_DURATION, SPEED, MIN_CADENCE = 300, (1.8, 6.5), 120
MIN_DAY_HOURS = 12                    # resting HR only for days with HR over at least 12 hours
MIN_EXERCISE = 120                    # s; shorter recorded exercises = accidental start / stop, not a training
PAUSE = 10                            # s in a row without steps and distance in live_pace = pause
KINDS = [("exercises", r"userexercises"), ("live_pace", r"(^|[-_ ])live_pace_\d{4}"), ("gps", r"(^|[-_ ])gps_location_\d{4}"),
         ("heart_rate", r"(^|[-_ ])heart_rate_\d{4}"), ("steps", r"(^|[-_ ])steps_\d{4}"),
         ("hrv_summary", r"daily_heart_rate_variability_summary"),
         ("hrv_detail", r"(heart_rate_variability_details|(^|[-_ ])heart_rate_variability_\d{4})")]


# ------------------------------------------------------------------ helpers: file kind, time, pauses
def kind_of(path):
    """File name -> kind in KINDS (exercises, live_pace, gps, ...) or None."""
    n = path.name.lower()
    return next((k for k, pat in KINDS if re.search(pat, n)), None)


def local(t):
    """'2026-04-10T07:09:22Z' / '2026-01-09 16:43:55+0000' (UTC) -> local; without a zone it is already local."""
    t = pd.Series(t, dtype=str).str.strip()
    utc = t.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$")                  # '...Z' or '...+0000' / '+01:00'
    out = pd.to_datetime(t.where(~utc), errors="coerce")
    conv = pd.to_datetime(t.where(utc), utc=True, errors="coerce", format="mixed")
    return out.fillna(conv.dt.tz_convert(TZ).dt.tz_localize(None))


def blocks(t):
    """Block number per row: new block after a gap > GAP."""
    return (t.diff() > GAP).cumsum()


def window_sum(df, col, st, en):
    """Sum of `col` between st and en (0 when there is no table)."""
    return df.loc[(df.t >= st) & (df.t <= en), col].sum() if df is not None else 0


def paused(lp, st, en):
    """Seconds of pause between st and en: blocks of >= PAUSE s without steps and distance in live_pace."""
    if lp is None:
        return 0
    b = lp[(lp.t >= st) & (lp.t <= en)]
    still = (b.steps.fillna(0) <= 0) & (b.dist_m.fillna(0) <= 0)
    block = (still != still.shift()).cumsum()
    size = still.groupby(block).transform("size")
    return int((still & (size >= PAUSE)).sum())


# ------------------------------------------------------------------ reading: files -> per-second tables
def read_files(sid, folder):
    """Every .csv of a known kind -> {kind: [tables]} and the file log."""
    paths = sorted(p for p in folder.rglob("*.csv") if p.is_file())
    data, log = {k: [] for k, _ in KINDS}, []
    for n, path in enumerate(paths, 1):
        progress(sid, n, len(paths), ".csv file")
        kind = kind_of(path)
        if kind:
            data[kind].append(pd.read_csv(path, dtype=str, encoding="utf-8-sig"))
        log.append({"study_id": sid, "source": SOURCE, "file_id": f"{sid}_fitbit{n}_{kind or 'other'}",
                    "file_hash": file_hash(path), "file_type": kind, "status": "used" if kind else "not used"})
    return data, log


def per_second(cat):
    """live_pace (steps, distance), GPS (position, step length), HR and steps per minute, in local time."""
    num = pd.to_numeric
    lp = cat["live_pace"]
    if lp is not None:
        lp = pd.DataFrame({"t": local(lp.timestamp), "steps": num(lp.steps, errors="coerce"),
                           "dist_m": num(lp["distance millimeters"], errors="coerce") / 1000})
        lp = lp.dropna(subset=["t"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    gps = cat["gps"]
    if gps is not None:
        gps = pd.DataFrame({"t": local(gps.timestamp), "lat": num(gps.latitude, errors="coerce"),
                            "lon": num(gps.longitude, errors="coerce")})
        gps = gps.dropna().drop_duplicates("t").sort_values("t").reset_index(drop=True)
        gps["step_m"] = np.r_[0, haversine(gps.lat.values, gps.lon.values)]
        gps.loc[gps.t.diff() > GAP, "step_m"] = 0
    hr = pd.Series(dtype=float)
    if cat["heart_rate"] is not None:
        h = cat["heart_rate"]
        hr = pd.Series(num(h["beats per minute"], errors="coerce").values, index=local(h.timestamp).values)
        hr = hr[hr > 0].dropna().sort_index()
        hr = hr[~hr.index.duplicated()]
    steps_min = pd.Series(dtype=float)
    if cat["steps"] is not None:
        s = cat["steps"]
        steps_min = pd.Series(num(s.steps, errors="coerce").values, index=local(s.timestamp).values).sort_index()
        steps_min = steps_min[~steps_min.index.duplicated()]          # a file present twice is not counted twice
    return lp, gps, hr, steps_min


# ------------------------------------------------------------------ trainings: candidates -> runs
def recorded_exercises(ex, lp, gps):
    """Candidates from UserExercises: runs of >= MIN_EXERCISE s; MANUAL repeats of a recorded run left out."""
    num = pd.to_numeric
    cands = []
    ex = ex.drop_duplicates("exercise_id") if "exercise_id" in ex else ex
    ex = ex[ex.activity_name.str.contains("run", case=False, na=False)]
    start, end = local(ex.exercise_start), local(ex.exercise_end)
    dist = (num(ex.get("tracker_total_distance_mm"), errors="coerce")
            .fillna(num(ex.get("manually_logged_total_distance_mm"), errors="coerce")) / 1000)
    for i, st, en, d in zip(ex.index, start, end, dist):
        if pd.isna(st) or pd.isna(en) or (en - st).total_seconds() < MIN_EXERCISE:
            continue                                              # accidental start / stop
        if pd.isna(d) or d == 0:                                  # e.g. auto-detected: from the data
            d = window_sum(lp, "dist_m", st, en) or window_sum(gps, "step_m", st, en) or np.nan
        cands.append({"start": st, "end": en, "dist_m": d, "how": "exercise", "checked": True, "i": i,
                      "activity": ex.at[i, "activity_name"], "log_type": ex.at[i, "log_type"],
                      "hr_mean": num(ex.at[i, "tracker_avg_heart_rate"], errors="coerce"),
                      "hr_max": num(ex.at[i, "tracker_peak_heart_rate"], errors="coerce")})
    # a MANUAL entry that repeats a recorded run (starts within duplicate_tolerance, 120 s) is left out
    rec = [c for c in cands if c["log_type"] != "MANUAL"]
    tol = pd.Timedelta(seconds=PROC["duplicate_tolerance"])
    return [c for c in cands if c["log_type"] != "MANUAL" or not any(abs(c["start"] - r["start"]) <= tol for r in rec)]


def candidates(cat, lp, gps, steps_min):
    """Possible runs: the recorded exercises, else blocks of live_pace and GPS data (fallback)."""
    cands = []
    ex = cat["exercises"]
    if ex is not None:
        cands = recorded_exercises(ex, lp, gps)
    elif lp is not None and len(lp):
        for _, b in lp.groupby(blocks(lp.t)):
            cands.append({"start": b.t.iloc[0], "end": b.t.iloc[-1], "dist_m": b.dist_m.sum(),
                          "steps": b.steps.sum(), "how": "live_pace"})
    if ex is None and gps is not None and len(gps):
        for _, b in gps.groupby(blocks(gps.t)):
            st, en = b.t.iloc[0], b.t.iloc[-1]
            if any(c["start"] <= en and c["end"] >= st for c in cands):
                continue
            stp = steps_min[(steps_min.index >= st.floor("min")) & (steps_min.index <= en)].sum()
            cands.append({"start": st, "end": en, "dist_m": b.step_m.sum(), "steps": stp, "how": "gps"})
    return cands


def runs(sid, cands, lp, gps, hr):
    """Candidates that are runs -> sessions and per-second streams (HR, speed, position)."""
    sessions, streams = [], []
    for c in sorted(cands, key=lambda c: c["start"]):
        dur = (c["end"] - c["start"]).total_seconds()
        if not c.get("checked"):                                      # fallback: is it a run?
            speed = c["dist_m"] / dur if dur else 0
            cadence = c["steps"] / (dur / 60) if dur else 0
            if dur < MIN_DURATION or not SPEED[0] <= speed <= SPEED[1] or cadence < MIN_CADENCE:
                continue                                              # not a run (short / walk / cycle)
        st, en = c["start"], c["end"]
        dur -= paused(lp, st, en)                                     # moving time: pauses left out
        sess_id = f"{sid}_fitbit_{st:%Y%m%d%H%M%S}"
        h = hr[(hr.index >= st) & (hr.index <= en)]
        x = pd.DataFrame(index=pd.date_range(st, en, freq="s"))
        x["hr"] = h.reindex(x.index)
        b = lp[(lp.t >= st) & (lp.t <= en)].set_index("t") if lp is not None else pd.DataFrame()
        if len(b):
            x["speed_ms"] = (b.dist_m / b.index.to_series().diff().dt.total_seconds().clip(lower=1)).reindex(x.index)
        g = gps[(gps.t >= st) & (gps.t <= en)].set_index("t") if gps is not None else None
        if g is not None and len(g):
            x[["lat", "lon"]] = g[["lat", "lon"]].reindex(x.index)
            if "speed_ms" not in x:
                x["speed_ms"] = (g.step_m / g.index.to_series().diff().dt.total_seconds()).reindex(x.index)
        x = x.reindex(columns=["hr", "speed_ms", "lat", "lon"]).dropna(how="all")
        start_pos = g.iloc[0] if g is not None and len(g) else None
        sessions.append({"study_id": sid, "session_id": sess_id, "source": SOURCE,
                         "file_id": f"{sid}_fitbit_{c['how']}", "activity": c.get("activity", "running"), "start": st,
                         "start_utc": pd.Timestamp(st).tz_localize(TZ, ambiguous="NaT").tz_convert("UTC").tz_localize(None),
                         "distance_km": c["dist_m"] / 1000, "duration_s": dur,
                         "pace_s_km": dur / (c["dist_m"] / 1000) if c["dist_m"] else np.nan,
                         "hr_mean": c.get("hr_mean") if pd.notna(c.get("hr_mean")) else (h.mean() if len(h) else np.nan),
                         "hr_max": c.get("hr_max") if pd.notna(c.get("hr_max")) else (h.max() if len(h) else np.nan),
                         "lat": start_pos.lat if start_pos is not None else np.nan,
                         "lon": start_pos.lon if start_pos is not None else np.nan})
        streams.append(pd.DataFrame({"session_id": sess_id, "t": x.index,
                                     "elapsed_s": (x.index - st).total_seconds(), "hr": x.hr.values,
                                     "speed_ms": x.speed_ms.values, "lat": x.lat.values, "lon": x.lon.values,
                                     "temp_c": np.nan}))
    return sessions, streams


# ------------------------------------------------------------------ daily values: steps, resting HR, HRV
def resting_hr(sid, hr):
    """Per day with >= MIN_DAY_HOURS of HR: the lowest 30-min mean."""
    rows = []
    for day, h in hr.groupby(hr.index.normalize()):
        if h.index[-1] - h.index[0] < pd.Timedelta(hours=MIN_DAY_HOURS):
            continue                                                  # part of a day only (e.g. file edge)
        m = h.rolling("30min", min_periods=5).mean()
        if m.notna().any():
            rows.append((day, round(m.min(), 1)))
    return pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": f"{sid}_fitbit_heart_rate",
                         "metric": "resting_hr", "start": [r[0] for r in rows], "end": [r[0] for r in rows],
                         "value": [r[1] for r in rows], "unit": "bpm",
                         "note": "lowest 30-min mean of all-day HR that day"})


def nightly_hrv(cat, details):
    """RMSSD per night: Fitbit's daily summary, else the mean of the 5-min values (night -> wake-up date)."""
    num = pd.to_numeric
    nights = pd.Series(dtype=float)
    if cat["hrv_summary"] is not None:
        s = cat["hrv_summary"]
        nights = pd.Series(num(s.rmssd, errors="coerce").values, index=local(s.timestamp).dt.normalize().values)
        nights = nights.dropna().groupby(level=0).mean()
    detail = []
    for s in details:                                                  # two formats: read each file on its own
        col = "rmssd" if "rmssd" in s else next(c for c in s.columns if "root mean square" in c)
        t5 = local(s.timestamp)
        detail.append(pd.Series(num(s[col], errors="coerce").values,
                                index=(t5 + pd.Timedelta(hours=12)).dt.normalize().values))   # night -> wake-up date
    if detail:
        v = pd.concat(detail)
        v = v[v > 0].dropna().groupby(level=0).mean()
        nights = pd.concat([nights, v[~v.index.isin(nights.index)]]).sort_index()
    return nights


def daily_values(sid, steps_min, hr, nights):
    """Steps per day, resting HR per day and nightly HRV as daily rows."""
    daily = []
    if len(steps_min):
        day = steps_min.groupby(steps_min.index.normalize()).sum()
        daily.append(pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": f"{sid}_fitbit_steps",
                                   "metric": "steps", "start": day.index, "end": day.index, "value": day.values,
                                   "unit": "steps (day total)", "note": "sum of steps per minute"}))
    if len(hr):
        daily.append(resting_hr(sid, hr))
    if len(nights):
        daily.append(pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": f"{sid}_fitbit_hrv",
                                   "metric": "hrv_rmssd", "start": nights.index, "end": nights.index,
                                   "value": nights.values.round(1), "unit": "ms",
                                   "note": "nightly RMSSD (Fitbit daily summary, else mean of 5-min values)"}))
    return daily


# ------------------------------------------------------------------ the reader
def parse(sid, folder):
    """One participant's Fitbit folder -> sessions, streams, daily values and the file log."""
    data, log = read_files(sid, folder)
    cat = {k: pd.concat(v, ignore_index=True) if v else None for k, v in data.items() if k != "hrv_detail"}
    lp, gps, hr, steps_min = per_second(cat)
    sessions, streams = runs(sid, candidates(cat, lp, gps, steps_min), lp, gps, hr)
    daily = daily_values(sid, steps_min, hr, nightly_hrv(cat, data["hrv_detail"]))
    return {"sessions": pd.DataFrame(sessions).reindex(columns=SESSION_COLS) if sessions else empty(SESSION_COLS),
            "streams": pd.concat(streams, ignore_index=True) if streams else empty(STREAM_COLS),
            "daily": pd.concat(daily, ignore_index=True).reindex(columns=DAILY_COLS) if daily else empty(DAILY_COLS),
            "log": pd.DataFrame(log)}
