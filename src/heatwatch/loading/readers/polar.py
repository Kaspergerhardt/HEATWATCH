"""Polar Flow export (.json).

  training-session_*.json  1 per training: startTime (LOCAL) + timezoneOffsetMinutes, durationMillis,
                           distanceMeters, hrAvg, hrMax, sport id ('1' = running), and per exercise
                           samples every intervalMillis (HEART_RATE, SPEED in km/h, TEMPERATURE) and
                           routes.route.wayPoints (lat, lon, elapsedMillis)
                           physicalInformation: birthday and gender
  activity-*.json          1 per day: summary.stepCount                        
  247ohr_*.json            24/7 HR, 1 file per month: deviceDays[].samples (heartRate, secondsFromDayStart)
"""
import json

import numpy as np
import pandas as pd

from .common import DAILY_COLS, INFO_COLS, SESSION_COLS, STREAM_COLS, age_on, empty, file_hash, progress

SOURCE = "polar"
RUN_SPORT_IDS = {"1"}                       # Polar sport id of running'
REST_WINDOW = "30min"


# ------------------------------------------------------------------ helpers: file kind, numbers, age, gender
def kind_of(path):
    """File name -> training-session, 247ohr, activity, training-target, generic-period or None."""
    n = path.name.lower()
    for k in ("training-session", "247ohr", "activity-", "training-target", "generic-period"):
        if k in n:
            return k.rstrip("-")
    return None


def num(v):
    """Values -> numbers (NaN when not a number)."""
    return pd.to_numeric(pd.Series(v, dtype=object), errors="coerce").values


def gender_of(v):
    """'MALE' / 'FEMALE' -> male / female."""
    v = str(v or "").upper()
    return "male" if v.endswith("MALE") and not v.endswith("FEMALE") else "female" if v.endswith("FEMALE") else None


# ------------------------------------------------------------------ reading: training sessions, 24/7 HR, the reader
def training(sid, fid, d):
    """One training-session file"""
    sport = str(d.get("sport", {}).get("id"))
    if sport not in RUN_SPORT_IDS:
        return f"not used: sport id {sport}"
    start = pd.Timestamp(d["startTime"])                                   # local time
    offset = pd.Timedelta(minutes=float(d.get("timezoneOffsetMinutes") or 0))
    sess_id = f"{sid}_polar_{start:%Y%m%d%H%M%S}"
    dur, dist = float(d.get("durationMillis") or "nan") / 1000, float(d.get("distanceMeters") or "nan")
    sess = {"study_id": sid, "session_id": sess_id, "source": SOURCE, "file_id": fid, "activity": "running",
            "start": start, "start_utc": start - offset, "distance_km": dist / 1000, "duration_s": dur,
            "pace_s_km": dur / (dist / 1000) if dist else np.nan,
            "hr_mean": float(d.get("hrAvg") or "nan"), "hr_max": float(d.get("hrMax") or "nan"),
            "lat": float(d.get("latitude") or "nan"), "lon": float(d.get("longitude") or "nan")}
    parts = []
    for ex in d.get("exercises", []):
        ex_start = pd.Timestamp(ex.get("startTime", d["startTime"]))
        cols = {}
        for smp in ex.get("samples", {}).get("samples", []):
            if smp["type"] in ("HEART_RATE", "SPEED", "TEMPERATURE"):
                step = float(smp.get("intervalMillis", 1000)) / 1000
                v = num(smp["values"])
                cols[smp["type"]] = pd.Series(v, index=np.round(np.arange(len(v)) * step).astype(int))
        x = pd.DataFrame(cols)
        wp = ex.get("routes", {}).get("route", {}).get("wayPoints", [])
        if wp:
            g = pd.DataFrame(wp)
            g = pd.DataFrame({"lat": num(g.latitude), "lon": num(g.longitude)},
                             index=(num(g.elapsedMillis) / 1000).round().astype(int))
            x = x.join(g[~g.index.duplicated()], how="outer")
        if len(x):
            el = x.index.values + (ex_start - start).total_seconds()
            parts.append(pd.DataFrame({"session_id": sess_id, "t": start + pd.to_timedelta(el, unit="s"),
                                       "elapsed_s": el, "hr": x.get("HEART_RATE"),
                                       "speed_ms": x.get("SPEED") / 3.6 if "SPEED" in x else np.nan,
                                       "lat": x.get("lat"), "lon": x.get("lon"),
                                       "temp_c": x.get("TEMPERATURE")}).reset_index(drop=True))
    if parts:
        sess["temp_c"] = pd.concat(parts).temp_c.mean()
    pi = d.get("physicalInformation", {})
    info = {"study_id": sid, "source": SOURCE, "file_id": fid, "date": start,
            "age": age_on(pd.to_datetime(pi.get("birthday"), errors="coerce"), start),
            "gender": gender_of(pi.get("sex"))}
    return sess, (pd.concat(parts) if parts else empty(STREAM_COLS)), info


def resting_hr(sid, fid, d):
    """247ohr file -> resting HR per day"""
    rows = []
    for day in d.get("deviceDays", []):
        s = pd.DataFrame(day.get("samples", []))
        if not len(s):
            continue
        t = pd.Timestamp(day["date"]) + pd.to_timedelta(num(s.secondsFromDayStart), unit="s")
        hr = pd.Series(num(s.heartRate), index=t).dropna().sort_index()
        hr = hr[hr > 0]
        m = hr.rolling(REST_WINDOW, min_periods=5).mean()
        if m.notna().any():
            rows.append((pd.Timestamp(day["date"]), round(m.min(), 1)))
    return pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": fid, "source_row": None,
                         "metric": "resting_hr", "start": [r[0] for r in rows], "end": [r[0] for r in rows],
                         "value": [r[1] for r in rows], "unit": "bpm",
                         "note": "lowest 30-min mean of 24/7 HR that day"})


def parse(sid, folder):
    """One participant's Polar .json files"""
    paths = sorted(p for p in folder.rglob("*.json") if p.is_file())
    sessions, streams, info, daily, log = [], [], [], [], []
    for n, path in enumerate(paths, 1):
        progress(sid, n, len(paths), ".json file")
        kind = kind_of(path)
        fid = f"{sid}_polar{n}_{kind or 'other'}"
        status = "used"
        if kind in ("training-session", "activity", "247ohr"):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
            except (ValueError, UnicodeDecodeError) as e:
                d, status = None, f"not read: {type(e).__name__}"
            if d is not None and kind == "training-session":
                r = training(sid, fid, d)
                if isinstance(r, str):
                    status = r
                else:
                    sessions.append(r[0])
                    streams.append(r[1])
                    info.append(r[2])
            elif d is not None and kind == "activity":
                day = pd.Timestamp(d["date"])
                daily.append(pd.DataFrame([{"study_id": sid, "source": SOURCE, "file_id": fid, "metric": "steps",
                                            "start": day, "end": day,
                                            "value": float(d.get("summary", {}).get("stepCount") or "nan"),
                                            "unit": "steps (day total)", "note": None}]))
                pi = d.get("physicalInformation", {})
                info.append({"study_id": sid, "source": SOURCE, "file_id": fid, "date": day,
                             "age": age_on(pd.to_datetime(pi.get("birthday"), errors="coerce"), day),
                             "gender": gender_of(pi.get("sex"))})
            elif d is not None:
                daily.append(resting_hr(sid, fid, d))
        else:
            status = "not used"
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": kind, "status": status})

    s = pd.DataFrame(sessions).reindex(columns=SESSION_COLS) if sessions else empty(SESSION_COLS)
    d = pd.concat(daily, ignore_index=True).reindex(columns=DAILY_COLS) if daily else empty(DAILY_COLS)
    return {"sessions": s.drop_duplicates("session_id"),
            "streams": pd.concat(streams, ignore_index=True) if streams else empty(STREAM_COLS),
            "daily": d.dropna(subset=["value"]), "info": pd.DataFrame(info, columns=INFO_COLS),
            "log": pd.DataFrame(log)}
