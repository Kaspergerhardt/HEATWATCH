"""Strava export: activities.csv + one .gpx per activity (also .gpx.gz / .fit / .fit.gz).
Columns are read by position instead of name:

  activities.csv  0 activity ID, 1 date (UTC: '12 apr 2026, 08:10:34' / 'Dec 19, 2025, 2:27:28 AM'),
                  3 type (running only), 5 elapsed time (s), 6 distance (km), 12 file name
                  ('activities/18079067975.gpx'), 16 moving time (s), 17 distance (m)
  .gpx / .fit     1 point per s: time, lat, lon
Coupling: the file name in column 12 = the name of the .gpx file;
the date script stores 'activities/8218444672.fit.gz' as 'activities/8218444672.gpx', so both work.
profile.csv (optional): only gender and date of birth if present
"""
import gzip
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ...settings import PROC
from .common import (DEG, INFO_COLS, MONTHS, SESSION_COLS, STREAM_COLS, age_on, empty, file_hash, haversine, num,
                     progress, read_table, utc_to_local)
from .garmin_connect import is_run

SOURCE = "strava"
TRACK = (".gpx", ".gpx.gz", ".fit", ".fit.gz")


# ------------------------------------------------------------------ helpers: file type, dates, GPS tracks, profile
def track_ext(path):
    """Track extension of a file (.gpx, .gpx.gz, .fit, .fit.gz) or None."""
    return next((e for e in TRACK if path.name.lower().endswith(e)), None)


def strava_date(v):
    """'12 apr 2026, 08:10:34' / 'Dec 19, 2025, 2:27:28 AM' / '2026-04-12 08:10:34' -> UTC Timestamp."""
    v = str(v).strip()
    if re.match(r"\d{4}-\d{2}-\d{2}", v):
        return pd.to_datetime(v, errors="coerce")
    low = v.lower().replace(",", " ").replace(".", " ")
    month = next((MONTHS[w] for w in re.findall(r"[a-zéû]+", low) if w in MONTHS), None)
    t = re.search(r"(\d{1,2}):(\d{2}):(\d{2})\s*([ap]m)?", low)
    nums = [int(n) for n in re.findall(r"\b\d+\b", low[:t.start()] if t else low)]
    year = next((n for n in nums if n > 31), None)
    day = next((n for n in nums if n <= 31), None)
    if not (month and day and year):
        return pd.NaT
    h, mi, s = (int(t[1]), int(t[2]), int(t[3])) if t else (0, 0, 0)
    if t and t[4]:
        h = h % 12 + (12 if t[4] == "pm" else 0)
    return pd.Timestamp(year, month, day, h, mi, s)


def read_track(path):
    """GPS points of a .gpx / .fit (optionally .gz): t (UTC), lat, lon. Heart rate is never read."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    if ".gpx" in path.name.lower():
        with opener(path, "rb") as f:
            root = ET.parse(f).getroot()
        ns = {"g": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {"g": ""}
        pre = "g:" if ns["g"] else ""
        kind = (root.findtext(f".//{pre}trk/{pre}type", "", ns) or "").lower()
        pts = [(p.findtext(f"{pre}time", None, ns), float(p.get("lat")), float(p.get("lon")))
               for p in root.iter(f"{{{ns['g']}}}trkpt" if ns["g"] else "trkpt")]
    else:
        import fitdecode
        pts, kind = [], ""
        with opener(path, "rb") as f, fitdecode.FitReader(f) as reader:
            for fr in reader:
                if isinstance(fr, fitdecode.FitDataMessage) and fr.name == "record":
                    v = {fl.name: fl.value for fl in fr.fields if fl.name in ("timestamp", "position_lat", "position_long")}
                    if v.get("position_lat") is not None and v.get("position_long") is not None:
                        pts.append((v["timestamp"], v["position_lat"] * DEG, v["position_long"] * DEG))
    g = pd.DataFrame(pts, columns=["t", "lat", "lon"])
    g["t"] = pd.to_datetime(g.t, utc=True, errors="coerce").dt.tz_localize(None).dt.floor("s")
    return g.dropna(subset=["t"]).sort_values("t").drop_duplicates("t").reset_index(drop=True), kind


def with_speed(g):
    """Speed (m/s) between consecutive GPS points (haversine)."""
    step = np.r_[np.nan, haversine(g.lat.values, g.lon.values)]
    dt = g.t.diff().dt.total_seconds().values
    return g.assign(step_m=step, speed_ms=np.where(dt > 0, step / dt, np.nan))


def profile(sid, fid, df, ref):
    """Gender (and age if a date of birth is there) from profile.csv; nothing else is read."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    gender_col = next((cols[c] for c in ("sex", "geslacht", "sexe", "gender") if c in cols), None)
    dob_col = next((cols[c] for c in cols if "birth" in c or "geboorte" in c or "naissance" in c), None)
    if not len(df) or not (gender_col or dob_col):
        return empty(INFO_COLS)
    gender = str(df.iloc[0][gender_col]).strip().lower()[:1] if gender_col else ""
    gender = {"m": "male", "f": "female", "v": "female", "w": "female"}.get(gender)
    dob = pd.to_datetime(df.iloc[0][dob_col], errors="coerce") if dob_col else pd.NaT
    return pd.DataFrame([{"study_id": sid, "source": SOURCE, "file_id": fid, "date": ref, "age": age_on(dob, ref),
                          "gender": gender}])


# ------------------------------------------------------------------ reading: files, activities.csv, GPS tracks
def read_files(sid, folder):
    """GPS tracks {file number: path} and the .csv files [(path, table, kind)]."""
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and (p.suffix.lower() == ".csv" or track_ext(p)))
    tracks, csvs = {}, []
    for n, path in enumerate(paths, 1):
        progress(sid, n, len(paths), "file")
        if track_ext(path):
            tracks[path.name.split(".")[0]] = path              # '18079067975.gpx' -> '18079067975'
        else:
            df = read_table(path)
            kind = "activities" if df.shape[1] >= 18 else "profile" if path.stem.lower() == "profile" else None
            csvs.append((path, df, kind))
    return tracks, csvs


def activities(sid, csvs, log):
    """activities.csv -> the runs, the track numbers of all activities (runs or not) and the profile files."""
    rows, profiles = [], []
    for i, (path, df, kind) in enumerate(csvs, 1):
        fid = f"{sid}_strava{i}_{kind or 'other'}"
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": kind, "n_rows": len(df), "status": "used" if kind else "not used"})
        if kind == "activities":
            c = lambda j: df.iloc[:, j]
            moving, elapsed = c(16).map(num), c(5).map(num)
            dist_km = (c(17).map(num) / 1000).fillna(c(6).map(num))
            rows.append(pd.DataFrame({
                "file_id": fid, "source_row": df.index + 2, "activity": c(3), "start_utc": c(1).map(strava_date),
                "distance_km": dist_km, "duration_s": moving.where(moving > 0, elapsed),
                "track": c(12).str.split("/").str[-1].str.split(".").str[0].replace("", None)}))   # None = no file
        elif kind == "profile":
            profiles.append((fid, df))
    acts = pd.concat(rows, ignore_index=True) if rows else \
        pd.DataFrame(columns=["file_id", "source_row", "activity", "start_utc", "distance_km", "duration_s", "track"])
    acts = acts.dropna(subset=["start_utc"]).drop_duplicates("start_utc")
    in_csv = set(acts.track)                                     # also the non-runs: their files are skipped
    acts = acts.loc[acts.activity.map(is_run).astype(bool).values]
    return acts, in_csv, profiles


def track_runs(sid, tracks, acts, in_csv, log):
    """Every GPS track: the run it belongs to in activities.csv, or a run of its own -> sessions and streams."""
    sessions, streams = [], []
    for key, path in tracks.items():
        fid = f"{sid}_strava_track{len(log) + 1}"
        g, kind = read_track(path)
        row = acts[acts.track == key]
        status = "used"
        if key in in_csv and not len(row):
            status = "not used: not a run in activities.csv"
        elif not len(g):
            status = "not used: no GPS points"
        elif not len(row):                                       # not in activities.csv: run of its own
            st = g.t.iloc[0]
            if kind and not is_run(kind):
                status = f"not used: type '{kind}'"
            elif len(acts) and (abs(acts.start_utc - st) <= pd.Timedelta(seconds=PROC["duplicate_tolerance"])).any():
                status = "not used: same run as a row in activities.csv"
            else:
                row = pd.DataFrame([{"file_id": fid, "source_row": None, "activity": kind or "running (gpx)",
                                     "start_utc": st, "distance_km": with_speed(g).step_m.sum() / 1000,
                                     "duration_s": (g.t.iloc[-1] - st).total_seconds(), "track": key}])
                status = "used: run not in activities.csv (values from GPS)"
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": track_ext(path).lstrip("."), "n_rows": len(g), "status": status})
        if status.startswith("used"):
            r = row.iloc[0]
            sessions.append({**r.to_dict(), "lat": g.lat.iloc[0], "lon": g.lon.iloc[0]})
            x = with_speed(g)
            streams.append(pd.DataFrame({"session_key": key, "t_utc": x.t, "elapsed_s": (x.t - r.start_utc).dt.total_seconds(),
                                         "hr": np.nan, "speed_ms": x.speed_ms, "lat": x.lat, "lon": x.lon, "temp_c": np.nan}))
    return sessions, streams


def tables(sid, sessions, streams, acts):
    """Sessions and streams in the common columns, local time; runs without a GPS file added."""
    linked = {s["track"] for s in sessions}
    sessions += acts[~acts.track.isin(linked)].to_dict("records")       # runs without a GPS file (manual / missing)
    s = pd.DataFrame(sessions).reindex(columns=list(acts.columns) + ["lat", "lon"]) if sessions else empty(SESSION_COLS)
    if len(s):
        s["start_utc"] = pd.to_datetime(s.start_utc)
        s["start"] = utc_to_local(s.start_utc)
        s = s.assign(study_id=sid, source=SOURCE, session_id=sid + "_strava_" + s.start.dt.strftime("%Y%m%d%H%M%S"),
                     pace_s_km=s.duration_s / s.distance_km, hr_mean=np.nan, hr_max=np.nan, location=None)
        s = s.sort_values("start").reindex(columns=SESSION_COLS + ["track"])
    st = pd.concat(streams, ignore_index=True) if streams else empty(STREAM_COLS)
    if len(streams):
        st["session_id"] = st.session_key.map(s.dropna(subset=["track"]).set_index("track").session_id)
        st["t"] = utc_to_local(st.t_utc)
        st = st[STREAM_COLS]
    return s.drop(columns="track", errors="ignore"), st


# ------------------------------------------------------------------ the reader
def parse(sid, folder):
    """One participant's Strava folder -> sessions, streams (GPS only), gender / age and the file log."""
    log = []
    tracks, csvs = read_files(sid, folder)
    acts, in_csv, profiles = activities(sid, csvs, log)
    sessions, streams = track_runs(sid, tracks, acts, in_csv, log)
    s, st = tables(sid, sessions, streams, acts)
    ref = s.start.max() if len(s) else pd.NaT
    info = pd.concat([profile(sid, fid, df, ref) for fid, df in profiles]) if profiles else empty(INFO_COLS)
    return {"sessions": s, "streams": st, "info": info, "log": pd.DataFrame(log)}
