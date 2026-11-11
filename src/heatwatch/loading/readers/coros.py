"""Coros export as .csv: the .fit messages of all activities, one ';'-separated file per message type.

  session.csv  1 row per activity: sport, start_time (UTC), timestamp (= end, UTC),
               total_timer_time (s), total_distance (m), avg_heart_rate, max_heart_rate,
               enhanced_avg_speed (m/s), avg_temperature, sub_sport
  record.csv   1 row per s for all activities: timestamp, heart_rate, enhanced_speed (or speed),
               position_lat / position_long (semicircles). A record belongs to the run whose
               start_time <= timestamp <= end.
Coros .fit files themselves are read by garmin_fit.py.
"""
import numpy as np
import pandas as pd

from .common import DEG, SESSION_COLS, STREAM_COLS, empty, file_hash, is_indoor, progress, utc_to_local

SOURCE = "coros"
KINDS = ("session", "record", "activity", "event", "lap", "device_info", "file_id",
         "developer_data_id", "field_description")


# ------------------------------------------------------------------ helpers: file kind, time, columns
def kind_of(path):
    """'session.csv' / 'a4461c3c-session.csv' -> 'session'."""
    stem = path.stem.lower()
    return next((k for k in sorted(KINDS, key=len, reverse=True) if stem == k or stem.endswith("-" + k)), None)


def utc(col):
    """Coros time column -> UTC time without a zone."""
    return pd.to_datetime(col, utc=True, errors="coerce").dt.tz_localize(None)


def num(df, *names):
    """First column that exists, as numbers (NaN if none)."""
    c = next((n for n in names if n in df.columns), None)
    return pd.to_numeric(df[c], errors="coerce") if c else pd.Series(np.nan, index=df.index)


# ------------------------------------------------------------------ reading: files, runs, records
def read_files(sid, folder):
    """session.csv and record.csv tables {kind: [tables]} and the file log."""
    files = sorted(p for p in folder.rglob("*.csv") if kind_of(p))
    tables, log = {"session": [], "record": []}, []
    for n, path in enumerate(files, 1):
        progress(sid, n, len(files), ".csv file")
        kind = kind_of(path)
        fid = f"{sid}_coros{n}_{kind}"
        if kind in tables:
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
            tables[kind].append(df)
            status = "used"
        else:
            df, status = None, "not used"
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": kind, "n_rows": None if df is None else len(df), "status": status})
    return tables, log


def runs(sid, sessions):
    """session rows -> one row per run (running only, each start once)."""
    s = pd.concat(sessions, ignore_index=True)
    s = s.assign(start_utc=utc(s.start_time), end_utc=utc(s.timestamp)).dropna(subset=["start_utc"])
    s = s.drop_duplicates("start_utc")
    s = s.loc[s.sport.astype(str).str.lower().eq("running").values].sort_values("start_utc")
    speed = num(s, "enhanced_avg_speed", "avg_speed")
    sess = pd.DataFrame({
        "study_id": sid, "source": SOURCE, "file_id": f"{sid}_coros_session", "source_row": s.index + 2,
        "activity": "running", "start_utc": s.start_utc,
        "start": utc_to_local(s.start_utc),
        "distance_km": num(s, "total_distance") / 1000, "duration_s": num(s, "total_timer_time"),
        "pace_s_km": 1000 / speed.where(speed > 0), "hr_mean": num(s, "avg_heart_rate"),
        "hr_max": num(s, "max_heart_rate"), "temp_c": num(s, "avg_temperature"), "end_utc": s.end_utc,
        "indoor": s.get("sub_sport", pd.Series("", index=s.index)).map(is_indoor)})
    sess["session_id"] = sid + "_coros_" + sess.start.dt.strftime("%Y%m%d%H%M%S")
    return sess


def records(sess, recs):
    """record rows -> the run they fall in (start <= time <= end) as streams; start position added to sess."""
    r = pd.concat(recs, ignore_index=True)
    r = pd.DataFrame({"t_utc": utc(r.timestamp), "hr": num(r, "heart_rate"),
                      "speed_ms": num(r, "enhanced_speed", "speed"),
                      "lat": num(r, "position_lat") * DEG, "lon": num(r, "position_long") * DEG})
    r = r.dropna(subset=["t_utc"]).drop_duplicates("t_utc").sort_values("t_utc")
    r = pd.merge_asof(r, sess[["start_utc", "end_utc", "session_id"]].sort_values("start_utc"),
                      left_on="t_utc", right_on="start_utc", direction="backward")
    r = r[r.session_id.notna() & (r.t_utc <= r.end_utc)]
    st = pd.DataFrame({"session_id": r.session_id,
                       "t": utc_to_local(r.t_utc),
                       "elapsed_s": (r.t_utc - r.start_utc).dt.total_seconds(), "hr": r.hr,
                       "speed_ms": r.speed_ms, "lat": r.lat, "lon": r.lon, "temp_c": np.nan})
    first = st.dropna(subset=["lat"]).groupby("session_id")[["lat", "lon"]].first()   # start position
    sess["lat"] = sess.session_id.map(first.lat)
    sess["lon"] = sess.session_id.map(first.lon)
    return st


# ------------------------------------------------------------------ the reader
def parse(sid, folder):
    """One participant's Coros .csv folder -> sessions, streams and the file log."""
    tables, log = read_files(sid, folder)
    if not tables["session"]:
        return {"sessions": empty(SESSION_COLS), "streams": empty(STREAM_COLS), "log": pd.DataFrame(log)}
    sess = runs(sid, tables["session"])
    st = records(sess, tables["record"]) if tables["record"] else empty(STREAM_COLS)
    sess = sess.reindex(columns=SESSION_COLS)
    return {"sessions": sess, "streams": st, "log": pd.DataFrame(log)}
