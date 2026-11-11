"""Apple Health export, converted to one ';'-separated .csv per data type + route_*.gpx files.

  Workout.csv          running workouts (meta:HKIndoorWorkout 1 -> location "indoor"): startDate, duration, DistanceWalkingRunning_sum,
                       HeartRate_average/_maximum, meta:HKWeatherTemperature (degF), meta:HKWeatherHumidity
  HeartRate.csv        HR samples
  route_*.gpx          1 point per s: time, lat, lon, speed
  RestingHeartRate.csv 1 value per day                                        
  HeartRateVariabilitySDNN.csv  a few samples per day
  StepCount.csv        samples from Watch + iPhone
  Me.csv               date of birth, biological sex 
Dates carry the offset at export time (e.g. '+0200' in January)
"""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ...settings import PROC
from .common import (DAILY_COLS, INFO_COLS, SESSION_COLS, STREAM_COLS, age_on, empty, file_hash, is_indoor,
                     progress)

SOURCE = "apple"
TZ = PROC["local_timezone"]
FILES = ["Workout", "HeartRate", "RestingHeartRate", "HeartRateVariabilitySDNN", "StepCount", "Me"]


# ------------------------------------------------------------------ helpers: time, numbers, GPX routes
def utc(col):
    """Apple date ('2026-01-13 20:49:00 +0100') -> UTC time."""
    return pd.to_datetime(col, format="%Y-%m-%d %H:%M:%S %z", errors="coerce", utc=True)


def local(col):
    """Apple date -> local time without a zone."""
    return utc(col).dt.tz_convert(TZ).dt.tz_localize(None)


def lead_num(v):
    """Number at the start of a text ('68.4 degF' -> 68.4)."""
    m = re.match(r"\s*(-?[\d.]+)", str(v))
    return float(m[1]) if m else np.nan


NAME_MATCH = pd.Timedelta(minutes=10)


def name_time(path):
    """'route_2026-01-13_8.49pm' -> Timestamp 2026-01-13 20:49 (local)."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{1,2})[.:](\d{2})\s*([ap]m)?", path.stem, re.I)
    if not m:
        return pd.NaT
    h = int(m[2]) % 12 + (12 if (m[4] or "").lower() == "pm" else 0) if m[4] else int(m[2])
    return pd.Timestamp(m[1]) + pd.Timedelta(hours=h, minutes=int(m[3]))


def match_route(routes, st, en, st_local, en_local):
    """Index of the GPX route that belongs to this workout, or None."""
    pad = pd.Timedelta(seconds=PROC["duplicate_tolerance"])
    for i, (g, _) in enumerate(routes):                      # 1) GPS times overlap the workout
        t = g.t.dropna()
        if len(t) and t.iloc[0] <= en + pad and t.iloc[-1] >= st - pad:
            return i
    for i, (_, nt) in enumerate(routes):                     # 2) time in the file name
        if pd.notna(nt) and min(abs(nt - st_local), abs(nt - en_local)) <= NAME_MATCH:
            return i
    return None


def read_gpx(path):
    """route_*.gpx -> t (UTC), lat, lon, speed per point."""
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    pts = [(p.findtext("g:time", namespaces=ns), float(p.get("lat")), float(p.get("lon")),
            float(p.findtext("g:extensions/g:speed", "nan", ns)))
           for p in ET.parse(path).getroot().iterfind(".//g:trkpt", ns)]
    g = pd.DataFrame(pts, columns=["t", "lat", "lon", "speed_ms"])
    g["t"] = pd.to_datetime(g.t, utc=True).dt.floor("s")
    return g


def daily(sid, fid, metric, start, value, unit, note, rows=None):
    """Rows of the daily table (resting HR, HRV, steps)."""
    return pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": fid, "source_row": rows, "metric": metric,
                         "start": start, "end": start, "value": value, "unit": unit, "note": note})


# ------------------------------------------------------------------ reading: files, workouts, daily values, Me
def read_files(sid, folder):
    """The known .csv files {name: (table, file_id)}, the GPX routes, the file log and the log rows of the routes."""
    found, gpx, log, gpx_log = {}, [], [], []
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in (".csv", ".gpx"))
    for n, path in enumerate(paths, 1):
        progress(sid, n, len(paths), "file")
        name = path.stem.split("-")[-1]               # tolerates prefixes like 'e490e2e4-Me'
        kind = "gpx" if path.suffix.lower() == ".gpx" else (name if name in FILES else None)
        fid = f"{sid}_apple_{kind or 'other'}" + (f"{len(gpx) + 1}" if kind == "gpx" else "")
        if kind == "gpx":
            gpx.append((read_gpx(path), name_time(path)))
            gpx_log.append(len(log))                     # status filled in once matched
        elif kind:
            found[kind] = (pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig"), fid)
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": kind, "status": "used" if kind else "not used"})
    return found, gpx, log, gpx_log


def workouts(sid, found, gpx, log, gpx_log):
    """Running workouts -> sessions and streams (watch HR + the matched GPX route); route status in the log."""
    w, fid = found["Workout"]
    w = w.loc[w.workoutActivityType.str.contains("Running", na=False).astype(bool).values]
    start, end = utc(w.startDate), utc(w.endDate)
    dur = pd.to_numeric(w.duration, errors="coerce") * 60
    dist = pd.to_numeric(w.get("DistanceWalkingRunning_sum"), errors="coerce")
    sessions = pd.DataFrame({
        "study_id": sid, "session_id": sid + "_apple_" + start.dt.tz_convert(TZ).dt.strftime("%Y%m%d%H%M%S"), "source": SOURCE,
        "lat": np.nan, "lon": np.nan,
        "file_id": fid, "source_row": w.index + 2, "activity": "running",
        "start": start.dt.tz_convert(TZ).dt.tz_localize(None), "start_utc": start.dt.tz_localize(None),
        "distance_km": dist, "duration_s": dur, "pace_s_km": dur / dist,
        "hr_mean": pd.to_numeric(w.get("HeartRate_average"), errors="coerce"),
        "hr_max": pd.to_numeric(w.get("HeartRate_maximum"), errors="coerce"),
        "weather_temp_c": (w.get("meta:HKWeatherTemperature", pd.Series(index=w.index)).map(lead_num) - 32) / 1.8,
        "weather_humidity_pct": w.get("meta:HKWeatherHumidity", pd.Series(index=w.index)).map(lead_num) / 100,
        "indoor": w.get("meta:HKIndoorWorkout", pd.Series("", index=w.index)).map(is_indoor)})
    hr = found.get("HeartRate", (empty(["startDate", "value", "device"]), None))[0]
    hr = pd.DataFrame({"t": utc(hr.startDate).dt.floor("s"), "hr": pd.to_numeric(hr.value, errors="coerce"),
                       "watch": hr.get("device", hr.get("sourceName", "")).str.contains("Watch", na=False)})
    hr = hr.loc[hr.watch.astype(bool).values].groupby("t").hr.mean().reset_index()       # HR from the watch only
    parts, used = [], {}
    for i, (sess, st, en) in enumerate(zip(sessions.itertuples(), start, end)):
        k = match_route(gpx, st, en, st.tz_convert(TZ).tz_localize(None), en.tz_convert(TZ).tz_localize(None))
        route = gpx[k][0] if k is not None else None
        if route is not None and len(route):
            used[k] = sess.session_id
            sessions.iloc[i, sessions.columns.get_loc("lat")] = route.lat.iloc[0]   # start position
            sessions.iloc[i, sessions.columns.get_loc("lon")] = route.lon.iloc[0]
        h = hr[(hr.t >= st) & (hr.t <= en)]
        x = h if route is None else route[(route.t >= st) & (route.t <= en)].merge(h, on="t", how="outer")
        if len(x):
            x = x.sort_values("t")
            parts.append(pd.DataFrame({"session_id": sess.session_id,
                                       "t": x.t.dt.tz_convert(TZ).dt.tz_localize(None),
                                       "elapsed_s": (x.t - st).dt.total_seconds(), "hr": x.hr,
                                       "speed_ms": x.get("speed_ms"), "lat": x.get("lat"),
                                       "lon": x.get("lon"), "temp_c": np.nan}))
    for j, row in enumerate(gpx_log):
        log[row]["status"] = f"used: route of {used[j]}" if j in used else "not used: no matching workout"
    return sessions, (pd.concat(parts) if parts else empty(STREAM_COLS))


def daily_values(sid, found):
    """Resting HR per day, HRV samples and steps per day (per hour the larger of Watch and iPhone)."""
    rows = []
    if "RestingHeartRate" in found:
        df, fid = found["RestingHeartRate"]
        rows.append(daily(sid, fid, "resting_hr", local(df.startDate).dt.normalize(),
                          pd.to_numeric(df.value, errors="coerce"), "bpm", "1 per day", df.index + 2))
    if "HeartRateVariabilitySDNN" in found:
        df, fid = found["HeartRateVariabilitySDNN"]
        rows.append(daily(sid, fid, "hrv_sdnn", local(df.startDate), pd.to_numeric(df.value, errors="coerce"),
                          "ms", "single sample", df.index + 2))
    if "StepCount" in found:
        df, fid = found["StepCount"]
        x = pd.DataFrame({"hour": local(df.startDate).dt.floor("h"), "value": pd.to_numeric(df.value, errors="coerce"),
                          "dev": np.where(df.get("device", df.sourceName).str.contains("Watch", na=False), "w", "p")})
        # Watch and iPhone count the same steps when both are carried: per hour take the larger one
        hourly = x.groupby(["hour", "dev"]).value.sum().unstack(fill_value=0).max(axis=1)
        day = hourly.groupby(hourly.index.normalize()).sum()
        rows.append(daily(sid, fid, "steps", pd.Series(day.index), day.values, "steps (day total)",
                          "per hour max(Watch, iPhone), summed per day"))
    return rows


def me(sid, found, sessions):
    """Me.csv -> age on the last run (date of birth not stored) and gender."""
    me, fid = found["Me"]
    dob = pd.to_datetime(me.iloc[0].get("HKCharacteristicTypeIdentifierDateOfBirth"), errors="coerce")
    ref = sessions.start.max() if len(sessions) else pd.NaT
    gender = {"HKBiologicalSexMale": "male", "HKBiologicalSexFemale": "female"}.get(
        me.iloc[0].get("HKCharacteristicTypeIdentifierBiologicalSex"))
    return pd.DataFrame([{"study_id": sid, "source": SOURCE, "file_id": fid, "date": ref, "age": age_on(dob, ref),
                          "gender": gender}])


# ------------------------------------------------------------------ the reader
def parse(sid, folder):
    """One participant's Apple folder -> sessions, streams, daily values, age / gender and the file log."""
    found, gpx, log, gpx_log = read_files(sid, folder)
    sessions, streams = empty(SESSION_COLS), empty(STREAM_COLS)
    if "Workout" in found:
        sessions, streams = workouts(sid, found, gpx, log, gpx_log)
    rows = daily_values(sid, found)
    info = me(sid, found, sessions) if "Me" in found else empty(INFO_COLS)
    return {"sessions": sessions, "streams": streams, "info": info, "log": pd.DataFrame(log),
            "daily": pd.concat(rows) if rows else empty(DAILY_COLS)}
