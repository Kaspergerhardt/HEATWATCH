"""Garmin Internal exports: one file per activity. Fields by number:

  session  2 start_time, 6 sub_sport, 3/4 start lat/long, 8 total_timer_time, 9 total_distance,
           16 avg HR, 17 max HR, 124 enhanced_avg_speed, 57/150/58 avg/min/max temperature,
           5 sport
  record   253 timestamp, 0/1 lat/long, 3 heart rate, 73 enhanced_speed, 13 temperature  (1 per s)
  unknown_79  field 1 age, field 4 gender
  activity    local_timestamp - timestamp = UTC offset
"""
import warnings

import fitdecode
import numpy as np
import pandas as pd

from ...settings import PROC
from .common import DEG, INFO_COLS, SESSION_COLS, STREAM_COLS, empty, file_hash, is_indoor, progress

SOURCE = "garmin_fit"
SESSION = {2: "start_utc", 3: "lat", 4: "lon", 8: "duration_s", 9: "distance_m", 16: "hr_mean",
           17: "hr_max", 124: "speed", 57: "temp_c", 5: "sport", 6: "sub_sport"}
RECORD = {253: "t", 0: "lat", 1: "lon", 3: "hr", 73: "speed_ms", 13: "temp_c"}


def fields(msg, wanted):
    """Fields of a message with these field numbers"""
    return {wanted[f.def_num]: f.value for f in msg.fields if f.def_num in wanted and f.value is not None}


def read(path):
    """One .fit file -> session, activity, age / gender message and the records."""
    ses, act, u79, recs = {}, {}, {}, []
    # newer watches write some (unused) fields with an odd size: fitdecode warns but reads them fine
    warnings.filterwarnings("ignore", category=UserWarning, module="fitdecode")
    with fitdecode.FitReader(str(path), check_crc=fitdecode.CrcCheck.DISABLED) as fr:
        for m in fr:
            if not isinstance(m, fitdecode.FitDataMessage):
                continue
            if m.name == "session" and not ses:
                ses = fields(m, SESSION)
            elif m.name == "activity":
                act = {k: m.get_value(k, fallback=None) for k in ("timestamp", "local_timestamp")}
            elif m.global_mesg_num == 79:
                u79 = fields(m, {1: "age", 4: "gender"})
            elif m.name == "record":
                recs.append(fields(m, RECORD))
    return ses, act, u79, recs


def parse(sid, folder):
    """One participant's .fit files"""
    sessions, streams, info, log = [], [], [], []
    fits = sorted(folder.rglob("*.fit"))
    for n, path in enumerate(fits, 1):
        progress(sid, n, len(fits), ".fit file")
        entry = {"study_id": sid, "source": SOURCE, "file_hash": file_hash(path), "file_type": "fit"}
        try:
            ses, act, u79, recs = read(path)
        except Exception as e:  # damaged file
            log.append({**entry, "status": f"not read: {type(e).__name__}"})
            continue
        if ses.get("sport") != "running":
            log.append({**entry, "status": f"not used: sport = {ses.get('sport')}"})
            continue
        start_utc = pd.Timestamp(ses["start_utc"]).tz_localize(None)
        if act.get("local_timestamp") and act.get("timestamp"):
            offset = act["local_timestamp"].replace(tzinfo=None) - act["timestamp"].replace(tzinfo=None)
        else:
            offset = pd.Timestamp(ses["start_utc"]).tz_convert(PROC["local_timezone"]).utcoffset()
        sess_id = f"{sid}_fit_{start_utc + offset:%Y%m%d%H%M%S}"           # local start time
        speed = ses.get("speed", np.nan)
        sessions.append({"study_id": sid, "session_id": sess_id, "source": SOURCE, "file_id": sess_id,
                         "activity": "running", "start": start_utc + offset, "start_utc": start_utc,
                         "distance_km": ses.get("distance_m", np.nan) / 1000, "duration_s": ses.get("duration_s"),
                         "pace_s_km": 1000 / speed if speed else np.nan, "hr_mean": ses.get("hr_mean"),
                         "hr_max": ses.get("hr_max"), "lat": ses.get("lat", np.nan) * DEG,
                         "lon": ses.get("lon", np.nan) * DEG, "temp_c": ses.get("temp_c"),
                         "indoor": is_indoor(ses.get("sub_sport"))})
        r = pd.DataFrame(recs, columns=list(RECORD.values()))
        r["t"] = pd.to_datetime(r.t).dt.tz_localize(None)
        streams.append(pd.DataFrame({"session_id": sess_id, "t": r.t + offset,
                                     "elapsed_s": (r.t - start_utc).dt.total_seconds(),
                                     "hr": r.hr, "speed_ms": r.speed_ms, "lat": r.lat * DEG,
                                     "lon": r.lon * DEG, "temp_c": r.temp_c}))
        info.append({"study_id": sid, "source": SOURCE, "file_id": sess_id, "date": start_utc,
                     "age": u79.get("age"), "gender": {1: "male", 0: "female"}.get(u79.get("gender"))})
        log.append({**entry, "file_id": sess_id, "n_rows": len(r), "status": "used"})
    return {"sessions": pd.DataFrame(sessions, columns=SESSION_COLS),
            "streams": pd.concat(streams) if streams else empty(STREAM_COLS),
            "info": pd.DataFrame(info, columns=INFO_COLS), "log": pd.DataFrame(log)}
