"""Garmin Connect export (.csv).

  Activities      col 1 type (running only), 2 date, 4 title -> location, 5 distance (km),
                  7 time, 8 mean HR, 9 max HR, mean tempo (found by heading; col 13 or 12)
  Resting HR      col 1 week (MM/DD/YYYY = week start), col 2 resting HR    
  Steps           col 1 week, col 2 steps in that week                           
  HRV status      col 1 date, col 2 overnight HRV (RMSSD)  
"""
import re

import pandas as pd

from .common import (DAILY_COLS, progress, SESSION_COLS, empty, file_hash, num, parse_dates, read_table,
                      seconds)

SOURCE = "garmin_connect"
RUN = ("hardlo", "running", "run", "trail", "loopband", "course", "piste", "tapis")
NOT_RUN = ("cycl", "fiets", "swim", "zwem", "natation", "wandel", "walk", "marche", "velo", "vélo")
PACE_HEADINGS = ("gemiddeld tempo", "gem. tempo", "avg pace", "average pace", "allure moy")
TWO_COL = {  # file kind: (words in file name / heading, metric, unit, days covered, thousands sep.)
    "hrv": (("hrv",), "hrv_rmssd", "ms", 1, False),
    "resting_hr": (("rust", "resting", "repos"), "resting_hr", "bpm", 7, False),
    "steps": (("stap", "step", "pas"), "steps", "steps per week", 7, True),
}


# ------------------------------------------------------------------ helpers: activity type, place, file kind
def is_run(t):
    """Activity type is running (nl / en / fr)"""
    t = str(t).lower()
    return any(w in t for w in RUN) and not any(w in t for w in NOT_RUN)


def place(title):
    """'Utrecht - W 16 Intervals' -> 'Utrecht'; 'Rotterdam Hardlopen' -> 'Rotterdam'."""
    t = re.sub(r"\s*\b(hardlopen.*|running|trail.*|run|loopband|course.*|treadmill.*)$", "",
               str(title).split(" - ")[0].strip(), flags=re.I)
    return t or None


def kind_of(path, df):
    """File -> 'activities', 'hrv', 'resting_hr', 'steps' or None."""
    if df.shape[1] >= 13:
        return "activities"
    text = (path.stem + " " + " ".join(map(str, df.columns))).lower()
    return next((k for k, (words, *_) in TWO_COL.items() if any(w in text for w in words)), None)


# ------------------------------------------------------------------ reading: activities, the reader
def activities(sid, fid, df):
    """Activities.csv -> the running sessions"""
    col = lambda i: df.iloc[:, i - 1]
    pace = next((i for i, h in enumerate(df.columns, 1) if str(h).strip().lower().startswith(PACE_HEADINGS)), 13)
    start, _ = parse_dates(col(2))
    s = pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": fid, "source_row": df.index + 2,
                      "activity": col(1), "start": start, "location": col(4).map(place),
                      "distance_km": col(5).map(num), "duration_s": col(7).map(seconds),
                      "hr_mean": col(8).map(num), "hr_max": col(9).map(num), "pace_s_km": col(pace).map(seconds)})
    s = s.loc[s.activity.map(is_run).astype(bool).values]       # .loc: safe for files without rows
    s["session_id"] = sid + "_gc_" + s.start.dt.strftime("%Y%m%d%H%M%S")
    return s


def parse(sid, folder):
    """One participant's Garmin Connect .csv files"""
    files = sorted(folder.rglob("*.csv"))
    log, sessions, daily = [], [], []
    for i, path in enumerate(files, 1):
        progress(sid, i, len(files), ".csv file")
        df = read_table(path)
        kind = kind_of(path, df)
        fid = f"{sid}_gc{i}_{kind or 'other'}"
        log.append({"study_id": sid, "source": SOURCE, "file_id": fid, "file_hash": file_hash(path),
                    "file_type": kind, "n_rows": len(df), "status": "used" if kind else "not used"})
        if kind == "activities":
            sessions.append(activities(sid, fid, df))
        elif kind:
            daily.append((kind, fid, df))

    s = pd.concat(sessions).drop_duplicates("session_id") if sessions else empty(SESSION_COLS)
    # year-less HRV dates: latest year that puts them before the last activity (+60 days), else today
    anchor = (s.start.max() + pd.Timedelta(days=60)) if len(s) else pd.Timestamp.today()
    rows = []
    for kind, fid, df in daily:
        _, metric, unit, days, thousands = TWO_COL[kind]
        start, note = parse_dates(df.iloc[:, 0], anchor)
        rows.append(pd.DataFrame({"study_id": sid, "source": SOURCE, "file_id": fid, "source_row": df.index + 2,
                                  "metric": metric, "start": start, "end": start + pd.Timedelta(days=days - 1),
                                  "value": df.iloc[:, 1].map(lambda v: num(v, thousands)), "unit": unit,
                                  "note": note}))
    d = pd.concat(rows).dropna(subset=["value"]) if rows else empty(DAILY_COLS)
    return {"sessions": s, "daily": d, "log": pd.DataFrame(log)}
