"""Helpers shared by the readers"""
import hashlib
import io
import re

import numpy as np
import pandas as pd

from ...settings import PROC

TZ = PROC["local_timezone"]

MISSING = {"", "--", "-", "nan", "none", "n/a"}
DEG = 180 / 2 ** 31                     # .fit semicircles -> degrees
MONTHS = {m: i for i, names in enumerate([
    "jan januari january janv janvier", "feb febr februari february fevr févr fevrier février",
    "mrt maart mar march mars", "apr april avr avril", "mei may mai", "jun juni june juin",
    "jul juli july juil juillet", "aug augustus august aout août", "sep sept september septembre",
    "okt oktober oct october octobre", "nov november novembre", "dec december déc decembre décembre"], 1)
    for m in names.split()}


# ------------------------------------------------------------------ files and values: hash, table, numbers, durations, dates
def file_hash(path):
    """Short fingerprint of the file content (lets you trace a file without storing its name)."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def read_table(path):
    """csv with ; , or tab, any common encoding; all cells as text."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sep = max(";,\t", key=text.split("\n", 1)[0].count)
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def num(v, thousands=False):
    """'10,02' / '10.02' / '1.234,5' / '48ms' -> float. thousands=True: '62.345' -> 62345."""
    if str(v).strip().lower() in MISSING:
        return np.nan
    t = re.sub(r"[^\d,.\-]", "", str(v))
    if not re.search(r"\d", t):
        return np.nan
    if "," in t and "." in t:                       # the last separator is the decimal one
        t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
    elif thousands:
        t = re.sub(r"[.,]", "", t)
    else:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return np.nan


def seconds(v):
    """'01:02:03' / '45:12' / '5:51' (pace) -> seconds."""
    try:
        parts = [float(x) for x in str(v).replace(",", ".").split(":")]
    except ValueError:
        return np.nan
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts))) if 2 <= len(parts) <= 3 else np.nan


def parse_dates(col, anchor=None):
    """Parse a whole date column. Returns (dates, note).
    - 'a/b/yyyy': month-first unless a value proves day-first (Garmin exports are MM/DD/YYYY)
    - ISO 'yyyy-mm-dd [hh:mm:ss]'
    - text months in nl/en/fr ('10 mrt. 2025', 'Mar 10, 2025'); without year ('Mrt 19')
      the year is the latest one that puts the date on or before `anchor`."""
    s = col.astype(str).str.strip()
    m = s.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})").astype(float)
    if m[0].notna().any():
        dayfirst = m[0].max() > 12 >= m[1].max()
        d = pd.to_datetime(pd.DataFrame({"year": m[2], "month": m[0 if not dayfirst else 1],
                                         "day": m[1 if not dayfirst else 0]}), errors="coerce")
        return d, "DD/MM/YYYY" if dayfirst else "MM/DD/YYYY"
    iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if iso.notna().any():
        return iso, None
    out, inferred = [], False
    for v in s:
        low = v.lower().replace(".", " ").replace(",", " ")
        mo = next((MONTHS[w] for w in re.findall(r"[a-zéû]+", low) if w in MONTHS), None)
        nums = [int(n) for n in re.findall(r"\d+", low)]
        yr = next((n for n in nums if n > 31), None)
        dy = next((n for n in nums if n <= 31), None)
        if mo and dy and yr is None and anchor is not None:
            yr, inferred = anchor.year - ((mo, dy) > (anchor.month, anchor.day)), True
        try:
            out.append(pd.Timestamp(yr, mo, dy))
        except (TypeError, ValueError):
            out.append(pd.NaT)
    return pd.Series(out, index=col.index, dtype="datetime64[ns]"), "year inferred" if inferred else None


INDOOR_WORDS = ("treadmill", "loopband", "indoor", "binnen", "tapis", "virtual")


# ------------------------------------------------------------------ runs: indoor, age, distance, time zone
def is_indoor(v):
    """'treadmill' / 'indoor_running' / 'Loopband' / 'Virtual Run' / Apple HKIndoorWorkout '1' -> True."""
    v = str(v).strip().lower()
    return v == "1" or any(w in v for w in INDOOR_WORDS)


def age_on(dob, day):
    """Age in whole years on `day`; None if one of them is missing."""
    if pd.isna(dob) or pd.isna(day):
        return None
    return day.year - dob.year - ((day.month, day.day) < (dob.month, dob.day))


def haversine(lat, lon):
    """Distance (m) between consecutive positions (degrees)"""
    la, lo = np.radians(lat), np.radians(lon)
    a = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def utc_to_local(t):
    """Column of UTC times without a zone -> local time without a zone."""
    return t.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)


# ------------------------------------------------------------------ output: progress line, empty tables, the columns every reader returns
def progress(sid, i, n, what="file"):
    """Live counter on one line, e.g. '    A07: file 12/75' (overwritten each time)."""
    print(f"\r    {sid}: {what} {i}/{n}   ", end="", flush=True)


def empty(cols):
    """Empty table with these columns."""
    return pd.DataFrame(columns=cols)


# Column layout every parser returns
SESSION_COLS = ["study_id", "session_id", "source", "file_id", "source_row", "activity", "start", "start_utc",
                "distance_km", "duration_s", "pace_s_km", "hr_mean", "hr_max", "location", "lat", "lon",
                "temp_c", "weather_temp_c", "weather_humidity_pct", "indoor"]
DAILY_COLS = ["study_id", "source", "file_id", "source_row", "metric", "start", "end", "value", "unit", "note"]
STREAM_COLS = ["session_id", "t", "elapsed_s", "hr", "speed_ms", "lat", "lon", "temp_c"]
INFO_COLS = ["study_id", "source", "file_id", "date", "age", "gender"]
LOG_COLS = ["study_id", "source", "file_id", "file_hash", "file_type", "n_rows", "status"]
