"""Cut the raw exports to the data window: raw -> interim.

Per device:

  Garmin Connect .csv   rows filtered on the date column
  Garmin / Coros .fit   GarminI: file copied when its date is in the window
  Coros                 the .fit messages of all files
  Apple                 export.xml
  Fitbit (Takeout)      files of 4 folders with their date in the window
  Polar (.json)         file copied when its date is in the window
  Strava                activities.csv filtered, activities stored as .gpx
"""
import csv
import gzip
import os
import re
import shutil
import struct
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .participants import RACES, find_folders, key
from ..settings import CFG, DATA_ROOT, RAW_ROOT

MONTHS = int(CFG.get("window", {}).get("months_before", 3))

# month abbreviations (Dutch, English, French) -> month number
MONTH_NR = {"jan": 1, "janv": 1, "feb": 2, "févr": 2, "fevr": 2, "mrt": 3, "mar": 3, "mars": 3,
            "apr": 4, "avr": 4, "mei": 5, "may": 5, "mai": 5, "jun": 6, "juin": 6,
            "jul": 7, "juil": 7, "aug": 8, "août": 8, "aout": 8, "sep": 9, "sept": 9,
            "okt": 10, "oct": 10, "nov": 11, "dec": 12, "déc": 12}


# ------------------------------------------------------------------ window per participant, files in a folder
def window(races, months=MONTHS):
    """(first day, last day): <months> months before the first race up to and including the last race."""
    days = [RACES[r] for r in races]
    return min(days) - pd.DateOffset(months=months), max(days)


def files(folder, *ext):
    """All files below folder ending with one of the extensions (case-insensitive)."""
    ext = tuple(e.lower() for e in ext)
    return sorted(Path(d) / f for d, _, fs in os.walk(folder) for f in fs if f.lower().endswith(ext))


# ------------------------------------------------------------------------------ Garmin Connect (.csv)
DATE_HEADER = re.compile(r"\b(datum|date)\b", re.IGNORECASE)        # 'Datum', 'Date', 'Datum van activiteit'
NUMBER = re.compile(r"-?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?")             # '41.26', '3,388', '-42'
DUTCH_DECIMAL = re.compile(r"-?\d+,\d{1,2}")                          # '21,26', '1,28'


def read_garmin_csv(path, sep):
    """Rows padded / cut to the header; rows quoted as one field ("Hardlopen,...,""42.91"",...") unwrapped.
    Returns the data and the header exactly as in the file."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=sep))
    header, data = rows[0], []
    for r in rows[1:]:
        if not r or all(v == "" for v in r):
            continue
        if len(r) == 1 and sep in r[0]:
            r = next(csv.reader([r[0]], delimiter=sep))
        data.append((r + [""] * len(header))[:len(header)])
    return pd.DataFrame(data, columns=range(len(header)), dtype=str), header


def date_column(header, filename):
    """First column with 'Datum' / 'Date' in its name, else column 2 for Activities, column 1 otherwise."""
    for i, col in enumerate(header):
        if DATE_HEADER.search(str(col)):
            return i
    return 1 if "activit" in filename.lower() else 0


def garmin_date(value, end):
    """'16 Mrt', 'Mrt 23', '16 mrt. 2026', '12 apr 2026, 08:43:18', '01-04-2026', '2026-03-29 10:15:00';
    a range ('2026-01-05 - 2026-01-11'): the first date. No year: the most recent date on or before end."""
    text = str(value).split(" - ")[0].strip()
    parts = text.lower().replace(".", "").replace(",", "").split()
    if len(parts) >= 2 and parts[0] in MONTH_NR:
        parts[0], parts[1] = parts[1], parts[0]
    if not (len(parts) >= 2 and parts[0].isdigit() and parts[1] in MONTH_NR):
        return pd.to_datetime(text, errors="coerce", dayfirst=not text[:4].isdigit())
    day, month = int(parts[0]), MONTH_NR[parts[1]]
    try:
        if len(parts) >= 3 and parts[2].isdigit():
            return pd.Timestamp(int(parts[2]), month, day)
        date = pd.Timestamp(end.year, month, day)
        return date if date <= end else pd.Timestamp(end.year - 1, month, day)
    except ValueError:
        return pd.NaT


def to_dutch_number(value, comma_is_thousands):
    """'41.26' -> '41,26', (',' separated files) '3,388' -> '3388'; times, paces, text and Dutch numbers kept."""
    if not isinstance(value, str) or not NUMBER.fullmatch(value.strip()):
        return value
    if "," in value and not comma_is_thousands:
        return value
    return value.replace(",", "").replace(".", ",")


def garmin_connect(folder, out, start, end):
    """Garmin Connect .csv: rows in the window, Dutch number notation, then the Excel fixes."""
    msgs = []
    for path in files(folder, ".csv"):
        try:
            with open(path, encoding="utf-8-sig") as f:
                first = f.readline()
            sep = ";" if first.count(";") > first.count(",") else ","
            df, header = read_garmin_csv(path, sep)
            col = date_column(header, path.name)
            dates = pd.to_datetime(df.iloc[:, col].apply(garmin_date, end=end)).dt.normalize()
            o = df[(dates >= start) & (dates <= end)]
            dutch = df.apply(lambda c: c.str.strip().str.fullmatch(DUTCH_DECIMAL)).any().any()
            if not dutch:
                o = o.apply(lambda c: c.map(lambda v: to_dutch_number(v, comma_is_thousands=sep == ",")))
            o.columns = header
            o.to_csv(out / path.name, index=False, sep=";", encoding="utf-8-sig", lineterminator="\r\n")
            msgs.append(f"{path.name}: {len(o)} of {len(df)} rows")
        except Exception as e:
            msgs.append(f"{path.name}: ERROR {e}")
    fixed = fix_excel(out)
    return f"{len(msgs)} .csv: " + "; ".join(msgs) + (f"; fixed: {fixed}" if fixed else "")


# Garmin Connect .csv saved in Excel (was help_functions/fix_csv_files.py), fixed in place in the interim folder
EXCEL_DATE = re.compile(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?: (\d{1,2}):(\d{2})(?::(\d{2}))?)?$")


def fix_excel_file(path):
    """Summary in one column -> renamed .notused; Activities '12-4-2026 10:10' -> '2026-04-12 10:10:00';
    weekly '17/12/2025' (day first) -> '12/17/2025' (as Garmin writes them). None = nothing to fix."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    sep = max(";,", key=text.split("\n", 1)[0].count)
    rows = list(csv.reader(text.splitlines(), delimiter=sep))
    if max(map(len, rows), default=0) <= 1:
        os.rename(path, str(path) + ".notused")
        return "not used (summary file)"
    col = 1 if len(rows[0]) >= 13 else 0
    dates = [(r, EXCEL_DATE.match(r[col].strip())) for r in rows[1:] if len(r) > col]
    dates = [(r, m) for r, m in dates if m]
    if not dates or (col == 0 and not any(int(m[1]) > 12 for _, m in dates)):
        return None
    for r, m in dates:
        d = datetime(int(m[3]), int(m[2]), int(m[1]), int(m[4] or 0), int(m[5] or 0), int(m[6] or 0))
        r[col] = d.strftime("%Y-%m-%d %H:%M:%S" if col == 1 else "%m/%d/%Y")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, delimiter=sep).writerows(rows)
    return "dates fixed"


def fix_excel(folder):
    """Fix every .csv in the folder (fix_excel_file); returns what was fixed."""
    return ", ".join(f"{p.name} ({m})" for p in files(folder, ".csv") if (m := fix_excel_file(p)))


# ------------------------------------------------------------------------------ Garmin internal (.fit)
DATE_IN_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
FIT_EPOCH = pd.Timestamp("1989-12-31")


def date_from_fit(path):
    """Date inside a .fit file (file_id time_created, else the first timestamp; UTC), NaT when unreadable.
    Reads the FIT records directly (no package)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return pd.NaT
    if len(data) < 12 or data[8:12] != b".FIT":
        return pd.NaT
    pos = data[0]
    end = min(pos + struct.unpack("<I", data[4:8])[0], len(data))
    defs = {}
    try:
        while pos < end:
            h = data[pos]
            pos += 1
            if h & 0x80:
                local, is_def, has_dev = (h >> 5) & 0x03, False, False
            else:
                local, is_def, has_dev = h & 0x0F, bool(h & 0x40), bool(h & 0x20)
            if is_def:
                endian = "<" if data[pos + 1] == 0 else ">"
                num = struct.unpack(endian + "H", data[pos + 2:pos + 4])[0]
                n, pos = data[pos + 4], pos + 5
                fields = []
                for _ in range(n):
                    fields.append((data[pos], data[pos + 1]))
                    pos += 3
                dev = 0
                if has_dev:
                    nd, pos = data[pos], pos + 1
                    for _ in range(nd):
                        dev += data[pos + 1]
                        pos += 3
                defs[local] = (num, endian, fields, dev)
                continue
            if local not in defs:
                return pd.NaT
            num, endian, fields, dev = defs[local]
            for field, size in fields:
                if size == 4:
                    v = struct.unpack(endian + "I", data[pos:pos + 4])[0]
                    if 0x10000000 <= v < 0xFFFFFFFF and ((num == 0 and field == 4) or field == 253):
                        return (FIT_EPOCH + pd.Timedelta(seconds=v)).normalize()
                pos += size
            pos += dev
    except (IndexError, struct.error):
        return pd.NaT
    return pd.NaT


def garmin_fit(folder, out, start, end):
    """Copy the .fit files dated in the window."""
    fits = files(folder, ".fit")
    kept, inside, none = 0, 0, []
    for path in fits:
        m = DATE_IN_NAME.search(path.name)
        date = pd.to_datetime(m.group(0), errors="coerce", format="%Y-%m-%d") if m else pd.NaT
        if pd.isna(date):
            date = date_from_fit(path)
            if pd.isna(date):
                none.append(path.name)
                continue
            inside += 1
        if start <= date <= end:
            shutil.copy2(path, out / path.name)
            kept += 1
    return f"{kept} of {len(fits)} .fit kept ({inside} dated from the file content, {len(none)} without date)"


# ------------------------------------------------------------------------------ Coros (.fit -> .csv per message)
def coros(folder, out, start, end):
    """All .fit messages in the window -> one .csv per message type."""
    import fitdecode
    start, end = start.date(), end.date()
    tables, used, errors = {}, 0, 0
    fits = files(folder, ".fit")
    for path in fits:
        with_time, without_time, checked = [], [], False
        try:
            with fitdecode.FitReader(path) as reader:
                for frame in reader:
                    if not isinstance(frame, fitdecode.FitDataMessage):
                        continue
                    rec = {f.name: f.value for f in frame.fields}
                    ts = rec.get("timestamp")
                    if not isinstance(ts, datetime):
                        without_time.append((frame.name, rec))
                        continue
                    if frame.name == "record" and not checked:     # file clearly outside: stop reading
                        checked = True
                        if ts.date() < start - timedelta(days=2) or ts.date() > end:
                            with_time = []
                            break
                    if start <= ts.date() <= end:
                        with_time.append((frame.name, rec))
        except Exception:
            errors += 1
            continue
        if with_time:
            used += 1
            for name, rec in without_time + with_time:
                tables.setdefault(name, []).append(rec)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False, sep=";", encoding="utf-8-sig")
    return f"{used} of {len(fits)} .fit with data in the window -> {len(tables)} .csv" + (f" ({errors} unreadable)" if errors else "")


# ------------------------------------------------------------------------------ Apple (export.xml -> .csv per type)
def short_type(t):
    """'HKQuantityTypeIdentifierHeartRate' -> 'HeartRate'."""
    return (t.replace("HKQuantityTypeIdentifier", "").replace("HKCategoryTypeIdentifier", "Cat_")
             .replace("HKDataTypeIdentifier", "").replace("HKWorkoutActivityType", ""))


def flatten(elem):
    """Attributes + MetadataEntry and WorkoutStatistics children as one row."""
    row = dict(elem.attrib)
    for child in elem:
        if child.tag == "MetadataEntry" and child.get("key"):
            row["meta:" + child.get("key")] = child.get("value", "")
        elif child.tag == "WorkoutStatistics":
            name = short_type(child.get("type", ""))
            for k in ("sum", "average", "minimum", "maximum", "unit"):
                if child.get(k) is not None:
                    row[f"{name}_{k}"] = child.get(k)
    return row


def apple(folder, out, start, end):
    """export.xml -> .csv per data type (creationDate in the window) + the routes (.gpx) in the window."""
    xml = [p for p in files(folder, ".xml") if p.name.lower() == "export.xml"]
    gpx = files(folder, ".gpx")
    msg = []
    if xml:
        s, e = f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"
        tables, depth = {}, 0
        for event, elem in ET.iterparse(xml[0], events=("start", "end")):
            if event == "start":
                depth += 1
                continue
            depth -= 1
            if depth == 1:
                name = short_type(elem.get("type")) if elem.get("type") else elem.tag
                tables.setdefault(name, []).append(flatten(elem))
                elem.clear()
        overview = []
        for name, rows in tables.items():
            df = pd.DataFrame(rows)
            total = len(df)
            if "creationDate" in df.columns:
                day = df["creationDate"].fillna("").str[:10]
                df, note = df[(day >= s) & (day <= e)], "filtered on creationDate"
            else:
                note = "no creationDate column, not filtered"
            if df.empty:
                continue
            df.to_csv(out / (re.sub(r'[\\/:*?"<>|]', "_", name) + ".csv"), index=False, sep=";", encoding="utf-8-sig")
            overview.append({"type": name, "rows": len(df), "rows_before": total, "note": note})
        if overview:
            pd.DataFrame(overview).sort_values("rows", ascending=False).to_csv(
                out / "_overview.csv", index=False, sep=";", encoding="utf-8-sig")
        msg.append(f"export.xml -> {len(overview)} .csv" + (f" (more than one export.xml, used {xml[0].name})" if len(xml) > 1 else ""))
    copied = 0
    for path in gpx:                                    # route_2026-03-03_6.28pm.gpx
        m = re.search(r"\d{4}-\d{2}-\d{2}", path.name)
        if m and start <= pd.Timestamp(m.group()) <= end:
            (out / "workout-routes").mkdir(exist_ok=True)
            shutil.copy2(path, out / "workout-routes")
            copied += 1
    msg.append(f"{copied} of {len(gpx)} .gpx")
    return "; ".join(msg)


# ------------------------------------------------------------------------------ Fitbit (Google Takeout)
FITBIT_FOLDERS = {"physical activity_googledata", "temperature", "oxygen saturation (spo2)", "heart rate variability"}
FITBIT_DATE = re.compile(r"(\d{4})-(\d{2})-\(?(\d{2})\)?")


def fitbit_period(name):
    """First and last day of the file (a date on the 1st = the whole month), None without a date."""
    dates = [pd.Timestamp(int(y), int(m), int(d)) for y, m, d in FITBIT_DATE.findall(name)]
    if not dates:
        return None
    a, b = min(dates), max(dates)
    return (a, a + pd.offsets.MonthEnd(0)) if len(dates) == 1 and a.day == 1 else (a, b)


def fitbit_csv(path, out_path, start, end):
    """Copy a .csv with only the rows in the window (the whole file without a date column)."""
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
    for col in df.columns:
        if re.search(r"time|date|day", col, re.IGNORECASE):
            dates = pd.to_datetime(df[col].str.extract(r"(\d{4}-\d{2}-\d{2})")[0], errors="coerce")
            if dates.notna().any():
                df[(dates >= start) & (dates <= end)].to_csv(out_path, index=False, sep=sep)
                return
    shutil.copy2(path, out_path)


def fitbit(folder, out, start, end):
    """Google Takeout: the files of FITBIT_FOLDERS dated in the window."""
    dirs = [Path(d) / s for d, subs, _ in os.walk(folder) for s in subs if s.lower() in FITBIT_FOLDERS]
    kept = 0
    for d in dirs:
        o = out / d.name
        o.mkdir(exist_ok=True)
        for path in sorted(d.iterdir()):
            period = fitbit_period(path.name)
            if period is None:
                if "readme" in path.name.lower():
                    shutil.copy2(path, o / path.name)
                continue
            if period[1] < start or period[0] > end:
                continue
            if path.name.lower().endswith(".csv"):
                fitbit_csv(path, o / path.name, start, end)
            else:
                shutil.copy2(path, o / path.name)
            kept += 1
    return f"{len(dirs)} folder(s), {kept} file(s) in the window"


# ------------------------------------------------------------------------------ Polar (.json)
POLAR_DAY = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")                  # activity-2019-06-06-<id>.json
POLAR_MONTH = re.compile(r"(20\d{2})_(\d{2})\b")                      # 247ohr_2018_01-<id>.json
CREATED = re.compile(r'"created"\s*:\s*"(\d{4}-\d{2}-\d{2})')
START_SECONDS = re.compile(r'"startTimeSeconds"\s*:\s*"?(\d+)')


def polar_period(path):
    """First and last day of a Polar file (name, monthly 24/7 HR, else the date inside); None without a date."""
    if m := POLAR_DAY.search(path.name):
        d = pd.Timestamp(*map(int, m.groups()))
        return d, d
    if m := POLAR_MONTH.search(path.name):
        a = pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
        return a, a + pd.offsets.MonthEnd(0)
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(5000)
    if m := CREATED.search(text):
        d = pd.Timestamp(m.group(1))
    elif m := START_SECONDS.search(text):
        d = pd.Timestamp(int(m.group(1)), unit="s").normalize()
    else:
        return None
    return d, d


def polar(folder, out, start, end):
    """Copy the .json files dated in the window."""
    js = files(folder, ".json")
    kept, none = 0, 0
    for path in js:
        period = polar_period(path)
        if period is None:
            none += 1
        elif period[0] <= end and period[1] >= start:
            shutil.copy2(path, out / path.name)
            kept += 1
    return f"{kept} of {len(js)} .json kept ({none} without date)"


# ------------------------------------------------------------------------------ Strava
def strava_date(value):
    """'11 apr 2026, 16:30:52', 'Apr 11, 2026, 4:30:52 PM', '2026-04-11 16:30:52' -> the day."""
    parts = str(value).lower().replace(",", " ").replace(".", " ").split()
    numbers = [x for x in parts if x.isdigit()]
    month = next((MONTH_NR[x] for x in parts if x in MONTH_NR), None)
    if month and len(numbers) >= 2:
        day, year = int(numbers[0]), int(numbers[1])
        if day > 31:
            day, year = year, day
        return pd.Timestamp(year, month, day)
    d = pd.to_datetime(str(value), errors="coerce", dayfirst=not str(value)[:4].isdigit())
    return pd.NaT if pd.isna(d) else d.normalize()


def fit_points(src):
    """(lat, lon, time) of the GPS points of a .fit or .fit.gz file."""
    import fitdecode
    opener = gzip.open if str(src).lower().endswith(".gz") else open
    points = []
    with opener(src, "rb") as f, fitdecode.FitReader(f) as reader:
        for frame in reader:
            if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record":
                v = {fl.name: fl.value for fl in frame.fields}
                if v.get("position_lat") is not None and v.get("position_long") is not None:
                    points.append((v["position_lat"] * 180 / 2 ** 31, v["position_long"] * 180 / 2 ** 31, v["timestamp"]))
    return points


def write_gpx(points, dst):
    """(lat, lon, time) points -> .gpx file."""
    rows = [f'   <trkpt lat="{lat:.7f}" lon="{lon:.7f}"><time>{t.strftime("%Y-%m-%dT%H:%M:%SZ")}</time></trkpt>'
            for lat, lon, t in points]
    with open(dst, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                '<gpx version="1.1" creator="fit_to_gpx" xmlns="http://www.topografix.com/GPX/1/1">\n'
                ' <trk>\n  <trkseg>\n' + "\n".join(rows) + '\n  </trkseg>\n </trk>\n</gpx>\n')


def strava(folder, out, start, end):
    """activities.csv in the window + its activities as .gpx; loose .gpx / .fit in the window too."""
    act = out / "activities"
    act.mkdir(exist_ok=True)
    stored, in_csv, msg = 0, set(), []
    csvs = [p for p in files(folder, ".csv") if p.name.lower() == "activities.csv"]
    if csvs:                                            # 1. activities.csv + the files it names
        root = csvs[0].parent
        raw = pd.read_csv(csvs[0], header=None, dtype=str, encoding="utf-8-sig", keep_default_na=False)
        header, df = list(raw.iloc[0]), raw.iloc[1:].reset_index(drop=True)
        dates = pd.to_datetime(df.iloc[:, 1].apply(strava_date))
        sel = df[(dates >= start) & (dates <= end)]
        sel.to_csv(out / "activities.csv", header=header, index=False, sep=";", encoding="utf-8-sig")
        msg.append(f"{len(sel)} of {len(df)} activities")
        col = header.index("Bestandsnaam") if "Bestandsnaam" in header else 12
        in_csv = {os.path.normcase(os.path.abspath(root / x.replace("/", os.sep))) for x in df.iloc[:, col] if x}
        for rel in sel.iloc[:, col]:
            if not rel:
                continue
            src = root / rel.replace("/", os.sep)
            if not src.exists():
                msg.append(f"file not found: {rel}")
                continue
            dst = act / (src.name.split(".")[0] + ".gpx")
            if src.name.lower().endswith(".gpx"):
                shutil.copy2(src, dst)
                stored += 1
            elif ".fit" in src.name.lower():
                write_gpx(fit_points(src), dst)
                stored += 1
    for src in files(folder, ".gpx") + files(folder, ".fit"):     # 2. loose files not in activities.csv
        if os.path.normcase(os.path.abspath(src)) in in_csv:
            continue
        dst = act / (src.stem + ".gpx")
        if src.name.lower().endswith(".gpx"):
            with open(src, encoding="utf-8", errors="ignore") as f:
                found = re.search(r"<time>(\d{4}-\d{2}-\d{2})", f.read())
            if found and start <= pd.Timestamp(found.group(1)) <= end:
                shutil.copy2(src, dst)
                stored += 1
        else:
            points = fit_points(src)
            if points and start <= pd.Timestamp(points[0][2].date()) <= end:
                write_gpx(points, dst)
                stored += 1
    return "; ".join(msg + [f"{stored} .gpx stored"])


# ------------------------------------------------------------------------------ per participant
# brand in the brand overview -> what is cut (as extract.READERS: 'Garmin (both)' = Connect .csv and .fit)
CUTTERS = {"GarminC": [garmin_connect], "GarminI": [garmin_fit], "Garmin (both)": [garmin_connect, garmin_fit],
           "Garmin": [garmin_connect, garmin_fit], "Coros": [coros], "Apple": [apple], "Fitbit": [fitbit],
           "Polar": [polar], "Strava": [strava]}


def cut_one(study_id, brand, races, raw_folders, months=MONTHS, interim=DATA_ROOT):
    """Cut one participant: every raw folder -> interim / group / study_id (a fresh folder). Returns log rows."""
    start, end = window(races, months)
    rows = []
    for group in dict.fromkeys(g for _, g in raw_folders):
        out = Path(interim) / group / study_id
        shutil.rmtree(out, ignore_errors=True)            # cut again: nothing of an earlier window left
        out.mkdir(parents=True, exist_ok=True)
        for folder, _ in [f for f in raw_folders if f[1] == group]:
            for cut in CUTTERS[brand]:
                try:
                    msg = cut(folder, out, start, end)
                except Exception as e:
                    msg = f"ERROR {type(e).__name__}: {e}"
                rows.append({"study_id": study_id, "brand": brand, "races": " & ".join(races),
                             "start": start.date(), "end": end.date(), "group": group, "step": cut.__name__,
                             "result": msg})
    return rows


def cut_participants(p, study_ids=None, brands=None, redo=False, months=MONTHS, interim=DATA_ROOT):
    """Cut the raw exports of the participants in p (participants.load()) to the window -> interim.
    study_ids: only these (always cut again) | brands: only these devices | redo: cut everyone again.
    Returns the log (1 row per participant, folder and device step)."""
    if RAW_ROOT is None:
        raise SystemExit("No raw_root in [paths]: put the folder of the raw exports in config/settings.local.toml")
    if brands:
        p = p[p.brand.map(key).isin(map(key, brands))]
    raw = find_folders(p.study_id, root=RAW_ROOT)
    todo = p[p.study_id.str.upper().isin(raw) & p.brand.isin(CUTTERS) & (p.races != "")]
    if study_ids:
        todo = todo[todo.study_id.str.upper().isin([x.upper() for x in study_ids])]
    elif not redo:
        done = set(find_folders(p.study_id, root=interim))
        todo = todo[~todo.study_id.str.upper().isin(done)]
    print(f"{len(p)} participants, raw folder found for {len(raw)}, to cut now ({months} months before the race): {len(todo)}")
    log = []
    try:
        for i, r in enumerate(todo.itertuples(), 1):
            rows = cut_one(r.study_id, r.brand, r.races.split(";"), raw[r.study_id.upper()], months, interim)
            log += rows
            errors = sum(x["result"].startswith("ERROR") for x in rows)
            print(f"[{i}/{len(todo)}] {r.study_id} ({r.brand}) {rows[0]['start']} - {rows[0]['end']}"
                  + (f"  {errors} ERROR(S)" if errors else ""))
    except KeyboardInterrupt:
        print("stopped - what is cut so far is kept")
    return pd.DataFrame(log)


# ------------------------------------------------------------------------------ helpers for single files
def unzip(folder):
    """Extract every .gz in folder (not subfolders): .tar.gz / .tgz into a subfolder, .gz next to it."""
    for gz in sorted(p for p in Path(folder).iterdir() if p.is_file() and p.name.lower().endswith(".gz")):
        try:
            if gz.name.lower().endswith((".tar.gz", ".tgz")):
                dest = gz.with_suffix("").with_suffix("")
                dest.mkdir(exist_ok=True)
                with tarfile.open(gz, "r:gz") as tf:
                    tf.extractall(dest)
            else:
                with gzip.open(gz, "rb") as src, open(gz.with_suffix(""), "wb") as dst:
                    shutil.copyfileobj(src, dst)
            print(f"{gz.name}: extracted")
        except (tarfile.TarError, gzip.BadGzipFile, OSError) as e:
            print(f"{gz.name}: failed ({e})")


def numbers_to_csv(numbers_file, csv_file):
    """First table of an Apple Numbers file -> ';' .csv (pip install numbers-parser)."""
    from numbers_parser import Document
    table = Document(numbers_file).sheets[0].tables[0]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        for row in table.rows():
            out = []
            for cell in row:
                m = re.search(r"value=([^,]+)", str(cell))
                out.append(m.group(1).strip("\"'") if m else "")
            w.writerow(out)
