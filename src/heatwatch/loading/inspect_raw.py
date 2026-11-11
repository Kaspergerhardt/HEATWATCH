"""Inspect the exports.

  .csv / .txt     separator, header, columns, number of rows
  .xlsx / .xls    per sheet: columns, number of rows
  .json           key paths ('exercises[].samples.samples[].type')
  .fit (.gz)      per message type (session, record, ...): fields, number of messages
  .gpx (.gz)      element and attribute paths
  .xml            per element type (Record: HeartRate, Workout, ...): attributes (Apple export.xml: ~2 min per GB)
  .zip / .gz      archive: number of files inside and their name patterns
"""
import csv
import gzip
import io
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .participants import find_folders, key
from ..settings import RAW_ROOT

N_EXAMPLES = 3
XML_MAX_ELEMENTS = None               # None = whole file (Apple export.xml: ~2 min per GB; its first part holds only
                                      # heart-rate records, the workouts come later); a number = only that many elements
JSON_MAX_MB = 50
NAME_DATE = re.compile(r"(20\d{2})[-_](\d{2})(?:[-_](\d{2}))?")


# ------------------------------------------------------------------ file names and example values
def pattern(name):
    """Numbers and ids ('5f3a9c2e') -> #: files of the same kind get the same pattern."""
    name = re.sub(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "#", name)   # uuid
    name = re.sub(r"(?<![A-Za-z0-9])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{6,}(?![A-Za-z0-9])", "#", name)   # hex ids
    return re.sub(r"\d+", "#", name)


def name_date(name):
    """Date in a file name (2026-04-12 or 2026_04) or NaT."""
    m = NAME_DATE.search(name)
    if not m:
        return pd.NaT
    try:
        return pd.Timestamp(int(m[1]), int(m[2]), int(m[3] or 1))
    except ValueError:
        return pd.NaT


class Fields:
    """Field name."""

    def __init__(self):
        self.ex, self.n = {}, {}

    def add(self, part, name, value):
        k = (part, name)
        self.n[k] = self.n.get(k, 0) + 1
        v = self.ex.setdefault(k, [])
        if value not in (None, "") and len(v) < N_EXAMPLES:
            s = str(value)[:60]
            if s not in v:
                v.append(s)

    def table(self):
        return pd.DataFrame([{"part": p, "column / field": f, "examples": " | ".join(self.ex[(p, f)]),
                              "n": self.n[(p, f)]} for p, f in self.ex])


# ------------------------------------------------------------------------------ per file type
def opener(path):
    """Open a file for reading, also when it is .gz."""
    return gzip.open(path, "rb") if path.name.lower().endswith(".gz") else open(path, "rb")


def describe_csv(path):
    """.csv: separator, rows, columns with examples."""
    with opener(path) as f:
        raw = f.read()
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.split("\n", 1)[0]
    sep = max([";", ",", "\t"], key=first.count)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=sep) if r]
    note = f"separator '{'tab' if sep == chr(9) else sep}', {len(rows) - 1} rows, {len(rows[0]) if rows else 0} columns" \
        + (", BOM" if raw[:3] == b"\xef\xbb\xbf" else "")
    fl = Fields()
    header = rows[0] if rows else []
    for r in rows[1:]:
        if len(r) == 1 and sep in r[0]:                       # whole row quoted as one field
            r = next(csv.reader([r[0]], delimiter=sep))
        for i, col in enumerate(header):
            fl.add("", f"{i}: {col}", r[i] if i < len(r) else None)
    return note, fl.table()


def describe_excel(path):
    """Excel: per sheet the columns with examples."""
    fl, notes = Fields(), []
    for sheet, df in pd.read_excel(path, sheet_name=None, dtype=str, nrows=200).items():
        notes.append(f"sheet '{sheet}': {df.shape[1]} columns")
        for i, col in enumerate(df.columns):
            for v in df[col].dropna():
                fl.add(sheet, f"{i}: {col}", v)
            if not df[col].notna().any():
                fl.add(sheet, f"{i}: {col}", None)
    return "; ".join(notes), fl.table()


def walk_json(fl, obj, path=""):
    """All key paths of a JSON object, with example values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(fl, v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for v in obj[:50]:                                    # the first 50 elements of every list
            walk_json(fl, v, path + "[]")
        if not obj:
            fl.add("", path + "[]", None)
    else:
        fl.add("", path, obj)


def describe_json(path):
    """JSON: key paths with examples (larger than JSON_MAX_MB: not opened)."""
    if path.stat().st_size > JSON_MAX_MB * 2 ** 20:
        return f"larger than {JSON_MAX_MB} MB, not opened", pd.DataFrame()
    with open(path, encoding="utf-8", errors="replace") as f:
        d = json.load(f)
    fl = Fields()
    walk_json(fl, d)
    return f"top level: {type(d).__name__}" + (f" with {len(d)} keys" if isinstance(d, dict) else f" of {len(d)}"), fl.table()


def describe_fit(path):
    """FIT: per message type the fields (with units) and examples."""
    import fitdecode
    fl, n = Fields(), {}
    with opener(path) as f, fitdecode.FitReader(f) as reader:
        for frame in reader:
            if isinstance(frame, fitdecode.FitDataMessage):
                n[frame.name] = n.get(frame.name, 0) + 1
                for fd in frame.fields:
                    fl.add(frame.name, fd.name + (f" ({fd.units})" if fd.units else ""), fd.value)
    t = fl.table()
    if len(t):
        t["n"] = t.part.map(n).where(t.part.notna(), t.n)        # number of messages of that type
    return ", ".join(f"{k} {v}" for k, v in n.items()), t


def local(tag):
    """XML tag without its namespace."""
    return tag.split("}")[-1]


def describe_xml(path, max_elements=XML_MAX_ELEMENTS):
    """Element types and attributes. Apple export.xml: per type (Record HeartRate, Workout, ...)."""
    fl, n, seen, stack = Fields(), {}, 0, []
    with opener(path) as f:
        for event, el in ET.iterparse(f, events=("start", "end")):
            if event == "start":
                stack.append(local(el.tag))
                part = local(el.tag) + (f" {el.get('type')}" if el.get("type") else "")
                if len(stack) <= 2:                          # the root and its direct children = the data types
                    n[part] = n.get(part, 0) + 1
                where = "/".join(stack[1:-1] + [part]) if len(stack) > 1 else part
                for a, v in el.attrib.items():
                    fl.add(where, "@" + a, v)
                seen += 1
            else:
                if el.text and el.text.strip() and not len(el):
                    fl.add("/".join(stack[1:]) or stack[0], "text", el.text.strip())
                stack.pop()
                el.clear()
            if max_elements and seen >= max_elements:
                break
    note = f"{'first ' + str(max_elements) + ' elements' if max_elements and seen >= max_elements else str(seen) + ' elements'}; " + \
        ", ".join(f"{k} {v}" for k, v in sorted(n.items(), key=lambda x: -x[1])[:30])
    return note, fl.table()


def describe_archive(path):
    """.zip: the name patterns of the files inside."""
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    else:
        return "gzip: one compressed file", pd.DataFrame()
    pats = pd.Series([pattern(Path(x).name) for x in names if not x.endswith("/")]).value_counts()
    return f"{len(names)} entries", pd.DataFrame({"part": "", "column / field": pats.index,
                                                  "examples": "", "n": pats.values})


def describe(path):
    """(note, table of columns / fields with examples) for one file."""
    name = path.name.lower()
    try:
        if name.endswith((".fit", ".fit.gz")):
            return describe_fit(path)
        if name.endswith((".gpx", ".gpx.gz", ".xml", ".tcx")):
            return describe_xml(path)
        if name.endswith((".csv", ".txt", ".csv.gz")):
            return describe_csv(path)
        if name.endswith((".xlsx", ".xls")):
            return describe_excel(path)
        if name.endswith(".json"):
            return describe_json(path)
        if name.endswith((".zip", ".gz")):
            return describe_archive(path)
    except Exception as e:
        return f"could not be read: {type(e).__name__}: {str(e)[:100]}", pd.DataFrame()
    return "not opened (file type not inspected)", pd.DataFrame()


# ------------------------------------------------------------------------------ per participant
def file_groups(folder):
    """1 row per subfolder + name pattern: number of files, MB, dates in the names, the sample file."""
    rows, todo = [], [Path(folder)]
    while todo:                                   # os.scandir: the size comes with the listing (fast on a network drive)
        d = todo.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir():
                    todo.append(Path(e.path))
                else:
                    rel = d.relative_to(folder)
                    rows.append({"subfolder": "" if rel == Path(".") else pattern(str(rel)), "pattern": pattern(e.name),
                                 "path": Path(e.path), "mb": e.stat().st_size / 2 ** 20, "date": name_date(e.name)})
    rows.sort(key=lambda r: (r["subfolder"], str(r["path"])))
    if not rows:
        return pd.DataFrame(columns=["subfolder", "pattern", "files", "MB", "first date", "last date", "sample"])
    f = pd.DataFrame(rows)
    out = []
    for (sub, pat), g in f.groupby(["subfolder", "pattern"], sort=False):
        sample = g.sort_values("date", na_position="first").path.iloc[-1]   # most recent (by name), else the last
        out.append({"subfolder": sub, "pattern": pat, "files": len(g), "MB": round(g.mb.sum(), 1),
                    "first date": g.date.min().date() if g.date.notna().any() else None,
                    "last date": g.date.max().date() if g.date.notna().any() else None, "sample": sample})
    return pd.DataFrame(out)


def inspect(study_id, folders):
    """One participant: [(raw folder, file groups, {group number: (note, table)})]."""
    out = []
    for folder, _ in folders:
        g = file_groups(folder)
        details = {i: describe(path) for i, path in g["sample"].items()}
        g["sample"] = g["sample"].map(lambda p: p.name)
        g.insert(2, "content", [details[i][0] for i in g.index])
        out.append((folder, g, details))
    return out


def run(p, study_ids=None, brands=None):
    """Raw folders of the participants in p (participants.load()), only study_ids / brands when given.
    Returns {study_id: inspect()}; show() displays them."""
    if RAW_ROOT is None:
        raise SystemExit("No raw_root in [paths]: put the folder of the raw exports in config/settings.local.toml")
    if brands:
        p = p[p.brand.map(key).isin(map(key, brands))]
    if study_ids:
        p = p[p.study_id.str.upper().isin([x.upper() for x in study_ids])]
    folders = find_folders(p.study_id, root=RAW_ROOT)
    print(f"{len(p)} participants, raw folder found for {sum(s.upper() in folders for s in p.study_id)}")
    out = {}
    try:
        for i, r in enumerate(p.itertuples(), 1):
            if r.study_id.upper() in folders:
                print(f"\r[{i}/{len(p)}] {r.study_id}" + " " * 20, end="")
                out[r.study_id] = (r, inspect(r.study_id, folders[r.study_id.upper()]))
    except KeyboardInterrupt:
        print("\nstopped - what is inspected so far is kept")
    print()
    return out


def show(result, details=True):
    """Per participant: brand, race(s), the raw folder(s), the file groups and (details) per group the columns /
    fields of the sample file with 3 examples."""
    from IPython.display import display, Markdown
    for sid, (r, parts) in result.items():
        display(Markdown(f"## {sid} &nbsp; {r.brand} &nbsp; race(s): {r.races.replace(';', ' & ') or '-'}"))
        for folder, groups, desc in parts:
            display(Markdown(f"`{Path(folder).relative_to(RAW_ROOT)}`"))
            display(groups.drop(columns="sample") if not details else groups)
            if details:
                for i, g in groups.iterrows():
                    note, table = desc[i]
                    if len(table):
                        display(Markdown(f"**{g.subfolder + '/' if g.subfolder else ''}{g.pattern}** "
                                         f"({g.files} file(s); sample {g['sample']}): {note}"))
                        display(table)
