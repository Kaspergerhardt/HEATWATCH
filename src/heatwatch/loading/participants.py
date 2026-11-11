"""Participants: brand overview -> group, race(s), device; and where their data folder is."""
import os
import re
from pathlib import Path

import pandas as pd

from ..settings import CFG, DATA_ROOT

RACES = {k: pd.Timestamp(v) for k, v in CFG["races"].items()}
GROUPS = sorted(CFG["groups"].items(), key=lambda kv: -len(kv[0]))     # longest prefix first
POP = {"patient": "patients", "pacer": "pacers", "healthy": "healthy runners"}


# ------------------------------------------------------------------ participants from the brand overview
def load():
    """Brand overview -> 1 row per participant: group, races, brand."""
    df = pd.read_excel(CFG["paths"]["brand_file"], sheet_name=CFG["brand_file"]["sheet"], dtype=str,
                       usecols="A:E").fillna("")
    df.columns = ["study_id", "brand", "both", "run", "comment"]
    df = df.apply(lambda c: c.str.strip())
    df = df[df.study_id != ""]
    rows = []
    for r in df.itertuples():
        prefix, g = next((p, g) for p, g in GROUPS if re.fullmatch(p + r"\d+", r.study_id))
        races = [x.strip() for x in r.run.split("&")] if g["group"] == "HR" else [g["race"]]
        races = [x for x in races if x in RACES]                            # VS etc. skipped for now
        if g["group"] == "HR" and r.run and not races:
            continue
        brand = r.brand.split("&")[0].strip()
        brand = "Garmin (both)" if brand == "Garmin" and r.both.lower() == "yes" else brand
        rows.append({"study_id": r.study_id, "group": g["group"], "population": g["population"],
                     "races": ";".join(races), "brand": brand})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ data folders and the brand overview as it is
def find_folders(study_ids, root=None):
    """study_id -> [(folder, output group)], searching max_depth levels below data_root (or root, e.g. raw_root)."""
    root, depth = Path(root or DATA_ROOT), CFG["search"]["max_depth"]
    pattern = re.compile(r"^(" + "|".join(sorted(study_ids, key=len, reverse=True)) + r")\b", re.I)
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        for d in list(dirnames):
            m = pattern.match(d)
            if m:
                top = (rel.parts or (d,))[0]
                group = CFG["output_groups"].get(top, top)
                found.setdefault(m[1].upper(), []).append((Path(dirpath) / d, group))
                dirnames.remove(d)                                         # don't search inside it
        if len(rel.parts) + 1 >= depth:
            dirnames.clear()
    return found


def key(brand):
    """'Garmin (both)' / 'GarminBoth' / 'garmin both' -> 'garminboth' (easy to type)."""
    return "".join(ch for ch in str(brand).lower() if ch.isalnum())


def overview(p, folders):
    """Per participant: data folder found? device readable?."""
    from .readers import READERS
    return p.assign(data_folder=p.study_id.str.upper().isin(folders), device_readable=p.brand.isin(READERS))


def brand_overview(path=None):
    """Brand overview as it is, by position: A study number, B brand, C both, D run, E comments."""
    raw = pd.read_excel(path or CFG["paths"]["brand_file"], sheet_name=CFG["brand_file"]["sheet"], dtype=str).fillna("")
    raw = raw.apply(lambda c: c.str.strip())
    b = raw.iloc[:, :4].copy()
    b.columns = ["study_id", "brand", "both", "run"]
    notes = raw.iloc[:, 4:].apply(lambda r: "; ".join(x for x in r if x), axis=1) if raw.shape[1] > 4 else ""
    b["comment"] = notes
    return b[b.study_id != ""]
