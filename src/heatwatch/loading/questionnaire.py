"""Questionnaires (Castor export).

  - Castor missing codes (-95 ... -99) means empty
  - race = from What_Race (CPC loop -> CPC, Rotterdam Marathon -> MR, others as written)
gender for the dataset: Gender from the questionnaire (Male / Female / Other)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..settings import CFG

SKIP = {"Participant Status", "Site Abbreviation", "Participant Creation Date"}
MISSING = {"-95", "-96", "-97", "-98", "-99"}
RACE = {"CPC loop": "CPC", "Rotterdam Marathon": "MR"}
GENDER = {"Male": "male", "Female": "female", "Other": "other"}


# ------------------------------------------------------------------ helpers: study number, file
def key(study_id):
    """'A02' / 'A2' -> 'A2'; 'HR104.1' -> ('HR104', 1)."""
    m = re.fullmatch(r"\s*([A-Za-z]+)0*(\d+)(?:\.(\d+))?\s*", str(study_id))
    return (m[1].upper() + m[2], int(m[3] or 0)) if m else (None, None)


def path():
    """paths.questionnaire_file from the settings (None if not set)."""
    p = CFG["paths"].get("questionnaire_file")
    return Path(p) if p else None


# ------------------------------------------------------------------ read the questionnaires, gender for the dataset
def read(file=None, study_ids=None):
    """Questionnaire table. study_ids (e.g. the brand overview) gives the spelling to use for the study number."""
    file = Path(file) if file else path()
    if file is None or not file.exists():
        print(f"no questionnaire file ({file or 'paths.questionnaire_file not set'}): questionnaires skipped")
        return pd.DataFrame(columns=["study_id", "race"])
    res = pd.read_excel(file, "Study results", dtype=str)
    var = pd.read_excel(file, "Study variable list", dtype=str)
    opt = pd.read_excel(file, "Field options", dtype=str)
    labels = {g: dict(zip(o["Option value"], o["Option name"])) for g, o in opt.groupby("Option group name")}
    types = dict(zip(var["Variable name"], var["Exported field type"]))
    groups = dict(zip(var["Variable name"], var["Optiongroup name"]))

    res = res[[c for c in res.columns if c not in SKIP]].apply(lambda c: c.str.strip())
    res = res.mask(res.isin(MISSING))
    k = res.pop("Participant Id").map(key)
    res.insert(0, "questionnaire_no", (k.str[1] + 1).astype("Int64"))
    spelling = {key(s)[0]: s for s in (study_ids if study_ids is not None else [])}
    res.insert(0, "study_id", k.str[0].map(lambda x: spelling.get(x, x)))
    res = res[res.study_id.notna() & (res.study_id != "9999")]
    if study_ids is not None:
        extra = sorted(set(res.study_id) - set(study_ids))
        if extra:
            print(f"questionnaire: {len(extra)} study numbers not in the brand overview (kept, check spelling): "
                  + ", ".join(extra))

    for c in res.columns[2:]:
        kind = types.get(c)
        if kind == "radio" and groups.get(c) in labels:
            res[c] = res[c].map(labels[groups[c]]).where(res[c].notna())
        elif kind == "checkbox":
            res[c] = res[c].map({"1": True, "0": False})
        elif kind in ("numeric", "slider"):
            res[c] = pd.to_numeric(res[c], errors="coerce")
    res.insert(2, "race", res.What_Race.map(lambda r: RACE.get(r, r) if isinstance(r, str) else None))
    return res.sort_values(["study_id", "questionnaire_no"]).reset_index(drop=True)


def add_gender(p, q):
    """gender from the questionnaire where answered, otherwise from the device; gender_source says which."""
    p = p.copy()
    if len(q) and "Gender" in q:
        qs = q.dropna(subset=["Gender"]).groupby("study_id").Gender.first().map(GENDER)
        from_q = p.study_id.map(qs)
        p["gender_source"] = np.where(from_q.notna(), "questionnaire", np.where(p.gender.notna(), "device", None))
        p["gender"] = from_q.fillna(p.gender)
    else:
        p["gender_source"] = np.where(p.gender.notna(), "device", None)
    return p
