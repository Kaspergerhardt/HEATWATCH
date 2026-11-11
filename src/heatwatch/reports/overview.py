"""Overview: what is in the dataset, what is missing and why.

  overview_tables    per population: participants with each parameter, smartwatch, trainings in the
                     14 days before the race
  data_check         per participant: read correctly or why not
  apply_exclusions   outcome per participant; excluded participants removed from the dataset (step 04)
  inclusion          per participant: included or excluded (+ reason), flow per population and group,
                     trainings without location, trainings with an impossible date
Outcome per participant (inclusion), first rule that applies:
  1. listed under [exclusion] in settings.toml (reason only you know, e.g. export could not be opened)
  2. data check not OK: no data folder / empty folder -> export could not be opened,
                         unsupported device -> no relevant data in export,
                         no trainings -> no running activity in study period
  3. trainings, but no run on the race day -> no data of the race
  4. otherwise included
"""
import numpy as np
import pandas as pd

from ..dataset import PARAMETERS, has
from ..loading.participants import POP, brand_overview
from ..loading.readers import READERS
from ..settings import CACHE, CFG


# ------------------------------------------------------------------------------ what is in the dataset
PARAMETER_NAMES = {"training": "Trainings", "location": "Location", "hr": "Heart rate (mean / max)",
                   "hr_5s": "Heart rate every 5 s", "gps_5s": "GPS every 5 s", "resting_hr": "Resting heart rate",
                   "steps": "Steps", "hrv": "HRV", "wbgt": "WBGT", "gender": "Gender", "age": "Age",
                   "questionnaire": "Questionnaire"}
DEVICES = {"GarminC": "Garmin", "GarminI": "Garmin", "Garmin (both)": "Garmin", "Garmin": "Garmin",
           "Apple": "Apple Watch", "Coros": "Coros", "Polar": "Polar", "Fitbit": "Fitbit",
           "Strava": "Strava (watch unknown)"}


def overview_tables(pt):
    """Per population, number of participants: participants with each parameter, smartwatch,
    trainings in the 14 days before the race (+ median (IQR))."""
    groups = list(POP.values())
    size = {**{g: int((pt.population == g).sum()) for g in groups}, "total": len(pt)}

    def count(mask):
        c = pt[mask].population.value_counts()
        n = {**{g: int(c.get(g, 0)) for g in groups}, "total": int(mask.sum())}
        return {g: str(v) for g, v in n.items()}

    head = {"": "participants", **{g: str(v) for g, v in size.items()}}
    par = pd.DataFrame([head] + [{"": PARAMETER_NAMES.get(k, k), **count(has(pt, k))} for k in PARAMETERS])
    device = pt.brand.map(DEVICES).fillna(pt.brand)
    order = [d for d in dict.fromkeys(DEVICES.values()) if (device == d).any()] + \
        sorted(set(device.dropna()) - set(DEVICES.values()))
    devices = pd.DataFrame([head] + [{"": d, **count(device == d)} for d in order])
    bins = [("0", pt.trainings_14d == 0), ("1-3", pt.trainings_14d.between(1, 3)),
            ("4-7", pt.trainings_14d.between(4, 7)), ("8 or more", pt.trainings_14d >= 8)]
    last = pd.DataFrame([head] + [{"": b, **count(m)} for b, m in bins] +
                        [{"": "median (IQR)", **{g: med_iqr(pt.trainings_14d[pt.population == g]) for g in groups},
                          "total": med_iqr(pt.trainings_14d)}])
    return par, devices, last


def med_iqr(x):
    """'median (Q1-Q3)' with one decimal, '-' when empty."""
    x = pd.Series(x).dropna()
    return f"{x.median():.1f} ({x.quantile(.25):.1f}-{x.quantile(.75):.1f})" if len(x) else "-"



# ------------------------------------------------------------------------------ data check
def data_check(info, folders, pt):
    """1 row per participant of the brand overview: read correctly (OK) or why not.
    info: participants.load(); folders: participants.find_folders; pt: dataset participants."""
    found = pt.set_index("study_id")
    rows = []
    for r in info.itertuples():
        pkl, err = CACHE / f"{r.study_id}.pkl", CACHE / f"{r.study_id}.error.txt"
        t = pd.read_pickle(pkl) if pkl.exists() else None
        log = t["log"] if t is not None else pd.DataFrame(columns=["status"])
        used = int(log.get("status", pd.Series(dtype=str)).astype(str).str.startswith("used").sum())
        if r.brand not in READERS:
            status = f"NO PARSER: device '{r.brand}' not supported yet"
        elif r.study_id.upper() not in folders:
            status = "NO DATA FOLDER found under data_root"
        elif err.exists():
            status = "ERROR: " + err.read_text()
        elif t is None:
            status = "NOT PROCESSED YET"
        elif not len(log):
            status = "EMPTY FOLDER: no files the parser reads"
        elif used == 0:
            status = "FILES NOT RECOGNISED: none of the files could be used"
        elif not len(t["sessions"]):
            status = "NO TRAININGS in the files"
        else:
            status = "OK"
        params = [k for k in PARAMETERS if r.study_id in found.index and
                  has(found.loc[[r.study_id]], k).iloc[0]]
        rows.append({"study_id": r.study_id, "population": POP.get(r.population), "brand": r.brand,
                     "status": status, "files": len(log), "files_used": used,
                     "parameters found": ", ".join(params) or None})
    df = pd.DataFrame(rows)
    return pd.concat([df[df.status != "OK"], df[df.status == "OK"]])      # problems first



# ------------------------------------------------------------------------------ inclusion / exclusion
ORDER = ["Race not part of the study", "Export could not be opened", "No relevant data in export",
         "No running activity in study period", "No data of the race"]
GROUPS = ["MRHR", "RMP", "RM", "CPCHR", "CPCP", "CPC", "HR, no study race"]
POP_ID = {"patients": r"^((A|C|K|S)\d|CPC\d)", "pacers": r"^(RMP|CPCP)\d", "healthy runners": r"^HR\d"}
NO_LOCATION = {"apple": "no route file (.gpx) for this workout", "fitbit": "no GPS (run without phone)",
               "strava": "no route file (manual entry or missing .gpx)", "gc": "no place in the Garmin Connect title",
               "fit": "no GPS in the .fit file", "coros": "no GPS in the file", "polar": "no route in the file"}


def yes(v):
    """'yes' / True / '1' -> True."""
    return v is True or str(v).strip().lower() in ("yes", "true", "1")


def study_group(study_id, run):
    """Study number (+ Run column for HR) -> group: RM, CPC, RMP, CPCP, MRHR, CPCHR."""
    s = study_id
    if s.startswith("CPCP"):
        return "CPCP"
    if s.startswith("CPC"):
        return "CPC"
    if s.startswith("RMP"):
        return "RMP"
    if s.startswith("HR"):
        g = [{"MR": "MRHR", "CPC": "CPCHR"}[x.strip()] for x in str(run).split("&") if x.strip() in ("MR", "CPC")]
        return " & ".join(g) or "HR, no study race"
    return "RM"


def outcomes(brand, check, participants):
    """1 row per participant in the brand overview: outcome + reason."""
    manual = CFG.get("exclusion", {})
    st = check.set_index("study_id").status
    pt = participants.set_index("study_id")
    rows = []
    for r in brand.itertuples():
        s, status = r.study_id, st.get(r.study_id, "not in data check")
        n_train = int(pd.to_numeric(pt.training.get(s, 0), errors="coerce") or 0) if s in pt.index else 0
        race = s in pt.index and yes(pt.race_present.get(s))
        if s in manual:
            out, why = manual[s]
        elif status.startswith(("NO DATA FOLDER", "EMPTY FOLDER", "ERROR", "FILES NOT RECOGNISED")):
            out, why = "Export could not be opened", status.split(":")[0].lower()
        elif status.startswith("NO PARSER"):
            out, why = "No relevant data in export", status
        elif status.startswith("NO TRAININGS") or n_train == 0:
            out, why = "No running activity in study period", r.comment or "no running activity found in the files"
        elif not race:
            out, why = "No data of the race", "trainings present, but no run on the race day"
        else:
            out, why = "Included", ""
        rows.append({"study_id": s, "group": study_group(s, r.run), "brand": r.brand, "run (overview)": r.run,
                     "outcome": out, "reason": why, "data check": status, "overview comment": r.comment,
                     "trainings": n_train, "race present": "yes" if race else ""})
    return pd.DataFrame(rows)


def apply_exclusions(ds, check):
    """Outcome per participant (outcomes) -> ds.outcomes; only included participants stay in the dataset.
    The data check gets the outcome as a column. Returns (ds, check)."""
    o = outcomes(brand_overview(), check, ds.participants)
    ds.outcomes = o
    ds.exclude(o.study_id[o.outcome == "Included"])
    check = check.merge(o[["study_id", "outcome"]], on="study_id", how="left")
    return ds, check


def excluded(o):
    """Compact table of the excluded participants: study_id, group, brand, reason, trainings."""
    return (o.loc[o.outcome != "Included", ["study_id", "group", "brand", "outcome", "trainings"]]
            .sort_values(["outcome", "study_id"]).rename(columns={"outcome": "reason"}))


def counts(df):
    """Flow (unique participants), per population, per group (HR runners with 2 races in both)."""
    extra = [o for o in df.outcome.unique() if o not in ORDER + ["Included"]]   # own labels from [exclusion]
    steps = ["Registered"] + ORDER + extra + ["Included"]
    sel = {o: df if o == "Registered" else df[df.outcome == o] for o in steps}
    flow = pd.DataFrame({"step": [s if s in ("Registered", "Included") else f"  excluded: {s.lower()}" for s in steps],
                         "participants": [len(sel[s]) * (1 if s in ("Registered", "Included") else -1) for s in steps]})
    pop = pd.DataFrame([{"": s, **{k: int(sel[s].study_id.str.match(p).sum()) for k, p in POP_ID.items()},
                         "total": len(sel[s])} for s in steps])
    g = df.assign(group=df.group.str.split(" & ")).explode("group")
    grp = pd.DataFrame([{"": s, **{k: int(((g.group == k) & ((g.outcome == s) | (s == "Registered"))).sum())
                                   for k in GROUPS}} for s in steps])
    return flow, pop, grp


def missing_location(trainings, included):
    """Trainings without a location (no GPS and no place name) -> list, per source and per participant."""
    t = trainings[trainings.study_id.isin(included)].copy()
    t["source"] = t.run_id.str.split("_").str[1]
    t["missing"] = t.location.isna() | (t.location.astype(str).str.strip() == "")
    m = t[t.missing].copy()
    short = (pd.to_numeric(m.duration_min, errors="coerce") < 1) | (pd.to_numeric(m.distance_km, errors="coerce") < 0.1)
    m["likely reason"] = np.where(short, "very short (< 1 min or < 0.1 km): accidental start / stop",
                                  m.source.map(NO_LOCATION).fillna("no position"))
    by_src = t.groupby("source").agg(trainings=("run_id", "size"), without_location=("missing", "sum"))
    by_src["% without"] = (100 * by_src.without_location / by_src.trainings).round(1)
    by_pp = t.groupby("study_id").agg(trainings=("run_id", "size"), without_location=("missing", "sum"),
                                      race_without_location=("missing", lambda x: int((x & (t.loc[x.index, "is_race"] == 1)).sum())))
    by_pp = by_pp[by_pp.without_location > 0].sort_values("without_location", ascending=False)
    by_pp["% without"] = (100 * by_pp.without_location / by_pp.trainings).round(1)
    cols = ["study_id", "run_id", "date", "start_time", "is_race", "duration_min", "distance_km", "source", "likely reason"]
    return m[cols], by_src.reset_index(), by_pp.reset_index()


def wrong_dates(trainings):
    """Trainings dated before 2000 (1 Jan 1970 / 31 Dec 1969 = empty or unreadable date in the export)."""
    d = pd.to_datetime(trainings.date, errors="coerce")
    w = trainings[d.dt.year < 2000].copy()
    w["likely reason"] = "date field empty or unreadable in the export (shown as 1 Jan 1970 / 31 Dec 1969)"
    return w[["study_id", "run_id", "date", "start_time", "duration_min", "distance_km", "likely reason"]]


def inclusion(df, trainings):
    """All sheets for export.save_inclusion: [(sheet name, [(title, note, df), ...]), ...].
    df = ds.outcomes (apply_exclusions, step 04)."""
    if df is None:
        raise SystemExit("No outcomes in the dataset: run notebook 04_build_dataset again.")
    flow, pop, grp = counts(df)
    inc = df[df.outcome == "Included"].study_id
    m, by_src, by_pp = missing_location(trainings, inc)
    w = wrong_dates(trainings)
    return [
        ("Flow", [("Inclusion and exclusion", "unique participants", flow), ("Per population", None, pop),
                  ("Per group", "healthy runners who ran both races are counted in MRHR and CPCHR", grp)]),
        ("Excluded", [(None, None, excluded(df))]),
        ("Missing location", [("Per source", "included participants; location = GPS city or place in the title", by_src),
                              ("Per participant", None, by_pp), ("All trainings without location", None, m)]),
        ("Dates 1969-1970", [(f"{len(w)} trainings with an impossible date", None, w)]),
        ("All participants", [(None, None, df)])]
