"""Tables for the report, as DataFrames (shown in the notebooks, saved to Excel).

  flow(res)                        flow of inclusion: n per step and per group, excluded per reason in between
  excluded(res)                    study numbers per exclusion reason and group (+ Total row with counts)
  ran_both(res, res_mr, res_cpc)   ran both races, excluded from one per-race analysis, included in the main one
  characteristics(res, ds)         included participants per group: gender, age, resting HR, highest HR, race, brand
  availability(ds)                 per parameter: participants with a value per group, missing records / 5-s windows
  checks(...)                      flow adds up, matches Excluded, included = 'Per participant', brands add up to n
Groups: RM and CPC = patients, HR = healthy runners, RMP and CPCP = pacers.
"""
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from ..analysis import compare
from ..dataset import has
from ..dataset.tables import weekly_daily
from ..settings import PROC

GROUPS = {"patients": "Patients", "healthy runners": "Healthy runners", "pacers": "Pacers"}
# (label in the analysis, label in the tables, flow step it is excluded at); first that applies = this order
# step 1 (usable export): nothing usable for the study; 2: no data around the race; 3: missing analysis parameters
REASONS = [("Export could not be opened", "Export could not be opened", 1),
           ("No relevant data in export", "No relevant data in export", 1),
           ("No running activity in study period", "No running activity in study period", 1),
           ("Race not part of the study", "Race not part of the study", 1),
           ("No data of the race", "No data of the race", 2),
           ("No training in ", "No training in the 14 days before the race", 2),     # compare.only_race()
           ("No resting HR", "No resting heart rate", 3),
           ("No training with HR and WBGT", "No training with both HR and WBGT", 3)]   # (start of the label)
STEPS = ["Registered (brand overview)", "Usable smartwatch export", "Data in the 14 days before the race",
         "Included in the analysis"]
BRANDS = {"Garmin (both)": "Garmin Connect + Internal", "Garmin": "Garmin Connect + Internal",
          "GarminC": "Garmin Connect only", "GarminI": "Garmin Internal only", "Apple": "Apple", "Polar": "Polar",
          "Fitbit": "Fitbit", "Coros": "Coros", "Strava": "Strava", "Samsung": "Samsung", "Huawei": "Huawei"}


# ------------------------------------------------------------------------------ helpers
def half_up(x, decimals=0):
    """Round half up (12.5 -> 13), not to even as round() does."""
    if x is None or x != x:
        return np.nan
    q = Decimal(1).scaleb(-decimals)
    r = Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)
    return int(r) if decimals == 0 else float(r)


def n_pct(n, total):
    """'n (%)', percentage rounded half up."""
    return f"{n} ({half_up(100 * n / total) if total else 0}%)"


def med_iqr(x, decimals=0):
    """'median (Q1-Q3)' rounded half up; '-' when empty."""
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return "-"
    f = lambda v: f"{half_up(v, decimals):.{decimals}f}"
    return f"{f(x.median())} ({f(x.quantile(.25))}-{f(x.quantile(.75))})"


def population(group):
    """Step 05 group (RM, CPC, MRHR & CPCHR, RMP, ...) -> patients / healthy runners / pacers."""
    g = str(group)
    return "pacers" if g.startswith(("RMP", "CPCP")) else "healthy runners" if "HR" in g else "patients"


def reason(label):
    """Analysis label -> (number 1-8, label in the tables, flow step). Unknown labels (other parameters) after 8."""
    for i, (start, name, step) in enumerate(REASONS, 1):
        if str(label).startswith(start):
            return i, name, step
    return len(REASONS) + 1, label, 3


def sort_ids(ids):
    """Study numbers by prefix, then number: A2 before A10."""
    def key(s):
        prefix, number = re.match(r"\D*", s).group(), re.search(r"\d+$", s)
        return prefix, float(number.group()) if number else np.nan
    return sorted(ids, key=key)


def _excluded(res):
    """Excluded participants with population, reason number, label in the tables and flow step."""
    e = res.excluded.copy()
    e["population"] = e.group.map(population)
    e[["nr", "reason_table", "step"]] = pd.DataFrame(e.reason.map(reason).tolist(), index=e.index)
    return e


# ------------------------------------------------------------------------------ a) flow
def flow(res):
    """One row per step (n total and per group), with the excluded per reason between the steps.
    res: compare.run(ds, s) for both races (every participant once)."""
    e, inc = _excluded(res), res.per_participant
    pops = pd.concat([e.population, inc.population])
    n = {g: int((pops == g).sum()) for g in GROUPS}
    rows = [{"": STEPS[0], **{GROUPS[g]: n[g] for g in GROUPS}, "Total": sum(n.values())}]
    for step in (1, 2, 3):
        for _, name in sorted(set(zip(e.nr[e.step == step], e.reason_table[e.step == step]))):   # in the order 1-8
            x = e[e.reason_table == name]
            c = {g: int((x.population == g).sum()) for g in GROUPS}
            rows.append({"": f"  excluded: {name[0].lower() + name[1:]}", **{GROUPS[g]: -c[g] for g in GROUPS},
                         "Total": -sum(c.values())})
            n = {g: n[g] - c[g] for g in GROUPS}
        rows.append({"": STEPS[step], **{GROUPS[g]: n[g] for g in GROUPS}, "Total": sum(n.values())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ b) excluded
def excluded(res):
    """One row per reason (in the order 1-8): the study numbers per group, + a Total row with the counts."""
    e = _excluded(res)
    rows = []
    for (nr, name), x in e.groupby(["nr", "reason_table"]):
        rows.append({"Reason": f"{nr} {name}" if nr <= len(REASONS) else name,
                     **{GROUPS[g]: ", ".join(sort_ids(x.study_id[x.population == g])) for g in GROUPS}})
    rows.append({"Reason": "Total", **{GROUPS[g]: str(int((e.population == g).sum())) for g in GROUPS}})
    return pd.DataFrame(rows)


def ran_both(res, res_mr, res_cpc):
    """Participants of both races excluded from one per-race analysis but included in the main analysis."""
    inc = res.per_participant.set_index("study_id")
    rows = []
    for race, other in (("MR", res_mr), ("CPC", res_cpc)):
        x = other.excluded[other.excluded.study_id.isin(inc.index)]
        for r in x.itertuples():
            rows.append({"study_id": r.study_id, "group": GROUPS[population(r.group)],
                         "excluded from": compare.RACE_NAME[race], "reason": reason(r.reason)[1],
                         "race used": compare.RACE_NAME[inc.race_used[r.study_id]]})
    return pd.DataFrame(rows, columns=["study_id", "group", "excluded from", "reason", "race used"])


# ------------------------------------------------------------------------------ c) characteristics
def characteristics(res, ds):
    """Included participants of the main analysis, per group (pooled over the races): n (%) or median (IQR)."""
    df = res.per_participant.copy()
    pt = ds.participants.set_index("study_id")
    rest = ds.trainings.groupby("study_id").resting_hr_week.median()      # = the median the HEI fallback uses
    df["resting_hr"] = df.study_id.map(rest)
    df["hr_max_observed"] = df.study_id.map(pt.hr_max_observed)
    df["hr_max_training"] = df.study_id.map(pt.hr_max_training) if "hr_max_training" in pt else np.nan
    df["gender"] = df.gender.str.lower().where(df.gender.str.lower().isin(["female", "male"]), "unknown")
    df["brand_table"] = df.brand.map(BRANDS).fillna(df.brand)
    grp = {g: df[df.population == g] for g in GROUPS}
    head = {g: f"{GROUPS[g]} (n={len(grp[g])})" for g in GROUPS}

    def counts(col, values):
        return [{"": f"  {v}", **{head[g]: n_pct(int((x[col] == v).sum()), len(x)) for g, x in grp.items()}}
                for v in values]

    def med(label, col):
        return {"": label, **{head[g]: med_iqr(x[col]) for g, x in grp.items()}}

    brands = [b for b in dict.fromkeys(BRANDS.values()) if (df.brand_table == b).any()] + \
        sorted(set(df.brand_table) - set(BRANDS.values()))
    races = [r for r in ("MR", "CPC") if (df.race_used == r).any()]
    rows = ([{"": "Gender, n (%)"}] + counts("gender", ["female", "male", "unknown"]) +
            [med("Age (years), median (IQR)", "age"),
             {"": "  age known, n", **{head[g]: str(int(x.age.notna().sum())) for g, x in grp.items()}},
             med("Resting HR (bpm), median (IQR)", "resting_hr"),
             med("Highest recorded HR (bpm), median (IQR)", "hr_max_observed"),
             med("  race excluded", "hr_max_training"),
             {"": "Race analyzed, n (%)"}] +
            [{"": f"  {compare.RACE_NAME[r]}", **{head[g]: n_pct(int((x.race_used == r).sum()), len(x))
                                                 for g, x in grp.items()}} for r in races] +
            [{"": "Smartwatch, n (%)"}] + counts("brand_table", brands))
    return pd.DataFrame(rows).fillna("")


# ------------------------------------------------------------------------------ d) availability
ROWS = [("Per training", "Time, distance and pace", "training", ["duration_min", "distance_km"]),
        ("Per training", "Mean heart rate", "hr", ["hr_mean"]),
        ("Per training", "Start location", "location", ["location"]),
        ("Per training", "WBGT", "wbgt", ["wbgt_mean"]),
        ("Within training", "Heart rate (5 s)", "hr_5s", ["hr_5s"]),
        ("Within training", "Speed and GPS position (5 s)", "gps_5s", ["gps_5s"]),
        ("Daily or weekly", "Resting heart rate per week", "resting_hr", []),
        ("Daily or weekly", "HRV per day", "hrv", []),
        ("Daily or weekly", "Step count per week", "steps", []),
        ("Participant", "Age", "age", []),
        ("Participant", "Gender", "gender", []),
        ("Participant", "Questionnaire", "questionnaire", [])]
PERIODIC = {"resting_hr": ("resting_hr_week", "W"), "steps": ("steps_week", "W"), "hrv": ("hrv", "D")}


def periods(t):
    """Per participant: first training -> race (last race when two; last training when no race)."""
    d = pd.to_datetime(t.date)
    race = d.where(t.is_race == 1).groupby(t.study_id).max()
    return pd.DataFrame({"first": d.groupby(t.study_id).min(), "last": race.fillna(d.groupby(t.study_id).max())})


def missing_periodic(ds, key):
    """% of weeks (resting HR, steps) or days (HRV) without a value, summed over the participants."""
    col, unit = PERIODIC[key]
    vals = weekly_daily(ds.daily)[col]
    have = set(vals.index) if len(vals) else set()
    expected = found = 0
    for sid, r in periods(ds.trainings).iterrows():
        if unit == "W":
            slots = pd.date_range(r["first"] - pd.Timedelta(days=r["first"].weekday()), r["last"], freq="W-MON")
        else:
            slots = pd.date_range(r["first"], r["last"], freq="D")
        expected += len(slots)
        found += sum((sid, s) in have for s in slots)
    return 100 * (1 - found / expected) if expected else np.nan


def missing_windows(five_s, col):
    """Trainings with 5-s data: expected windows = (last - first second) / step + 1 per run; % without a value."""
    g = five_s.groupby("run_id")
    step = PROC.get("stream_seconds", 5)
    expected = ((g.second.max() - g.second.min()) / step + 1).sum()
    return 100 * (1 - five_s[col].notna().sum() / expected) if expected else np.nan


def availability(ds):
    """Per parameter: participants with >= 1 value per group and in total, % missing records, % missing 5-s
    windows (within-training HR and GPS). Rows whose column is not in the dataset are skipped (printed)."""
    pt, t = ds.participants, ds.trainings
    rows = []
    for cat, label, key, cols in ROWS:
        absent = ([key] if key not in pt.columns else []) + [c for c in cols if c not in t.columns]
        if absent:
            print(f"availability: '{label}' skipped, not in the dataset: {', '.join(absent)}")
            continue
        h = has(pt, key)
        row = {"Category": cat, "Parameter": label,
               **{GROUPS[g]: int((h & (pt.population == g)).sum()) for g in GROUPS}, "Total": int(h.sum())}
        if cat in ("Per training", "Within training"):
            ok = t[cols].notna().all(axis=1) if cat == "Per training" else t[cols[0]].fillna(False).astype(bool)
            if key == "location":
                ok &= t.location.astype(str).str.strip().ne("")
            row["Missing records (%)"] = 100 * (1 - ok.mean())
        elif cat == "Daily or weekly":
            row["Missing records (%)"] = missing_periodic(ds, key)
        else:
            row["Missing records (%)"] = 100 * (1 - h.mean())
        if key in ("hr_5s", "gps_5s"):
            row["Missing 5-s windows (%)"] = missing_windows(ds.five_s, "hr" if key == "hr_5s" else "lat")
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ("Missing records (%)", "Missing 5-s windows (%)"):
        if c in out:
            out[c] = out[c].map(lambda v: half_up(v, 1))
    return out


# ------------------------------------------------------------------------------ checks
def checks(res, flow_df, excluded_df, char_df, per_participant_file=None):
    """Print: flow adds up and matches Excluded; included = 'Per participant' sheet; brands add up to n."""
    problems = []
    groups, cols = list(GROUPS.values()), list(GROUPS.values()) + ["Total"]
    step_rows = flow_df[flow_df[""].isin(STEPS)].index
    for a, b in zip(step_rows[:-1], step_rows[1:]):
        if not (flow_df.loc[a, cols] + flow_df.loc[a + 1:b - 1, cols].sum() == flow_df.loc[b, cols]).all():
            problems.append(f"FLOW DOES NOT ADD UP: {flow_df.loc[a, '']} -> {flow_df.loc[b, '']}")
    total = excluded_df.set_index("Reason").loc["Total", groups].astype(int)
    lost = -flow_df[~flow_df[""].isin(STEPS)][groups].sum()
    if not (total == lost).all():
        problems.append(f"FLOW != EXCLUDED TABLE: flow {lost.to_dict()}, Excluded {total.to_dict()}")
    listed = excluded_df[excluded_df.Reason != "Total"][groups]
    n_listed = listed.map(lambda s: len([x for x in s.split(", ") if x])).sum()
    if not (n_listed == total).all():
        problems.append(f"EXCLUDED: study numbers listed {n_listed.to_dict()} != Total row {total.to_dict()}")
    inc = set(res.per_participant.study_id)
    if per_participant_file is not None:
        saved = set(pd.read_excel(per_participant_file, sheet_name="Per participant").study_id)
        if saved != inc:
            problems.append(f"INCLUDED != 'Per participant' sheet: only in table {sorted(inc - saved)}, "
                            f"only in sheet {sorted(saved - inc)}")
    if flow_df.iloc[-1]["Total"] != len(inc):
        problems.append(f"FLOW: included {flow_df.iloc[-1]['Total']} != {len(inc)} participants in the analysis")
    i = char_df.index[char_df[""] == "Smartwatch, n (%)"][0]
    for c in char_df.columns[1:]:
        n = int(c.split("n=")[1].rstrip(")"))
        s = char_df.loc[i + 1:, c].map(lambda v: int(v.split(" ")[0])).sum()
        if s != n:
            problems.append(f"BRANDS: {c} adds up to {s}, not {n}")
    for p in problems:
        print(p)
    print("some checks FAILED (see above)" if problems else "all checks OK")
    return not problems
