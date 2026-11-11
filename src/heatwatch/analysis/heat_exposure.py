"""Heat exposure index (HEI): how much heat a participant was exposed to during recorded training before
the race. A proxy for the heat stimulus, NOT a measure of physiological heat adaptation.

Per training i (indoor runs included, with their estimated gym WBGT):
    dHR_i = (HR_mean,i - HR_rest) / (HR_max - HR_rest), clipped to 0-1      fraction of heart-rate reserve
            HR_rest = resting HR of the training's week (else the participant's median weekly resting HR)
            HR_max  = hr_max_observed: highest HR of any training of the participant (incl. the race);
                      or hr_max_training: HR max without the race (HR_MAX; dataset column heat_dose_hrmax_training)
    y_i   = 0.64 exp(1.92 dHR_i) men, 0.86 exp(1.67 dHR_i) women, mean of both when gender is unknown
    E_i   = dHR_i * y_i                                                     effort (Banister TRIMP per minute)
    HD_i  = D_i * E_i * max(0, WBGT_mean,i - 14)                           heat dose, D_i in minutes
HEI on race day t:  HEI = sum over trainings before the race day of HD_i * exp(-decay * (t - t_i))
One race per participant (`main` when both were run), with the trainings in the `days` days before it.
decay, days and main: the Settings of notebook 06 (compare.Settings). The WBGT threshold (14) is [weather]
wbgt_threshold in settings.toml, as the heat dose is also part of the dataset (step 04). Trainings without mean HR, WBGT or resting HR get no heat dose. HEI = 0: no training above the
WBGT threshold (not exposed, still a value).
"""
import numpy as np
import pandas as pd

from ..settings import CFG

THRESHOLD = float(CFG["weather"]["wbgt_threshold"])
HR_MAX = {"hr_max_observed": "HR max incl. race", "hr_max_training": "HR max without race"}   # column: label


# ------------------------------------------------------------------ race used and the trainings before it
def chosen_races(trainings, main, race=None):
    """1 row per participant with a recorded race: the race used (date, start time, which race).
    race: only that race (analysis per race); None: `main` when both were run, else the one run."""
    r = trainings[trainings.is_race == 1].copy()
    if race:
        r = r[r.race == race]
    r["date"] = pd.to_datetime(r.date)
    r["other"] = r.race != main
    r = r.sort_values(["other", "date", "start_time"]).groupby("study_id").first()
    return r[["population", "group", "race", "date", "start_time"]].rename(
        columns={"race": "race_used", "date": "race_date", "start_time": "race_start"})


def before_race(trainings, races, days):
    """Trainings (race itself not included) in the `days` days before the participant's race used."""
    t = trainings[(trainings.is_race == 0) & trainings.study_id.isin(races.index)].copy()
    t["date"] = pd.to_datetime(t.date)
    t = t.join(races[["race_used", "race_date", "race_start"]], on="study_id")
    keep = (t.race == t.race_used) & (t.date < t.race_date) & (t.date >= t.race_date - pd.Timedelta(days=days))
    return t[keep]


# ------------------------------------------------------------------ heat dose per training
def effort(dhr, gender):
    """Banister weighting: E = dHR * y; y for men, women, or the mean of both when gender is unknown."""
    y_m, y_f = 0.64 * np.exp(1.92 * dhr), 0.86 * np.exp(1.67 * dhr)
    y = np.where(gender == "male", y_m, np.where(gender == "female", y_f, (y_m + y_f) / 2))
    return dhr * y


def doses(t, trainings, participants, hr_max="hr_max_observed"):
    """Heat dose of the trainings t (all columns + hr_rest, hr_max_used, hr_rest_fallback, dHR, effort,
    wbgt_excess, heat_dose) and a mask of the trainings that have one (mean HR, WBGT and resting HR known).
    trainings: all trainings, for the participant's median resting HR (fallback).
    hr_max: column of participants used as HR max (HR_MAX)."""
    pt = participants.set_index("study_id")
    hr_rest = t.resting_hr_week.fillna(t.study_id.map(trainings.groupby("study_id").resting_hr_week.median()))
    hr_max = t.study_id.map(pt[hr_max])
    ok = t.hr_mean.notna() & t.wbgt_mean.notna() & hr_rest.notna() & (hr_max > hr_rest)
    t = t.assign(hr_rest=hr_rest.where(ok), hr_max_used=hr_max.where(ok), hr_rest_fallback=t.resting_hr_week.isna() & ok)
    t["dHR"] = ((t.hr_mean - t.hr_rest) / (t.hr_max_used - t.hr_rest)).clip(0, 1)
    t["effort"] = effort(t.dHR.to_numpy(), t.study_id.map(pt.gender).to_numpy())
    t["wbgt_excess"] = (t.wbgt_mean - THRESHOLD).clip(lower=0).where(ok)
    t["heat_dose"] = t.duration_min * t.effort * t.wbgt_excess
    return t, ok


def heat_doses(trainings, participants, races, days, decay, hr_max="hr_max_observed"):
    """Heat dose per training in the `days` days before the race used (races: chosen_races(); all columns of
    the training + dHR, effort, HD, HEI part with `decay` per day); only trainings with a heat dose."""
    t, ok = doses(before_race(trainings, races, days), trainings, participants, hr_max)
    t = t[ok].copy()
    t["days_before_race"] = (t.race_date - t.date).dt.days
    t["hei_part"] = t.heat_dose * np.exp(-decay * t.days_before_race)
    return t


# ------------------------------------------------------------------ heat exposure index (HEI) and resting HR fallback
def hei(trainings, participants, races, days, decay, hr_max="hr_max_observed"):
    """Per participant with a race and >= 1 training with a heat dose: HEI on race day, exposed (HEI > 0),
    number of trainings with a heat dose and with WBGT above the threshold."""
    hd = heat_doses(trainings, participants, races, days, decay, hr_max)
    g = hd.groupby("study_id")
    out = pd.DataFrame({"HEI": g.hei_part.sum().round(1), "trainings_with_dose": g.size(),
                        "trainings_above_threshold": g.wbgt_excess.apply(lambda x: int((x > 0).sum()))})
    out["exposed"] = out.HEI > 0
    return out


def fallback(hd):
    """Resting HR fallback (no resting HR for the training's week -> the participant's median used).
    Returns (1 row per participant with a heat dose, the trainings that used the fallback)."""
    g = hd.groupby("study_id")
    pp = pd.DataFrame({"population": g.population.first(), "trainings_with_dose": g.size(),
                       "with_fallback": g.hr_rest_fallback.sum().astype(int)})
    pp["% fallback"] = (100 * pp.with_fallback / pp.trainings_with_dose).round(0).astype(int)
    pp["only_fallback"] = pp.with_fallback == pp.trainings_with_dose
    pp["median_resting_hr_used"] = hd[hd.hr_rest_fallback].groupby("study_id").hr_rest.first()
    pp = pp.reset_index().sort_values(["only_fallback", "% fallback", "study_id"], ascending=[False, False, True])
    cols = ["study_id", "population", "run_id", "date", "start_time", "duration_min", "hr_mean", "resting_hr_week",
            "hr_rest", "wbgt_mean", "heat_dose"]
    return pp.reset_index(drop=True), hd.loc[hd.hr_rest_fallback, cols].reset_index(drop=True)
