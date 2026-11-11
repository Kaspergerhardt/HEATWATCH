"""The dataset: three linked tables.

  participants   1 row per participant: group, race, device, gender, age; per parameter the number of
                 trainings / weeks / days with it (0 = none); race present; days of data before the race
  trainings      1 row per training, every parameter as a column                   key: run_id (-> study_id)
  five_s         1 row per 5 s inside a training: lat, lon, speed, HR             key: run_id + second
select() gives the participants with ticked parameters, their trainings and (with hr_5s / gps_5s)
the matching 5-s rows.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import combine
from ..loading import extract, participants as pp, questionnaire as qn
from .location import place_coords, place_lookup, place_names
from ..loading.participants import POP
from ..loading.readers.common import is_indoor
from ..settings import CFG, DATASET, PROC

PARAMETERS = ["training", "location", "hr", "hr_5s", "gps_5s", "resting_hr", "steps", "hrv", "wbgt", "gender", "age",
              "questionnaire"]
STREAMS = {"hr_5s", "gps_5s"}
BASE = ["study_id", "population", "group", "race", "run_id", "date", "start_time", "is_race",
        "duration_min", "distance_km", "speed_ms"]
COLUMNS = {"location": ["location"], "hr": ["hr_mean", "hr_max"], "hr_5s": ["n_5s"], "gps_5s": ["n_5s"],
           "resting_hr": ["resting_hr_week"], "steps": ["steps_week"], "hrv": ["hrv_ms"],
           "wbgt": ["wbgt_mean", "wbgt_max", "heat_dose", "heat_dose_hrmax_training"],
           "gender": ["gender"], "age": ["age"], "questionnaire": []}


# ------------------------------------------------------------------ the dataset object
@dataclass
class Dataset:
    """The three linked tables + what is needed to recount and check them."""
    participants: pd.DataFrame          # 1 row per participant (+ availability per parameter)
    trainings: pd.DataFrame             # 1 row per training
    five_s: pd.DataFrame                # 1 row per 5 s inside a training
    daily: pd.DataFrame = field(repr=False)        # resting HR / steps / HRV values (for the tables above)
    info: pd.DataFrame = field(repr=False)         # participant info from the brand overview (+ age / gender)
    where: pd.DataFrame = field(repr=False)        # start position / time / indoor per training (heat stress)
    questionnaire: pd.DataFrame = field(default=None, repr=False)   # 1 row per participant per race (Castor)
    outcomes: pd.DataFrame = field(default=None, repr=False)        # step 05 outcome per participant of the brand overview

    def exclude(self, keep):
        """Keep only these study numbers in every table."""
        keep = set(keep)
        for name in ("participants", "trainings", "five_s", "daily", "info", "questionnaire"):
            d = getattr(self, name)
            if d is not None:
                setattr(self, name, d[d.study_id.isin(keep)].reset_index(drop=True))
        self.where = self.where[self.where.run_id.isin(self.trainings.run_id)].reset_index(drop=True)
        return self

    def refresh(self):
        """Recount the participant table (e.g. after heat stress was added to the trainings)."""
        self.participants = participants_table(self.info, self.trainings, self.daily, self.questionnaire)
        return self

    def __repr__(self):
        return (f"Dataset: {len(self.participants)} participants, {len(self.trainings)} trainings, "
                f"{len(self.five_s)} rows of 5-s data")


# ------------------------------------------------------------------ build from the cache / load the saved dataset
def build():
    """Cache of step 2 -> Dataset (without heat stress: heat_stress.add(ds) adds it)."""
    info = pp.load()
    p, s, s5, d = combine.combine(info, extract.load_cache())
    p = p[p.study_id.isin(s.study_id) | p.study_id.isin(d.study_id)]       # participants with any data
    q = qn.read(study_ids=info.study_id)
    p = qn.add_gender(p, q)                                                    # gender: questionnaire, else device
    t, where = trainings(p, s, d, s5)
    return Dataset(participants_table(p, t, d, q), t, five_seconds(s5), d, p, where, q)


def load():
    """The dataset as saved by export.save_dataset (to continue in another notebook)."""
    f = DATASET / "HEATWATCH.pkl"
    if not f.exists():
        raise SystemExit("No dataset saved yet: run notebook 04_build_dataset first.")
    return pd.read_pickle(f)


# ------------------------------------------------------------------ trainings table: 1 row per training (location, weekly / daily values, positions)
def weekly_daily(d):
    """resting HR per week (mean), steps per week (sum), HRV per day (mean; Garmin RMSSD, Apple SDNN).
    Garmin Connect gives resting HR and steps as 1 value per 7-day block starting on any weekday:
    <metric>_block = that value on every day of the block; <metric>_daily = the other sources (1 value per day)
    per calendar week (resting HR: mean, steps: sum). <metric>_week = all values per calendar week (for counting)."""
    keys = [f"{m}_{k}" for m in ("resting_hr", "steps") for k in ("week", "block", "daily")] + ["hrv"]
    if not len(d):
        return {k: pd.Series(dtype=float) for k in keys}
    d = d.assign(week=d.start.dt.normalize() - pd.to_timedelta(d.start.dt.weekday, unit="D"), day=d.start.dt.normalize())
    out = {"hrv": d[d.metric.str.startswith("hrv")].groupby(["study_id", "day"]).value.mean()}
    for metric, agg in (("resting_hr", "mean"), ("steps", "sum")):
        r = d[d.metric == metric]
        block = r.end.dt.normalize() > r.start.dt.normalize()
        spread = r[block].assign(day=[list(pd.date_range(a, b)) for a, b in
                                      zip(r[block].start.dt.normalize(), r[block].end.dt.normalize())]).explode("day")
        out[f"{metric}_week"] = r.groupby(["study_id", "week"]).value.agg(agg)
        out[f"{metric}_block"] = spread.groupby(["study_id", "day"]).value.mean()
        out[f"{metric}_daily"] = r[~block].groupby(["study_id", "week"]).value.agg(agg)
    return out


def locations(s, day):
    """Location per run: known place in the Garmin Connect title ('Soest Track' -> Soest, 'Lunch' -> none),
    otherwise the city of the GPS start; 'Marathon' on the race day = the race city ([race_city]), on other days
    ignored (Marathon is also a town in Greece / Florida); no place and no position but an indoor run ->
    'indoor'. Returns (location, indoor)."""
    found = s.location.map(place_lookup)
    title = found.map(lambda r: r[0] if r else None)
    marathon = title.astype(str).str.lower().eq("marathon")
    race_date = s.race.map(lambda r: pd.Timestamp(CFG["races"][r]) if r in CFG["races"] else pd.NaT)
    race_city = s.race.map(CFG.get("race_city", {})).where(day == race_date)
    title = title.where(~marathon)
    loc = title.where(title.notna(), pd.Series(place_names(s.lat.tolist(), s.lon.tolist()), index=s.index))
    loc = loc.where(~(marathon & race_city.notna()), race_city)
    # indoor: .fit / Coros sub_sport, Apple HKIndoorWorkout, Garmin Connect 'Loopband', Strava 'Virtual Run'
    indoor = s.get("indoor", pd.Series(False, index=s.index)).fillna(False).astype(bool) | s.activity.map(is_indoor)
    return loc.where(loc.notna() | ~indoor, "indoor"), indoor


def weather_positions(s, loc, indoor):
    """For heat_stress.add: start position (or the coordinates of the place in the title), UTC start, indoor,
    duration per run."""
    named = s.lat.isna() & loc.notna() & ~indoor
    xy = pd.DataFrame(place_coords(loc[named].tolist()), index=loc[named].index, columns=["lat", "lon"])
    lat, lon = s.lat.copy(), s.lon.copy()
    lat[named], lon[named] = xy.lat, xy.lon
    start_utc = s.start.dt.tz_localize(PROC["local_timezone"], ambiguous="NaT", nonexistent="shift_forward") \
        .dt.tz_convert("UTC").dt.tz_localize(None)
    return pd.DataFrame({"run_id": s.session_id, "lat": lat, "lon": lon, "start_utc": start_utc,
                         "indoor": indoor, "duration_s": s.duration_s})


def trainings(p, s, d, s5):
    """One row per training with every parameter -> (trainings, positions for the heat-stress step)."""
    s = s.merge(p[["study_id", "population", "group", "gender", "age"]], on="study_id")
    first = s5.dropna(subset=["lat"]).groupby("session_id")[["lat", "lon"]].first()   # start from 5-s data
    s["lat"] = s.lat.fillna(s.session_id.map(first.lat))
    s["lon"] = s.lon.fillna(s.session_id.map(first.lon))
    week = s.start.dt.normalize() - pd.to_timedelta(s.start.dt.weekday, unit="D")
    day = s.start.dt.normalize()
    v = weekly_daily(d)
    kw, kd = list(zip(s.study_id, week)), list(zip(s.study_id, day))
    loc, indoor = locations(s, day)
    n5 = s5.groupby("session_id").size()
    t = pd.DataFrame({
        "study_id": s.study_id, "population": s.population.map(POP), "group": s.group, "race": s.race,
        "run_id": s.session_id, "date": day.dt.date, "start_time": s.start.dt.strftime("%H:%M"),
        "is_race": (s.session_type == "race").astype(int), "duration_min": (s.duration_s / 60).round(2), "parts": s.get("parts", 1),
        "distance_km": s.distance_km.round(3), "speed_ms": (s.distance_km * 1000 / s.duration_s).round(3),
        "location": loc, "hr_mean": pd.to_numeric(s.hr_mean, errors="coerce").round(1),
        "hr_max": pd.to_numeric(s.hr_max, errors="coerce").round(0),
        # resting HR / steps: the Garmin Connect block that contains the training day, else the calendar week
        "resting_hr_week": np.round(np.array([v["resting_hr_block"].get(a, v["resting_hr_daily"].get(b, np.nan))
                                              for a, b in zip(kd, kw)], dtype=float), 1),
        "steps_week": [v["steps_block"].get(a, v["steps_daily"].get(b)) for a, b in zip(kd, kw)],
        "hrv_ms": np.round(np.array([v["hrv"].get(k, np.nan) for k in kd], dtype=float), 1),
        "n_5s": s.session_id.map(n5), "hr_5s": s.n_5s_hr > 0, "gps_5s": s.n_5s_gps > 0,
        "gender": s.study_id.map(p.set_index("study_id").gender), "age": s.age})     # gender: questionnaire, else device
    where = weather_positions(s, loc, indoor)
    order = t.sort_values(["study_id", "date", "start_time"]).index
    return t.loc[order].reset_index(drop=True), where.loc[order].reset_index(drop=True)


# ------------------------------------------------------------------ 5-s table and participants table
def five_seconds(s5):
    """5-s windows -> table with run_id, second, position, speed and HR."""
    f = s5.rename(columns={"session_id": "run_id", "elapsed_s": "second"})
    f.insert(0, "study_id", f.run_id.str.split("_").str[0])
    return f[["study_id", "run_id", "second", "lat", "lon", "speed_ms", "hr"]].round(
        {"lat": 6, "lon": 6, "speed_ms": 3, "hr": 1})


def participants_table(p, t, d, q=None):
    """1 row per participant: per parameter the number of trainings / weeks / days with it (0 = none),
    race present, the number of trainings in the 14 days before the (first) race, and the highest HR of any
    training (hr_max_observed) and of any training except the race (hr_max_training)."""
    g = t.groupby("study_id")
    v = weekly_daily(d)
    n = pd.DataFrame({"training": g.size(), "location": g.location.count(), "hr": g.hr_mean.count(),
                      "hr_5s": g.hr_5s.sum(), "gps_5s": g.gps_5s.sum()})
    for k, key in (("resting_hr", "resting_hr_week"), ("steps", "steps_week"), ("hrv", "hrv")):
        n[k] = v[key].groupby(level=0).size() if len(v[key]) else 0
    n["wbgt"] = g.wbgt_mean.count() if "wbgt_mean" in t else 0
    n["questionnaire"] = q.groupby("study_id").size() if q is not None and len(q) else 0
    n = n.reindex(p.study_id).fillna(0).astype(int)
    out = p[["study_id", "population", "group", "races", "brand", "gender", "age"]].copy()
    out.insert(out.columns.get_loc("gender") + 1, "gender_source", p.get("gender_source"))
    out["population"] = out.population.map(POP)
    out = out.join(n, on="study_id")
    out["hr_max_observed"] = out.study_id.map(g.hr_max.max()) if "hr_max" in t else np.nan
    out["hr_max_training"] = out.study_id.map(t[t.is_race == 0].groupby("study_id").hr_max.max())         if "hr_max" in t else np.nan                                   # race left out (may not be representative)
    races = g.is_race.sum().reindex(out.study_id).fillna(0).values
    out["race_present"] = races > 0
    # trainings (race not counted) in the 14 days before the first race
    race_day = out.races.str.split(";").map(lambda r: min((pd.Timestamp(CFG["races"][x]) for x in r if x), default=pd.NaT))
    td = pd.to_datetime(t.date)
    rd = t.study_id.map(dict(zip(out.study_id, race_day)))
    last = t[(t.is_race == 0) & (td < rd) & (td >= rd - pd.Timedelta(days=14))]
    out["trainings_14d"] = out.study_id.map(last.groupby("study_id").size()).fillna(0).astype(int)
    return out.reset_index(drop=True)


# ------------------------------------------------------------------ selection: participants with all ticked parameters
def has(pt, k):
    """Per participant: at least one value of parameter k (gender / age: known)."""
    return (pt[k] > 0) if k not in ("gender", "age") else pt[k].notna()


def select(pt, t, f, params):
    """Participants with ALL ticked parameters, their trainings (only those with 5-s data when hr_5s /
    gps_5s is ticked) with the matching columns, and their 5-s rows (or None)."""
    ok = np.logical_and.reduce([has(pt, k) for k in params]) if params else np.ones(len(pt), bool)
    ids = pt.study_id[ok]
    r = t[t.study_id.isin(ids)]
    for k in STREAMS & set(params):
        r = r[r[k]]
    cols = BASE + [c for k in params for c in COLUMNS.get(k, [])]
    r = r[list(dict.fromkeys(cols))]
    five = f[f.run_id.isin(r.run_id)] if STREAMS & set(params) else None
    return pt[ok], r, five
