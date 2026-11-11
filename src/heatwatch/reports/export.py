"""Saving: styled Excel (openpyxl) and the dataset / report / selection files.

  save_dataset(ds)             dataset/: participants.csv, trainings.csv, questionnaire.csv, 5s/<study_id>.csv,
                               HEATWATCH.xlsx (Read me | Overview | Availability | Participants | Trainings | Questionnaire),
                               HEATWATCH.pkl (to continue in the next notebook)
  save_report_tables(tables)   extra sheets (Flow, Excluded, Characteristics) in analysis/HEATWATCH_analysis.xlsx
  save_inclusion(sheets)       reports/HEATWATCH_inclusion_exclusion.xlsx
  save_selection(sel, params)  selections/selection_<params>.xlsx (+ _5s.csv)
"""
import pandas as pd

from ..excel import add_sheets, write
from ..settings import ANALYSIS, DATASET, PROC, REPORTS, SELECTIONS


# ------------------------------------------------------------------------------ files
def save_dataset(ds):
    """Also saves HEATWATCH.pkl (dataset.load() reads it back). Excluded participants: report of step 05."""
    from .overview import overview_tables
    from .paper_tables import availability
    par, devices, last = overview_tables(ds.participants)
    (DATASET / "5s").mkdir(parents=True, exist_ok=True)
    ds.participants.to_csv(DATASET / "participants.csv", index=False)
    ds.trainings.to_csv(DATASET / "trainings.csv", index=False)
    pd.to_pickle(ds, DATASET / "HEATWATCH.pkl")
    q = getattr(ds, "questionnaire", None)
    if q is not None and len(q):
        q.to_csv(DATASET / "questionnaire.csv", index=False)
    for sid, g in ds.five_s.groupby("study_id"):
        g.to_csv(DATASET / "5s" / f"{sid}.csv", index=False)
    out = write(DATASET / "HEATWATCH.xlsx", readme(), [
        ("Overview", [("Participants with each parameter", None, par), ("Smartwatch", None, devices),
                      ("Trainings in the 14 days before the race", None, last)]),
        ("Availability", [("Availability and completeness per parameter", "participants with >= 1 value; missing "
                           "records: trainings / weeks / days / participants without the value; missing 5-s "
                           "windows: in trainings with 5-s data", availability(ds))]),
        ("Participants", [(None, None, ds.participants)]),
        ("Trainings", [(None, None, ds.trainings)]),
        ("Questionnaire", [(None, None, q if q is not None and len(q) else pd.DataFrame({"": ["no questionnaire file"]}))])])
    print(f"dataset -> {DATASET}  ({out.name}, participants.csv, trainings.csv, 5s/)")
    return out


def save_report_tables(tables, race=None):
    """Add the report tables (dict sheet name -> [(title, note, df), ...]) to HEATWATCH_analysis(.xlsx), written by
    compare.run(save=True) - run that first."""
    path = ANALYSIS / f"HEATWATCH_analysis{'_' + race if race else ''}.xlsx"
    if not path.exists():
        raise SystemExit(f"{path.name} not found: run compare.run(ds, s) (save=True) first.")
    out = add_sheets(path, list(tables.items()))
    print(f"report tables ({', '.join(tables)}) -> {out}")
    return out


def save_inclusion(sheets):
    """Inclusion / exclusion report -> reports/HEATWATCH_inclusion_exclusion.xlsx."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = write(REPORTS / "HEATWATCH_inclusion_exclusion.xlsx", None, sheets)
    print(f"report -> {out}")
    return out


def save_selection(selection, params):
    """Selection -> selections/selection_<params>.xlsx (+ 5-s data as .csv)."""
    sel_p, sel_t, sel_f = selection
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    name = "selection_" + "_".join(params)
    out = write(SELECTIONS / f"{name}.xlsx", None, [("participants", [(None, None, sel_p)]),
                                                     ("trainings", [(None, None, sel_t)])])
    if sel_f is not None:
        sel_f.to_csv(SELECTIONS / f"{name}_5s.csv", index=False)
    print(f"{name}: {len(sel_p)} participants, {len(sel_t)} trainings"
          + (f", {len(sel_f)} rows of 5-s data" if sel_f is not None else "") + f" -> {SELECTIONS}")
    return out


# ------------------------------------------------------------------ read me of the dataset file
def readme():
    """Text of the 'Read me' sheet of HEATWATCH.xlsx."""
    return [
        "Tables, linked by keys",
        "  participants   1 row per participant                key: study_id",
        "  trainings      1 row per training                   key: run_id  (e.g. A07_fit_20260412095931 = study_id_source_start)",
        f"  5s             1 row per {PROC['stream_seconds']} s inside a training     key: run_id + second (seconds since the start)",
        "  questionnaire  1 row per participant per race       key: study_id + race",
        "  trainings -> participants: study_id.   trainings -> 5s: run_id  (5s\\<study_id>.csv)",
        "  Only included participants (excluded ones and the reasons: report of notebook 05)",
        "",
        "Overview",
        "  Per population, number of participants: with each parameter, smartwatch,",
        "  trainings in the 14 days before the race",
        "",
        "Participants",
        "  One column per parameter = number of trainings / weeks / days with it (0 = none)",
        "  gender: questionnaire (Gender), otherwise the device (gender_source says which)",
        "  questionnaire: number of questionnaires (1 per race)",
        "  hr_max_observed: highest hr_max of any training (incl. race)",
        "  hr_max_training: highest hr_max of any training except the race",
        "  trainings_14d: trainings in the 14 days before the (first) race, race not counted",
        "",
        "Trainings",
        "  race: race the training prepares for; is_race = 1 for the race itself (longest run on race day)",
        f"  parts: runs merged into this training (a run starting <= {PROC.get('merge_gap', 120)} s after the end of the previous one)",
        "  speed_ms = distance / duration",
        "  location: place in the Garmin Connect title, otherwise city of the start position ('Marathon' on a race day = race city); 'indoor' = treadmill / indoor run",
        "  hr_mean / hr_max: mean and highest HR of the training, as summarised by the device",
        "  resting_hr_week: Garmin Connect: the weekly value of the 7-day block that contains the training day;",
        "                   Apple, Polar, Fitbit: mean of the Monday-Sunday week (Polar, Fitbit: lowest 30-min mean of all-day HR per day)",
        "  steps_week: Garmin Connect: the 7-day block that contains the training day; other devices: sum of the Monday-Sunday week",
        "  hrv_ms: that day (Garmin RMSSD, Apple SDNN: different measures)",
        "  n_5s: number of 5-s rows; hr_5s / gps_5s: HR / position present in them",
        "  wbgt_mean / wbgt_max: WBGT (Liljegren, as KNMI) during the training, every 10 min from start to finish at the",
        "               start position, weather from Open-Meteo; empty for trainings without GPS position or place name.",
        "               Indoor runs: 0.7 Tnwb + 0.3 Tg at 19.5 C, RH 40 %, 0.2 m/s (no sun)",
        "  heat_dose: HD = duration (min) x effort x max(0, wbgt_mean - 14); effort = dHR x Banister weighting (gender),",
        "             dHR = (hr_mean - resting HR) / (hr_max_observed - resting HR), resting HR = resting_hr_week, else the",
        "             participant's median; empty when mean HR, WBGT or resting HR is missing; 0 = WBGT not above 14 C",
        "  heat_dose_hrmax_training: the same with HR max without the race (hr_max_training)",
        "",
        "5s",
        "  Mean HR and speed of the window, first position. Strava: no HR at all (training + GPS only)",
    ]
