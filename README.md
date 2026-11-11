# HEATWATCH dataset

Smartwatch exports per study number → one dataset (participants, trainings, 5-s data) with location,
WBGT per training, a data check and the inclusion / exclusion overview.
Only study numbers are stored: no names, file names or paths.

## Folders

```
HEATWATCH/
├── config/
│   ├── settings.toml          study settings
│   ├── settings.local.example.toml  template for your paths (copy to settings.local.toml)
│   └── settings.local.toml    your own paths (not on GitHub)
├── notebooks/                 step by step pipeline
│   ├── 01_participants.ipynb          group, race, device, data folder found?
│   ├── 02_inspect_raw.ipynb           data inspection, what is in the data?
│   ├── 03_window.ipynb                cut raw to [window] months_before → interim, read → cache (.pkl file for speed)
│   ├── 04_build_dataset.ipynb         combine, location, WBGT, data check → dataset/
│   ├── 05_inclusion_exclusion.ipynb   flow, excluded, missing location, wrong dates → reports/
│   ├── 06_analysis.ipynb              heat exposure + training timing per group, 9 boxplots → analysis/
│   └── 07_route_maps.ipynb            route of a run coloured by HR and pace → analysis/route_maps/
├── src/heatwatch/             the functions the notebooks use
│   ├── settings.py            reads config/, output folders
│   ├── excel.py               styled Excel output
│   ├── loading/               01–03
│   │   ├── participants.py        brand overview, groups, data folders
│   │   ├── inspect_raw.py         what is in the raw folders and how it is written (step 02)
│   │   ├── extract.py             read the exports → cache
│   │   ├── window.py              cut the raw exports to the data window → interim 
│   │   ├── questionnaire.py       Castor questionnaires
│   │   └── readers/               one reader per brand
│   ├── dataset/               04
│   │   ├── combine.py             merge Garmin Connect + .fit
│   │   ├── tables.py              the three tables, select(), load()
│   │   └── location.py            GPS → municipality
│   ├── heat_stress/           04
│   │   ├── weather.py             Open-Meteo weather → WBGT per training
│   │   └── wbgt.py                WBGT calculation, indoor estimate
│   ├── reports/               04–05
│   │   ├── overview.py            parameter tables, data check, inclusion / exclusion
│   │   └── export.py              save dataset / report / selection
│   └── analysis/              06–07
│       ├── heat_exposure.py       heat dose per training, heat exposure index (HEI) on race day
│       ├── compare.py             outcomes per participant, statistics, boxplots
│       └── routes.py              route maps coloured by HR and pace
├── run_pipeline.py            all steps at once (same as notebooks 01, 03–05; cutting only with --window; no analysis)
└── requirements.txt
```

## Output

Everything is written inside `output_root` (e.g. `data\processed`). Every folder can be rebuilt by
running its step again.

```
cache/       <study_id>.pkl, <study_id>.xlsx                    step 03
dataset/     HEATWATCH.pkl, HEATWATCH.xlsx, participants.csv, trainings.csv,
             questionnaire.csv, 5s/<study_id>.csv               step 04
reports/     HEATWATCH_inclusion_exclusion.xlsx                 step 05
analysis/    HEATWATCH_analysis(_MR / _CPC).xlsx, 9 boxplots, route_maps/   steps 06, 07
selections/  selection_<parameters>.xlsx (+ _5s.csv)            run_pipeline.py --select
weather/     <lat>_<lon>_<YYYY-MM>.csv                          step 04
```

**cache/**: what was read per participant from interim, before anything is combined.
- `<study_id>.pkl`: all tables the reader made (trainings, 5-s streams, daily values, info, file log).
  Step 04 builds the dataset from these files, and the data check uses them. Step 03 skips participants
  that already have a .pkl, which is why reading can be stopped and continued. Delete a .pkl (or use
  `redo` / the study number) to read that participant again.
- `<study_id>.xlsx`: the same reading as a list of files and runs, only for checking by hand. The code
  never reads it.

**dataset/**: the final dataset with only the included participants. The tables are linked by `study_id`
and `run_id` (see the Read me sheet of HEATWATCH.xlsx).
- `HEATWATCH.pkl`: the whole dataset as one Python object. Notebooks 05, 06, 07 and `--select` read
  this file (`dataset.load()`). It is the only file in this folder that the code reads back.
- `HEATWATCH.xlsx`: read me, overview, availability per parameter, participants, trainings and the
  questionnaire, for reading in Excel.
- `participants.csv`, `trainings.csv`, `questionnaire.csv`: the same tables as plain .csv, for R, SPSS
  or other software.
- `5s/<study_id>.csv`: the 5-s data per participant, to open one participant in Excel or use in other
  software. In the code, the 5-s data comes from `HEATWATCH.pkl` (`ds.five_s`).

  For the code, only `HEATWATCH.pkl` is needed. The other files are copies for use outside this code.
  `5s.parquet` (if present) comes from an older version and is not made any more. `data_check.csv` and `excluded.csv` (if present) come
  from an older version and are not made any more: the data check and the exclusions are now in reports/.

**reports/**: `HEATWATCH_inclusion_exclusion.xlsx`, with the flow per population and group, the excluded
participants and the reason, trainings without a location, impossible dates and every participant's
outcome. Use it to check the inclusion and to describe it in the methods.

**analysis/**: the results of notebook 06, plus the route maps of notebook 07.
- `HEATWATCH_analysis.xlsx` (both races), `_MR.xlsx`, `_CPC.xlsx`: group comparison, patients vs healthy
  runners, flow, excluded, per participant, resting HR fallback. The file for both races also has the
  report tables (Flow, Excluded, Characteristics).
- 9 boxplots: `boxplots_direction`, `boxplots_heat` and `boxplots_components`, each for both races, `_MR`
  and `_CPC`. Notebook 06 first removes older boxplots and analysis files (`compare.clear_old()`), so only
  these 9 stay.
- `route_maps/<run_id>.png`: the route of a run coloured by HR and pace (notebook 07).

**selections/**: `selection_<parameters>.xlsx`, the participants that have all the given parameters and
their trainings, with only the matching columns (+ `_5s.csv` with their 5-s data when `hr_5s` / `gps_5s` is asked for).
Made with `run_pipeline.py --select ...` from the saved dataset. Use it to hand a subset to someone else
or for a separate analysis.

**weather/**: hourly Open-Meteo weather, one .csv per grid cell (about 2 km) and month. Step 04 computes
WBGT from these files. They are kept so a rerun does not download again; delete a file to download it
again.

## Use

1. `pip install -r requirements.txt` (Python 3.11+)
2. Copy `config/settings.local.example.toml` to `config/settings.local.toml` and put your paths in it (brand_file, raw_root, data_root = interim, questionnaire_file, output_root)
3. Run the notebooks 01 → 07 in order, or steps 03 to 05 at once with `run_pipeline.py`. The analysis
   (06, 07) is not in the pipeline: run those notebooks afterwards.

How `run_pipeline.py` handles the raw data:

- There are two copies of the watch data: `raw_root` holds the original exports, and `data_root` (interim)
  holds the same exports cut to the data window (`[window] months_before` the first race up to and including the last race). Steps 03–05 read
  only interim.
- Step 02 (inspecting the raw exports) is not in the pipeline. It only shows what is inside the data.
- **Without `--window`** the pipeline does not touch `raw_root`. It reads the interim folder as it is.
  Interim must already be filled, by notebook 03 or an earlier `--window` run.
- **With `--window`** it first cuts the raw exports into interim and then reads them. It cuts the following
  participants: new ones, the study numbers you give, or everyone with `--redo`.
- After you change `months_before`, run `--window --redo`. Without `--window`, the old cut in interim
  stays in use.

```
python run_pipeline.py                        read what is new, build dataset + heat stress, reports
python run_pipeline.py --window               first cut the raw exports of new participants to the window
python run_pipeline.py --window --redo        cut and read everyone again (e.g. after changing the window)
python run_pipeline.py A07 HR136              read these participants again
python run_pipeline.py --brand Fitbit Polar   only read these devices
python run_pipeline.py --redo                 read everyone again
python run_pipeline.py --select hr resting_hr hr_5s
```

Step 02 only shows (nothing saved; the examples are real data, so clear the output before sharing the notebook).
Step 03 skips participants already cut / read, so it can be stopped and continued.
Step 03: `[window] months_before` sets the window; cut a participant again to apply a new window
(study_ids, or redo = True) - the interim folder is then replaced.
Step 04 sends only the grid position and dates to Open-Meteo (no study number or other data);
set `[weather] enabled = false` in the settings to skip heat stress.
