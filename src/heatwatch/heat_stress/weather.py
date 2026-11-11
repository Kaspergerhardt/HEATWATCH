"""Heat stress. Weather at the start of each training from Open-Meteo (https://open-meteo.com) -> WBGT and heat dose.

Per training: start position and start time (UTC); only the position and dates are sent (no study number).
Starts within the same grid cell ([weather] grid_deg, default 0.02 deg = about 2 km, close to the
resolution of the weather model) share one download; one grid cell for one month is downloaded once and
kept in <output_root>/weather/, so a rerun does not download again.
Indoor runs (treadmill / indoor): estimated WBGT of a commercial gym, no sun, from [weather]
indoor_temp_c (19.5), indoor_rh_pct (40) and indoor_air_speed (0.2 m/s) -> 14.8 degC.

Hourly variables: temperature_2m (degC), relative_humidity_2m (%), surface_pressure (hPa),
wind_speed_10m (m/s), shortwave_radiation (W/m2, mean of the PRECEDING hour, so it is placed at the
middle of that hour). Values are interpolated linearly to the start time of the training.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..settings import CFG, WEATHER
from ..analysis import heat_exposure as he
from .wbgt import indoor_wbgt, wbgt

W = CFG.get("weather", {})
URL = W.get("url", "https://historical-forecast-api.open-meteo.com/v1/forecast")
GRID = float(W.get("grid_deg", 0.02))
INDOOR = indoor_wbgt(float(W.get("indoor_temp_c", 19.5)), float(W.get("indoor_rh_pct", 40)),
                     float(W.get("indoor_air_speed", 0.2)))
VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "shortwave_radiation"]
BATCH = 50                                   # grid cells per request


# ------------------------------------------------------------------ weather: grid cells and Open-Meteo downloads
def cell(v):
    """Coordinate rounded to the weather grid (grid_deg); None if missing."""
    v = pd.to_numeric(v, errors="coerce")
    return round(round(v / GRID) * GRID, 4) if v == v else None


def download(cells, month, cache):
    """Hourly weather for these cells in this month (YYYY-MM) -> cache files."""
    start = pd.Timestamp(month + "-01")
    end = min(start + pd.offsets.MonthEnd(0), pd.Timestamp.today().normalize() - pd.Timedelta(days=1))
    for i in range(0, len(cells), BATCH):
        part = cells[i:i + BATCH]
        q = {"latitude": ",".join(str(a) for a, _ in part), "longitude": ",".join(str(b) for _, b in part),
             "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}", "hourly": ",".join(VARS),
             "wind_speed_unit": "ms", "timezone": "GMT"}
        with urllib.request.urlopen(URL + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            data = json.load(r)
        for (a, b), d in zip(part, data if isinstance(data, list) else [data]):
            pd.DataFrame(d["hourly"]).to_csv(cache / f"{a}_{b}_{month}.csv", index=False)
        time.sleep(1)                                    # stay well below the free-use rate limit


# ------------------------------------------------------------------ WBGT per training: moments, cached weather, WBGT columns
STEP = pd.Timedelta(minutes=10)                  # WBGT during the training: every 10 min + the end


def times_during(start, duration_s):
    """Start, every 10 min after it, and the end of the training (only the start if the duration is unknown)."""
    if duration_s != duration_s or duration_s is None or duration_s <= 0:
        return [start]
    end = start + pd.Timedelta(seconds=float(duration_s))
    ts = list(pd.date_range(start, end, freq=STEP))
    return ts + [end] if ts[-1] < end else ts


def download_missing(cells, moments, cache):
    """Download the grid cell-months that are not in the cache yet (no internet: those trainings stay empty)."""
    need = {}
    for i, (a, b) in cells.items():
        for m in {x.strftime("%Y-%m") for x in moments[i]}:
            if not (cache / f"{a}_{b}_{m}.csv").exists() and (a, b) not in need.get(m, []):
                need.setdefault(m, []).append((a, b))
    if need:
        n = sum(len(v) for v in need.values())
        print(f"\nOpen-Meteo: downloading weather for {n} grid cell-months ...")
        for m, part in sorted(need.items()):
            try:
                download(part, m, cache)
            except Exception as e:                       # no internet / API down: those trainings stay empty
                print(f"  ! weather for {m} not downloaded ({type(e).__name__}: {e}); rerun later to fill it")


def hourly(tables, f):
    """Cached weather file -> {variable: (times, values)} for interpolation; read once (kept in `tables`)."""
    if f not in tables:
        tables[f] = None
        if f.exists():
            h = pd.read_csv(f, parse_dates=["time"])
            x = h.time.values.astype("datetime64[ns]").astype("int64")
            half = pd.Timedelta(minutes=30).value          # radiation = mean of the preceding hour
            tables[f] = {c: (x - half if c == "shortwave_radiation" else x, h[c].to_numpy(float)) for c in VARS}
    return tables[f]


def add_wbgt(t, lat, lon, start_utc, indoor, duration_s=None, cache=WEATHER):
    """Columns for the trainings table t:
      wbgt_mean, wbgt_max   WBGT during the training: every 10 min from start to finish (+ the finish),
                            weather of the START position (the route itself is not followed)
    Indoor runs get the estimated WBGT of a gym (INDOOR) in all columns; no position -> empty."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    dur = duration_s if duration_s is not None else pd.Series(np.nan, index=t.index)
    ok = lat.notna() & lon.notna() & start_utc.notna() & ~indoor
    cells = {i: (cell(lat.loc[i]), cell(lon.loc[i])) for i in t.index[ok.values]}
    moments = {i: times_during(start_utc.loc[i], dur.loc[i]) for i in cells}

    download_missing(cells, moments, cache)
    tables = {}

    def at(i, ts):
        h = hourly(tables, cache / f"{cells[i][0]}_{cells[i][1]}_{ts:%Y-%m}.csv")
        if h is None:
            return np.nan
        v = {c: float(np.interp(ts.value, *h[c])) for c in VARS}
        return wbgt(v["temperature_2m"], v["relative_humidity_2m"], v["surface_pressure"], v["wind_speed_10m"],
                    v["shortwave_radiation"], lat.loc[i], lon.loc[i], ts.floor("s").to_pydatetime())

    mean, high = (np.full(len(t), np.nan) for _ in range(2))
    pos = {i: k for k, i in enumerate(t.index)}
    for n, i in enumerate(cells, 1):
        if n % 500 == 0:
            print(f"\r    WBGT: training {n}/{len(cells)}", end="", flush=True)
        w = np.array([at(i, ts) for ts in moments[i]], dtype=float)
        if np.isfinite(w).any():
            mean[pos[i]], high[pos[i]] = np.nanmean(w), np.nanmax(w)
    for arr in (mean, high):
        arr[indoor.values] = INDOOR
    t["wbgt_mean"] = np.round(mean, 1)
    t["wbgt_max"] = np.round(high, 1)
    return t


# ------------------------------------------------------------------ dataset step: WBGT and heat dose columns
def add(ds):
    """Add wbgt_mean, wbgt_max, heat_dose and heat_dose_hrmax_training to ds.trainings and recount the participant
    table. heat_dose (HD) = duration (min) x effort x max(0, WBGT_mean - wbgt_threshold), as in the analysis (heat_exposure.doses),
    with HR max = hr_max_observed (incl. race); heat_dose_hrmax_training: the same with hr_max_training (race left
    out). Empty when mean HR, WBGT or resting HR is missing."""
    if not CFG.get("weather", {}).get("enabled", True):
        print("weather.enabled = false in the settings: heat stress skipped")
        return ds
    same = len(ds.where) == len(ds.trainings) and (ds.where.run_id.values == ds.trainings.run_id.values).all()
    w = ds.where if same else ds.where.drop_duplicates("run_id").set_index("run_id").reindex(ds.trainings.run_id).reset_index()
    ds.trainings = add_wbgt(ds.trainings, w.lat, w.lon, w.start_utc, w.indoor.fillna(False).astype(bool),
                            w.duration_s)
    ds = ds.refresh()                                  # hr_max_observed per participant, needed for the dose
    for i, hr_max in enumerate(he.HR_MAX):
        t, _ = he.doses(ds.trainings, ds.trainings, ds.participants, hr_max)
        name = "heat_dose" if hr_max == "hr_max_observed" else "heat_dose_hrmax_training"
        ds.trainings.insert(ds.trainings.columns.get_loc("wbgt_max") + 1 + i, name, t.heat_dose.round(1))
    return ds
