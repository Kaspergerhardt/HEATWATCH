"""Settings: config/settings.toml (study, shared) + config/settings.local.toml (your paths, overrides).

Output folders (all inside paths.output_root):
    cache/        step 03: what was read per participant from interim (<study_id>.pkl + <study_id>.xlsx to check)
    dataset/      participants.csv, trainings.csv, 5s/<study_id>.csv, HEATWATCH.xlsx (+ .pkl)
    reports/      HEATWATCH_inclusion_exclusion.xlsx
    selections/   analysis datasets made with select()
    weather/      Open-Meteo downloads (one file per grid cell and month)
    analysis/     heat exposure and group comparison: tables (.xlsx) and boxplots (.png)
"""
import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                    # project folder (contains config/)
CONFIG = Path(os.environ.get("HEATWATCH_CONFIG_DIR", ROOT / "config"))


# ------------------------------------------------------------------ read the settings
def _merge(base, extra):
    """Put the values of `extra` into `base` (sections merged, not replaced)."""
    for k, v in extra.items():
        base[k] = _merge(base.get(k, {}), v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return base


def load():
    """settings.toml + settings.local.toml (your paths) -> one dict."""
    with open(CONFIG / "settings.toml", "rb") as f:
        cfg = tomllib.load(f)
    local = CONFIG / "settings.local.toml"
    if local.exists():
        with open(local, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    if "paths" not in cfg:
        raise SystemExit(f"No [paths] found: copy {CONFIG / 'settings.local.example.toml'} to {local} "
                         "and fill in your own paths")
    return cfg


CFG = load()
PROC = CFG["processing"]
DATA_ROOT = Path(CFG["paths"]["data_root"])
OUT = Path(CFG["paths"]["output_root"])
CACHE, DATASET, REPORTS, SELECTIONS, WEATHER, ANALYSIS = (
    OUT / d for d in ("cache", "dataset", "reports", "selections", "weather", "analysis"))
RAW_ROOT = Path(CFG["paths"]["raw_root"]) if "raw_root" in CFG["paths"] else None   # raw exports (step 02, 03)
