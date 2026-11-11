"""HEATWATCH: smartwatch exports -> one dataset (participants, trainings, 5-s data) + heat stress + reports.

  settings.py    config/settings.toml + settings.local.toml, output folders
  excel.py       styled Excel output (used everywhere)
  loading/       01-03  participants, inspect the raw exports, cut them to the window -> interim, read -> cache
  dataset/       04     combine -> participants / trainings / 5 s, location
  heat_stress/   04     WBGT (Open-Meteo) and heat dose per training
  reports/       04-05  data check, inclusion / exclusion, saving
  analysis/      06-07  heat exposure index and training timing per group, boxplots, route maps

One notebook per step in notebooks/, or steps 01-05 at once with run_pipeline.py (analysis 06-07 only in the notebooks).
"""
from .settings import CFG, PROC

__all__ = ["CFG", "PROC"]
