"""Step 04 - heat stress: WBGT and heat dose per training.

  weather.py   Open-Meteo weather at start and during each training -> WBGT columns
  wbgt.py      Liljegren WBGT (as KNMI), indoor (gym) estimate

Use:  ds = heat_stress.add(ds)
"""
from .weather import add, add_wbgt

__all__ = ["add", "add_wbgt"]
