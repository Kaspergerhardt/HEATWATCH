"""Step 04 - the dataset: three linked tables (participants, trainings, 5 s).

  combine.py    Garmin Connect + .fit merged, race labelled, 5-s windows, age / gender
  tables.py     the three tables, weekly / daily values, select()
  location.py   GPS -> municipality (offline), place names in titles

Use:  ds = dataset.build()   ds = dataset.load()   dataset.select(...)
"""
from .tables import PARAMETERS, Dataset, build, has, load, select

__all__ = ["PARAMETERS", "Dataset", "build", "has", "load", "select"]
