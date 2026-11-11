"""Steps 01-03 - loading: who is in the study and what is in their exports.

  participants.py   brand overview -> group, race(s), device; data folder per study number   (01)
  inspect_raw.py    per raw folder: file groups, columns / fields with 3 examples (shown only)  (02)
  window.py         raw exports cut to [window] months_before the race -> interim (data_root) (03)
  extract.py        per participant all files of the device (interim) -> cache/<study_id>.pkl (03)
  questionnaire.py  Castor questionnaires (CRF + HEAT-WATCH questionnaire) -> 1 row per participant per race
  readers/          one reader per brand
"""
