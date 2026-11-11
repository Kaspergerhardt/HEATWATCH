"""Read the exports: per participant all files of the device -> cache/<study_id>.pkl."""
import time

import pandas as pd

from .. import excel
from .participants import find_folders, key
from .readers import READERS
from ..settings import CACHE

TABLES = ("sessions", "streams", "daily", "info", "log")


# ------------------------------------------------------------------ read one participant
def read_one(study_id, brand, folders):
    """Run the brand's reader(s) on every folder of one participant -> dict of tables."""
    out = {k: [] for k in TABLES}
    for folder, group in folders:
        for read in READERS.get(brand, []):
            for k, df in read(study_id, folder).items():
                if len(df):
                    out[k].append(df if k == "streams" else df.assign(output_group=group))
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}


def save_one(study_id, brand, folders):
    """Read one participant and keep the result in the cache (+ an Excel file to check what was read)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    error = CACHE / f"{study_id}.error.txt"
    try:
        t = read_one(study_id, brand, folders)
    except Exception as e:
        error.write_text(f"{type(e).__name__}: {e}")
        raise
    error.unlink(missing_ok=True)
    pd.to_pickle(t, CACHE / f"{study_id}.pkl")
    excel.write(CACHE / f"{study_id}.xlsx", None, [("files", [(None, None, t["log"])]),
                                                    ("runs", [(None, None, t["sessions"])])])
    return t


# ------------------------------------------------------------------ read all participants / load what was read
def read_participants(p, study_ids=None, brands=None, redo=False):
    """Read the exports of the participants in p (from participants.load()).
    study_ids: only these | brands: only these devices | redo: read everyone again.
    Returns the folders found (needed for the data check)."""
    if brands:
        p = p[p.brand.map(key).isin(map(key, brands))]
    folders = find_folders(p.study_id)
    todo = p[p.study_id.str.upper().isin(folders) & p.brand.isin(READERS)]
    if study_ids:
        todo = todo[todo.study_id.str.upper().isin([x.upper() for x in study_ids])]
    elif not redo:
        todo = todo[~todo.study_id.map(lambda x: (CACHE / f"{x}.pkl").exists())]
    print(f"{len(p)} participants, data folder found for {len(folders)}, to read now: {len(todo)}")
    try:
        for i, r in enumerate(todo.itertuples(), 1):
            start = time.time()
            try:
                t = save_one(r.study_id, r.brand, folders[r.study_id.upper()])
                msg = f"{len(t['log'])} files, {len(t['sessions'])} trainings"
            except Exception as e:                       # one odd file must not stop the run
                msg = f"SKIPPED ({type(e).__name__}: {e})"
            print(f"\r[{i}/{len(todo)}] {r.study_id} ({r.brand}): {msg}  [{time.time() - start:.0f} s]" + " " * 10)
    except KeyboardInterrupt:
        print("stopped - what is read so far is kept")
    return folders


def load_cache():
    """All participants read so far -> one dict of tables."""
    parsed = [pd.read_pickle(f) for f in sorted(CACHE.glob("*.pkl"))]
    for t in parsed:                                 # read before 'sex' was renamed 'gender'
        t["info"] = t["info"].rename(columns={"sex": "gender"})
    if not parsed:
        raise SystemExit("Nothing read yet: run step 2 (read the exports) first.")
    return {k: pd.concat([t[k] for t in parsed if len(t[k])], ignore_index=True)
            if any(len(t[k]) for t in parsed) else pd.DataFrame() for k in TABLES}
