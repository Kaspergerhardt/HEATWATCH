"""Combine what was read for all participants: one run per training (Garmin Connect + .fit
merged; runs that follow each other within merge_gap s merged), race labels, 5-s windows, age / gender.
Input: the cache of step 2."""
import numpy as np
import pandas as pd

from ..loading.participants import RACES
from ..loading.readers.common import SESSION_COLS
from ..settings import PROC


FROM_CONNECT = ["location", "hr_mean", "hr_max", "pace_s_km"]     # per-run values Connect delivers best


# ------------------------------------------------------------------ runs: the same run from two sources, runs split in parts
def merge_duplicate_runs(s):
    """Same run in Garmin Connect (.csv) and .fit (starts within duplicate_tolerance) -> ONE row:
    place name, mean / max HR and mean tempo from Connect; everything else (start, distance, duration,
    start position, link to the 5-s data) from the .fit file; gaps in one filled from the other."""
    s = s.sort_values(["study_id", "start"])
    new_run = s.study_id.ne(s.study_id.shift()) | \
        s.groupby("study_id").start.diff().dt.total_seconds().gt(PROC["duplicate_tolerance"])
    s = s.assign(run_no=new_run.cumsum()).sort_values(["run_no", "has_stream"], ascending=[True, False])
    runs = s.groupby("run_no").first()                    # .fit row first; empty cells filled from Connect
    runs.update(s[s.source == "garmin_connect"].groupby("run_no")[FROM_CONNECT].first())   # Connect wins
    runs["sources"] = s.groupby("run_no").source.agg(lambda x: "+".join(sorted(set(x))))
    return runs.reset_index(drop=True)


def merge_split_runs(s, s5):
    """Runs of one participant that follow each other -> ONE training: the next run starts at most merge_gap s
    (120) after the end (start + duration) of the previous one (a few seconds of overlap counts as 0; larger
    overlaps are not merged). The training keeps start, run_id, position, place and source of the first part;
    duration and distance are summed, mean HR is weighted by duration, max HR is the highest; the 5-s data of
    the parts are joined, seconds counted from the first start. parts = number of runs merged."""
    gap_max, overlap = PROC.get("merge_gap", 120), 60
    s = s.sort_values(["study_id", "start"]).reset_index(drop=True)
    end = s.start + pd.to_timedelta(pd.to_numeric(s.duration_s, errors="coerce").fillna(0), unit="s")
    gap = (s.start - end.groupby(s.study_id).shift()).dt.total_seconds()
    s["first"] = s.session_id.where(~gap.between(-overlap, gap_max)).ffill()
    if s["first"].eq(s.session_id).all():
        return s.drop(columns="first").assign(parts=1), s5
    dur = pd.to_numeric(s.duration_s, errors="coerce")
    hr = pd.to_numeric(s.hr_mean, errors="coerce")
    w = dur.where(hr.notna())
    g = s.assign(hr_x_dur=hr * w, dur_hr=w).groupby("first", sort=False)
    runs = g.first()                                       # first part; empty cells filled from the next parts
    runs["duration_s"] = g.duration_s.sum(min_count=1)
    runs["distance_km"] = g.distance_km.sum(min_count=1)
    runs["hr_mean"] = (g.hr_x_dur.sum(min_count=1) / g.dur_hr.sum(min_count=1)).fillna(runs.hr_mean)
    runs["hr_max"] = g.hr_max.max()
    runs["pace_s_km"] = runs.duration_s / runs.distance_km.where(runs.distance_km > 0)
    for c in ("n_5s_hr", "n_5s_gps"):
        runs[c] = g[c].sum()
    runs["has_stream"] = g.has_stream.any()
    runs["parts"] = g.size()
    runs = runs.drop(columns=["hr_x_dur", "dur_hr"]).reset_index(drop=True)
    # 5-s data: the parts become one run, seconds from the start of the first part
    if len(s5):
        start = s.set_index("session_id").start
        to = s.set_index("session_id")["first"]
        offset = (start - to.map(start)).dt.total_seconds()
        s5 = s5.assign(elapsed_s=s5.elapsed_s + s5.session_id.map(offset).fillna(0),
                       session_id=s5.session_id.map(to).fillna(s5.session_id))
        s5["window"] = (s5.elapsed_s // PROC["stream_seconds"]).astype(int)
    return runs, s5


# ------------------------------------------------------------------ race labels and 5-s windows
def label_races(s, p):
    """race = the race the run prepares for (next race of that participant); session_type 'race' =
    the longest run on that race date; days_to_race = days until that race."""
    races = p.set_index("study_id").races.to_dict()
    s["race"], s["days_to_race"] = None, np.nan
    for sid, idx in s.groupby("study_id").groups.items():
        dates = sorted((RACES[x], x) for x in races.get(sid, "").split(";") if x)
        for i in idx:
            nxt = [(d, n) for d, n in dates if d >= s.at[i, "start"].normalize()] or dates[-1:]
            if nxt:
                s.at[i, "race"] = nxt[0][1]
                s.at[i, "days_to_race"] = (nxt[0][0] - s.at[i, "start"].normalize()).days
    longest = s[s.days_to_race == 0].groupby(["study_id", "race"]).distance_km.idxmax()
    s["session_type"] = np.where(s.index.isin(longest.values), "race", "training")
    return s


def to_5s(streams):
    """Non-overlapping windows of stream_seconds from the run start: mean HR / speed / temperature,
    first position. Apple already delivers 1 HR per 5 s; 1-s devices are averaged."""
    sec = PROC["stream_seconds"]
    num = ["hr", "speed_ms", "temp_c", "lat", "lon", "elapsed_s"]
    streams = streams.assign(**{c: pd.to_numeric(streams[c], errors="coerce") for c in num})
    g = streams.assign(window=(streams.elapsed_s // sec).astype(int)).groupby(["session_id", "window"])
    out = g.agg(t=("t", "first"), hr=("hr", "mean"), speed_ms=("speed_ms", "mean"), temp_c=("temp_c", "mean"),
                lat=("lat", "first"), lon=("lon", "first"), n_samples=("elapsed_s", "size")).reset_index()
    out["elapsed_s"] = out.window * sec
    return out


# ------------------------------------------------------------------ everything together (used by tables.build)
def combine(p, t):
    """The raw data is already limited to the study period (3 months before the race)."""
    s, st, d = t["sessions"], t["streams"], t["daily"]
    s = s.reindex(columns=list(dict.fromkeys(SESSION_COLS + list(s.columns))))      # same columns for every brand

    # run_id = study_id_source_<LOCAL start time> for every source (older .fit / Apple files used UTC)
    new = s.session_id.str.rsplit("_", n=1).str[0] + "_" + s.start.dt.strftime("%Y%m%d%H%M%S")
    st = st.assign(session_id=st.session_id.map(dict(zip(s.session_id, new)))) if len(st) else st
    s = s.assign(session_id=new)

    # 5-s data and per-run summaries of it
    s5 = to_5s(st[st.elapsed_s >= 0]) if len(st) else \
        pd.DataFrame(columns=["session_id", "window", "t", "hr", "speed_ms", "temp_c", "lat", "lon", "n_samples", "elapsed_s"])
    g = s5.groupby("session_id")
    s = s.merge(pd.DataFrame({"n_5s_hr": g.hr.count(), "n_5s_gps": g.lat.count()}),
                left_on="session_id", right_index=True, how="left")
    s[["n_5s_hr", "n_5s_gps"]] = s[["n_5s_hr", "n_5s_gps"]].fillna(0).astype(int)
    s["has_stream"] = s.n_5s_hr > 0                 # HR inside the run every 5 s

    # duplicates, runs that follow each other within merge_gap s, race labels
    s, s5 = merge_split_runs(merge_duplicate_runs(s), s5)
    s = label_races(s, p)
    s5 = s5[s5.session_id.isin(s.session_id)]
    if len(d):                                    # complementary files repeat the same values
        d = d.drop_duplicates(["study_id", "metric", "start", "value"])

    # age / gender: most recent file; flag if files disagree
    info = t["info"]
    if len(info):
        info = info.assign(age=pd.to_numeric(info.age, errors="coerce")).sort_values("date").groupby("study_id").agg(
            age=("age", "last"), gender=("gender", "last"), age_spread=("age", lambda a: a.max() - a.min()),
            n_gender=("gender", "nunique"), info_source=("source", "last"))
        info["info_flag"] = np.where((info.age_spread > 1) | (info.n_gender > 1), "files disagree", None)
        p = p.join(info[["age", "gender", "info_source", "info_flag"]], on="study_id")
    else:
        p[["age", "gender", "info_source", "info_flag"]] = None
    return p, s, s5, d
