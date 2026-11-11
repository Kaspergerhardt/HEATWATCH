"""Group comparison of patients, healthy runners and pacers: outcomes per participant, statistics, boxplots.

  run(ds, s, race)     race None: both races (one per participant: s.race, else the one with data); 'MR' / 'CPC': per race
  outcomes()           training timing, HEI (+ exposed yes / no), pace and the three factors of HEI (OUTCOMES)
  exclusions()         step 05 + no training in the window, missing parameters (s.params), first that applies
  compare()            median (IQR) per group, Kruskal-Wallis (groups >= s.min_group_n), pairwise Mann-Whitney
                       (Bonferroni), exposed yes / no with Fisher's exact test
  two_groups()         patients vs healthy runners on every outcome (Mann-Whitney r, Fisher odds ratio)
  figures()            boxplots per group (FIGURES)
  clear_old()          remove boxplots / analysis files in analysis/ that run() no longer makes
Settings: set in notebook 06.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..dataset import has
from ..excel import write
from ..reports import overview
from ..reports.overview import med_iqr
from ..settings import ANALYSIS
from . import heat_exposure as he

PER_TRAINING = {"hr": "hr_mean", "wbgt": "wbgt_mean", "location": "location"}   # parameter: column in a training
NAME = {"hr": "HR", "wbgt": "WBGT", "resting_hr": "resting HR", "hrv": "HRV", "hr_5s": "5-s HR", "gps_5s": "5-s GPS"}
GROUPS = ["patients", "healthy runners", "pacers"]
REGISTERED = {"MR": {"MRHR", "RMP", "RM"}, "CPC": {"CPCHR", "CPCP", "CPC"}}   # brand-overview groups per race
RACE_NAME = {"MR": "Marathon Rotterdam", "CPC": "CPC", None: "both races"}
LABEL = {"patients": "Patients", "healthy runners": "Healthy runners", "pacers": "Pacers"}
COLOR = {"patients": "#E4572E", "healthy runners": "#2E86AB", "pacers": "#3BB143"}
EXPOSED = "Exposed (HEI > 0)"
FIGURES = {   # name: (title, panels); a panel is a column of OUTCOMES, or EXPOSED (% of participants, bars)
    "direction": ("Training timing: earlier or later than the race", ["median_shift_h", "earlier_1h_pct", "later_1h_pct"]),
    "heat": ("Heat exposure", [EXPOSED, "HEI_exposed"]),
    "components": ("The three factors of the heat exposure index", ["training_hours_14d", "effort_mean", "wbgt_mean_14d"]),
}
PACE_RANGE = (2.5, 15)   # min/km; outside = distance or duration not plausible, training not used for pace
OUTCOMES = {   # column: (title in table / plot, y-axis label, y-scale); {h} = within_hours
    "median_diff_h": ("Training vs race start time", "median |difference| (h)", "linear"),
    "within_1h_pct": ("Trainings within {h} h of race time", "% of trainings", "linear"),
    "median_shift_h": ("Training start relative to race start", "median difference (h), < 0 = earlier", "linear"),
    "earlier_1h_pct": ("Trainings > {h} h earlier than race time", "% of trainings", "linear"),
    "later_1h_pct": ("Trainings > {h} h later than race time", "% of trainings", "linear"),
    "trainings_heat": ("Trainings in the heat (heat dose > 0)", "number of trainings", "linear"),
    "HEI": ("Heat exposure index, all (0 = not exposed)", None, None),
    "HEI_exposed": ("Heat exposure index, exposed only", "HEI (log scale)", "log"),
    "pace_mean": ("Pace of the trainings", "mean pace (min/km)", "linear"),
    "training_hours_14d": ("Training duration in last 14 days", "total duration (h)", "linear"),
    "effort_mean": ("Effort of the trainings", "effort (Banister TRIMP / min)", "linear"),
    "wbgt_mean_14d": ("WBGT during the trainings", "mean WBGT (°C)", "linear"),
}


@dataclass
class Settings:
    """The analysis settings (notebook 06, Settings)."""
    params: list            # parameters a participant needs (names as in dataset.PARAMETERS); hr / wbgt / location:
                            # in the same training, in the window_days before the race; others: any value in the data
    within_hours: float     # training timing: hours around the race start time
    min_group_n: int        # groups with fewer values are described but not tested
    window_days: int        # acclimatization window: only trainings in this many days before the race
    decay_per_day: float    # HEI: loss of heat adaptation per day
    race: str               # race used when a participant ran both (otherwise the race they ran)


# ------------------------------------------------------------------ outcomes per participant: timing, HEI, pace and the factors of HEI
def names(within):
    """OUTCOMES with the number of hours filled in."""
    return {col: (title.format(h=f"{within:g}"), ylab, scale) for col, (title, ylab, scale) in OUTCOMES.items()}


def hours(hhmm):
    """'08:30' -> 8.5 (hours since midnight)."""
    t = pd.to_datetime(hhmm, format="%H:%M", errors="coerce")
    return t.dt.hour + t.dt.minute / 60


def weighted(t, col):
    """Per participant: mean of `col` over the trainings, weighted by duration."""
    t = t[t[col].notna() & t.duration_min.notna()]
    return (t[col] * t.duration_min).groupby(t.study_id).sum() / t.duration_min.groupby(t.study_id).sum()


def outcomes(ds, races, s):
    """1 row per participant with a race (races: races_used()) and >= 1 training before it (s: Settings)."""
    within = s.within_hours
    t = he.before_race(ds.trainings, races, s.window_days)
    t["shift_h"] = hours(t.start_time) - hours(t.race_start)          # < 0: earlier in the day than the race
    t["diff_h"] = t.shift_h.abs()
    g = t.groupby("study_id")
    out = races[["population", "race_used", "race_date", "race_start"]].join(pd.DataFrame({
        "trainings": g.size(),
        "median_diff_h": g.diff_h.median().round(2),
        "within_1h_pct": g.diff_h.apply(lambda x: 100 * (x <= within).mean()).round(1),
        "median_shift_h": g.shift_h.median().round(2),
        "earlier_1h_pct": g.shift_h.apply(lambda x: 100 * (x < -within).mean()).round(1),
        "later_1h_pct": g.shift_h.apply(lambda x: 100 * (x > within).mean()).round(1)}), how="inner")
    out = out.join(he.hei(ds.trainings, ds.participants, races, s.window_days, s.decay_per_day), how="left")
    hd = he.heat_doses(ds.trainings, ds.participants, races, s.window_days, s.decay_per_day)
    out["trainings_heat"] = hd[hd.heat_dose > 0].groupby("study_id").size().reindex(out.index).fillna(0).astype(int)
    out["HEI_exposed"] = out.HEI.where(out.HEI > 0)                     # part 2: exposed participants only
    # pace, and the three factors of HEI: duration (all trainings), effort (trainings with a heat dose), WBGT
    w = t[t.wbgt_mean.notna()]
    p = t[(t.duration_min / t.distance_km).between(*PACE_RANGE)].groupby("study_id")
    out = out.join(pd.DataFrame({"training_hours_14d": (g.duration_min.sum() / 60).round(2),
                                 "pace_mean": (p.duration_min.sum() / p.distance_km.sum()).round(2),
                                 "effort_mean": weighted(hd, "effort").round(3),
                                 "wbgt_mean_14d": weighted(w, "wbgt_mean").round(1)}), how="left")
    out = out.join(ds.participants.set_index("study_id")[["gender", "age", "brand"]], how="left")
    return out.reset_index()


def only_race(days):
    """Exclusion reason: no training in the window before the race."""
    return f"No training in {days} days before race"


# ------------------------------------------------------------------ who is in the analysis: race used and exclusions
def missing(ds, races, s):
    """Exclusion reason -> study numbers without the parameters, in this order: each participant-level parameter
    (no value anywhere in the data), then the per-training ones together (no training in the HEI window before
    the race used with all of them)."""
    pt = ds.participants
    out = {f"No {NAME.get(k, k)}": set(pt.study_id[~has(pt, k)])
           for k in s.params if k not in PER_TRAINING and k != "training"}
    per_training = [k for k in s.params if k in PER_TRAINING]
    if per_training:
        t = he.before_race(ds.trainings, races, s.window_days)
        ok = t[t[[PER_TRAINING[k] for k in per_training]].notna().all(axis=1)].study_id
        names = " and ".join(NAME.get(k, k) for k in per_training)
        out[f"No training with {names} in {s.window_days} days before race"] = set(pt.study_id) - set(ok)
    return out


def eligible(ds, races, s):
    """Study numbers with >= 1 training before their race used and all parameters."""
    return set(he.before_race(ds.trainings, races, s.window_days).study_id) - set().union(*missing(ds, races, s).values())


def races_used(ds, s, race=None):
    """The race used per participant. race 'MR' / 'CPC': that race. None: s.race when both were run, but the
    other race when only that one has the data for the analysis (otherwise the participant would be excluded)."""
    races = he.chosen_races(ds.trainings, s.race, race)
    if race:
        return races
    alt = he.chosen_races(ds.trainings[ds.trainings.race != s.race], s.race)          # the other race
    both = alt.index[alt.index.isin(races.index)]
    both = both[races.race_used.reindex(both).values != alt.race_used.reindex(both).values]
    ok, ok_alt = eligible(ds, races, s), eligible(ds, alt, s)
    swap = [s for s in both if s not in ok and s in ok_alt]
    races.loc[swap] = alt.loc[swap]
    return races


def exclusions(ds, df, races, s, race=None):
    """Step 05 outcomes (ds.outcomes; per race only those registered for it, 'No data of the race' when it is not
    in their data) + only_race() for included participants not in df (outcomes()) + the missing parameters.
    Returns (all outcomes, flow per population, excluded participants)."""
    if ds.outcomes is None:
        raise SystemExit("No outcomes in the dataset: run notebook 04_build_dataset again.")
    o = ds.outcomes.copy()
    if race:
        o = o[o.group.str.split(" & ").map(lambda g: bool(REGISTERED[race] & set(g)))]
        o.loc[(o.outcome == "Included") & ~o.study_id.isin(races.index), "outcome"] = "No data of the race"
    o.loc[(o.outcome == "Included") & ~o.study_id.isin(df.study_id), "outcome"] = only_race(s.window_days)
    steps = [only_race(s.window_days)]
    for reason, ids in missing(ds, races, s).items():
        o.loc[(o.outcome == "Included") & o.study_id.isin(ids), "outcome"] = reason
        steps.append(reason)
    o = pd.concat([o[~o.outcome.isin(steps)]] + [o[o.outcome == s] for s in steps])   # flow in this order
    pop = overview.counts(o)[1]
    pop.iloc[-1, 0] = "Included in analysis"
    return o, pop, overview.excluded(o)


# ------------------------------------------------------------------ statistics: group comparison, patients vs healthy runners
def kruskal(df, col, title, min_n):
    """Kruskal-Wallis over the groups with >= min_n values; pairwise Mann-Whitney (Bonferroni) if p < 0.05."""
    data = {g: df.loc[df.population == g, col].dropna() for g in GROUPS}
    data = {g: v for g, v in data.items() if len(v) >= min_n}
    if len(data) < 2:
        return np.nan, []
    p = stats.kruskal(*data.values()).pvalue
    pairs = []
    if p < 0.05:
        names = list(data)
        combos = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        for a, b in combos:
            pu = stats.mannwhitneyu(data[a], data[b]).pvalue
            pairs.append({"outcome": title, "comparison": f"{LABEL[a]} vs {LABEL[b]}",
                          "p (Bonferroni)": round(min(1.0, pu * len(combos)), 4)})
    return p, pairs


def fmt_p(p):
    """p-value as text: '-', '<0.001' or '0.123'."""
    return "-" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


@dataclass
class Result:
    """Everything one analysis (run) returns."""
    per_participant: pd.DataFrame
    table: pd.DataFrame          # 1 row per outcome: median (IQR) per group, p, significant
    pairwise: pd.DataFrame
    exposed: pd.DataFrame        # exposed yes / no per group (Fisher)
    participants: pd.DataFrame   # description per group
    figures: dict                # 'direction' | 'heat' | 'components' -> boxplot figure
    flow: pd.DataFrame          # registered -> exclusions -> included in analysis, per population
    excluded: pd.DataFrame       # excluded participants + reason
    fallback: pd.DataFrame       # resting HR fallback per participant (trainings with the median resting HR)
    fallback_trainings: pd.DataFrame   # the trainings that used the fallback
    patients_vs_healthy: pd.DataFrame = None   # every outcome: patients vs healthy runners only
    race: str = None             # 'MR' / 'CPC', None = both races together


def two_groups(df, within, a="patients", b="healthy runners"):
    """Planned comparison of two groups on every outcome, regardless of the overall test: Mann-Whitney U
    (rank-biserial r: > 0 = higher in `a`, -1 to 1) and for exposed yes / no Fisher's exact test (odds ratio)."""
    rows = []
    for col, (title, *_) in names(within).items():
        x, y = (df.loc[df.population == g, col].dropna() for g in (a, b))
        p, r = np.nan, np.nan
        if len(x) and len(y):
            u = stats.mannwhitneyu(x, y)
            p, r = u.pvalue, 2 * u.statistic / (len(x) * len(y)) - 1
        rows.append({"outcome": title, LABEL[a]: f"{med_iqr(x)} (n={len(x)})", LABEL[b]: f"{med_iqr(y)} (n={len(y)})",
                     "test": "Mann-Whitney U", "p": fmt_p(p), "effect size": f"r = {r:.2f}" if r == r else "-",
                     "significant": "yes" if p < 0.05 else "no"})
        if col == "HEI":                                                   # part 1: exposed yes / no
            e = df[df.HEI.notna() & df.population.isin([a, b])]
            tab = pd.crosstab(e.population, e.exposed).reindex(index=[a, b], columns=[True, False]).fillna(0).astype(int)
            res = stats.fisher_exact(tab.to_numpy()) if len(e) else None
            p = float(res.pvalue) if res else np.nan
            rows.append({"outcome": EXPOSED,
                         **{LABEL[g]: f"{tab.at[g, True]} of {tab.loc[g].sum()} ({100 * tab.at[g, True] / max(1, tab.loc[g].sum()):.0f}%)"
                            for g in (a, b)},
                         "test": "Fisher's exact", "p": fmt_p(p),
                         "effect size": f"OR = {res.statistic:.2f}" if res and np.isfinite(res.statistic) else "-",
                         "significant": "yes" if p < 0.05 else "no"})
    return pd.DataFrame(rows)


def compare(df, within, min_n):
    """Per outcome median (IQR) per group + Kruskal-Wallis, exposed yes / no (Fisher), description per group."""
    rows, pairs = [], []
    for col, (title, *_) in names(within).items():
        p, pw = kruskal(df, col, title, min_n)
        pairs += pw
        rows.append({"outcome": title,
                     **{LABEL[g]: f"{med_iqr(df.loc[df.population == g, col])} (n={df.loc[df.population == g, col].notna().sum()})"
                        for g in GROUPS},
                     "p (Kruskal-Wallis)": fmt_p(p), "significant": "yes" if p < 0.05 else "no"})
    e = df[df.HEI.notna()]
    tab = pd.crosstab(e.exposed, e.population).reindex(index=[True, False], columns=GROUPS).fillna(0).astype(int)
    used = tab.loc[:, tab.sum(axis=0) > 0]                         # groups without anyone with an HEI left out
    used = used.loc[used.sum(axis=1) > 0]
    p_exp = float(np.asarray(stats.fisher_exact(used.to_numpy()).pvalue)) if used.shape == (2, used.shape[1]) \
        and used.shape[1] >= 2 else np.nan
    exposed = pd.DataFrame({LABEL[g]: {"exposed (HEI > 0)": f"{tab.at[True, g]} ({100 * tab.at[True, g] / max(1, tab[g].sum()):.0f}%)",
                                       "not exposed": int(tab.at[False, g]), "n with HEI": int(tab[g].sum()),
                                       "no HEI (no training with HR, WBGT and resting HR)":
                                           int(((df.population == g) & df.HEI.isna()).sum())} for g in GROUPS})
    exposed["p (Fisher)"] = ["", "", "", ""]
    exposed.iloc[0, -1] = fmt_p(p_exp)
    rows.insert(list(OUTCOMES).index("HEI") + 1,                      # part 1, between all and exposed only
                {"outcome": EXPOSED, **{LABEL[g]: exposed.at["exposed (HEI > 0)", LABEL[g]] for g in GROUPS},
                 "p (Kruskal-Wallis)": f"{fmt_p(p_exp)} (Fisher)", "significant": "yes" if p_exp < 0.05 else "no"})
    desc = pd.DataFrame({LABEL[g]: {
        "n": int((df.population == g).sum()),
        "male / female / unknown": f"{(d.gender == 'male').sum()} / {(d.gender == 'female').sum()} / {d.gender.isna().sum()}",
        "age, median (IQR)": med_iqr(d.age),
        "race MR / CPC": f"{(d.race_used == 'MR').sum()} / {(d.race_used == 'CPC').sum()}",
        "trainings, median (IQR)": med_iqr(d.trainings)} for g in GROUPS for d in [df[df.population == g]]})
    return pd.DataFrame(rows), pd.DataFrame(pairs), exposed, desc


# ------------------------------------------------------------------ figures: boxplots per group
def exposed_bars(ax, df):
    """Part 1 of HEI: % of participants exposed (HEI > 0) per group."""
    n = [int(df.loc[df.population == g, "HEI"].notna().sum()) for g in GROUPS]
    pct = [100 * df.loc[df.population == g, "exposed"].eq(True).sum() / max(1, k) for g, k in zip(GROUPS, n)]
    ax.bar(range(1, len(GROUPS) + 1), pct, width=0.6, color=[COLOR[g] for g in GROUPS], alpha=0.85)
    for i, v in enumerate(pct, 1):
        ax.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(1, len(GROUPS) + 1))
    ax.set_xticklabels([f"{LABEL[g]}\n(n={k})" for g, k in zip(GROUPS, n)], fontsize=9)
    ax.set_title(EXPOSED, fontsize=10, fontweight="bold")
    ax.set_ylabel("% of participants")
    ax.grid(axis="y", alpha=0.3)


def boxplots(df, panels, title, within):
    """One panel per outcome (FIGURES): box per group with the individual participants as points;
    EXPOSED: % of participants with HEI > 0 (bars)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.8))
    rng = np.random.default_rng(1)
    for ax, col in zip(np.atleast_1d(axes), panels):
        if col == EXPOSED:
            exposed_bars(ax, df)
            continue
        name, ylab, scale = names(within)[col]
        vals = [df.loc[df.population == g, col].dropna().to_numpy() for g in GROUPS]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops={"color": "black"})
        for patch, g in zip(bp["boxes"], GROUPS):
            patch.set_facecolor(COLOR[g])
            patch.set_alpha(0.85)
        for i, v in enumerate(vals, 1):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color="black", alpha=0.5, zorder=3)
        ax.set_xticks(range(1, len(GROUPS) + 1))
        ax.set_xticklabels([f"{LABEL[g]}\n(n={len(v)})" for g, v in zip(GROUPS, vals)], fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)
        ax.set_yscale(scale)                 # HEI: log, a few very high values would flatten the boxes
        if col == "median_shift_h":          # 0 = same time of day as the race
            ax.axhline(0, ls="--", color="grey", lw=1)
        if col == "wbgt_mean_14d":           # heat counts only above the threshold
            ax.axhline(he.THRESHOLD, ls="--", color="grey", lw=1)
            ax.text(0.55, he.THRESHOLD, f" threshold {he.THRESHOLD:g} °C", va="bottom", fontsize=8, color="grey")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.close(fig)                       # shown only when asked for (res.figures[...]), not on every call
    return fig


def figures(df, within, race=None, note=None):
    """The boxplot figures: timing direction, heat exposure, the factors of HEI. Title: + the race for a per-race analysis, + note (e.g. HR max without race)."""
    end = "".join(f": {x}" for x in (RACE_NAME[race] if race else None, note) if x)
    return {name: boxplots(df, panels, title + end, within) for name, (title, panels) in FIGURES.items()}


# ------------------------------------------------------------------ run one analysis and save it
def run(ds, s, race=None, save=True):
    """s: Settings. race: 'MR' / 'CPC' for the analysis of one race, None for both races together.
    HR max for the heat dose: hr_max_observed (highest HR of any training, race included)."""
    races = races_used(ds, s, race)
    df = outcomes(ds, races, s)
    o, flow, excluded = exclusions(ds, df, races, s, race)
    df = df[df.study_id.isin(o.study_id[o.outcome == "Included"])].reset_index(drop=True)
    table, pairwise, exposed, desc = compare(df, s.within_hours, s.min_group_n)
    figs = figures(df, s.within_hours, race)
    hd = he.heat_doses(ds.trainings, ds.participants, races, s.window_days, s.decay_per_day)
    fb, fb_t = he.fallback(hd[hd.study_id.isin(df.study_id)])
    two = two_groups(df, s.within_hours)
    res = Result(df, table, pairwise, exposed, desc, figs, flow, excluded, fb, fb_t, two, race)
    if save:
        save_result(res, s.params)
    return res


def save_result(res, params):
    """Boxplots (.png) and HEATWATCH_analysis(_race).xlsx in analysis/."""
    race = res.race
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    end = f"_{race}" if race else ""
    for name, fig in res.figures.items():
        fig.savefig(ANALYSIS / f"boxplots_{name}{end}.png", dpi=200, bbox_inches="tight")
    pairwise = res.pairwise
    write(ANALYSIS / f"HEATWATCH_analysis{end}.xlsx", None, [
        ("Results", [("Group comparison", f"HEI with {he.HR_MAX['hr_max_observed']}; median (IQR); HEI: all participants (primary), exposed yes / no (Fisher), exposed only", res.table),
                     ("Pairwise (Mann-Whitney U, Bonferroni)", "only when Kruskal-Wallis p < 0.05",
                      pairwise if len(pairwise) else pd.DataFrame({"": ["no significant overall test"]})),
                     ("Patients vs healthy runners", "planned comparison of these two groups only, every outcome; "
                      "r = rank-biserial (> 0: higher in patients), OR = odds ratio of being exposed", res.patients_vs_healthy),
                     ("Heat exposure", None, res.exposed.reset_index(names=""))]),
        ("Participants", [("Inclusion and exclusion", f"{RACE_NAME[race]}; unique participants; parameters: {', '.join(params)}", res.flow),
                          ("Per group", None, res.participants.reset_index(names=""))]),
        ("Excluded (detail)", [(None, None, res.excluded)]),
        ("Per participant", [(None, None, res.per_participant)]),
        ("Resting HR fallback", [("Per participant", "trainings with a heat dose; fallback = no resting HR for "
                                  "the training's week, the participant's median used", res.fallback),
                                 ("Trainings with the fallback", "resting_hr_week empty, hr_rest = median used",
                                  res.fallback_trainings)])])
    print(f"analysis ({RACE_NAME[race]}) -> {ANALYSIS}  (HEATWATCH_analysis{end}.xlsx, "
          f"boxplots_direction / _heat / _components{end}.png)")


def saved_files(races=(None, "MR", "CPC")):
    """The files save_result() writes in analysis/ for these races: 3 boxplots + 1 .xlsx per race."""
    ends = [f"_{r}" if r else "" for r in races]
    return ({f"boxplots_{name}{e}.png" for name in FIGURES for e in ends}
            | {f"HEATWATCH_analysis{e}.xlsx" for e in ends})


def clear_old(races=(None, "MR", "CPC")):
    """Delete boxplots_*.png / boxplots.png and HEATWATCH_analysis*.xlsx in analysis/ that run() no longer makes
    (old figure names, the former sensitivity analysis _hrmax_training). route_maps/ is not touched."""
    keep = saved_files(races)
    old = [p for p in list(ANALYSIS.glob("boxplots*.png")) + list(ANALYSIS.glob("HEATWATCH_analysis*.xlsx"))
           if p.name not in keep]
    for p in old:
        p.unlink()
    print(f"removed {len(old)} old file(s) from {ANALYSIS}" + (f": {', '.join(sorted(p.name for p in old))}" if old else ""))
    return old
