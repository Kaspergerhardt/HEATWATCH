"""Route maps: the GPS route of a training coloured by heart rate and by pace, plus HR and pace along the
distance. Uses the 5-s data (lat, lon, hr, speed): the training needs 5-s HR and 5-s GPS (hr_5s, gps_5s).
HR and speed are smoothed over 30 s; pace only when running (>= 1.5 m/s). No background map is downloaded,
so the coordinates never leave the computer.

  usable(five_s, run_ids)                    the runs with enough 5-s data (>= 1 min with position and HR)
  plot(trainings, five_s, run_id)            one figure: map by HR | map by pace | HR and pace along the distance
  save_all(trainings, five_s, run_ids)       one .png per training -> <output_root>/analysis/route_maps/
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter

from ..settings import ANALYSIS

SMOOTH = 6            # 5-s rows in the rolling mean (30 s)
MIN_SPEED = 1.5       # m/s; slower (standing, walking) -> no pace
MIN_POINTS = 12       # 5-s rows with position and HR needed for a route map (1 min)
PACE = FuncFormatter(lambda x, _: f"{int(x)}:{int(round(x % 1 * 60)):02d}")     # 5.5 -> 5:30


# ------------------------------------------------------------------ 5-s data of a training: enough for a map, smoothed route
def usable(five_s, run_ids):
    """The runs with enough 5-s data with position and HR for a route map."""
    f = five_s[five_s.run_id.isin(run_ids) & five_s.lat.notna() & five_s.hr.notna()]
    n = f.groupby("run_id").size()
    return [r for r in run_ids if n.get(r, 0) >= MIN_POINTS]


def route(five_s, run_id):
    """5-s rows of one training with a position: smoothed HR, speed, pace (min/km) and distance (km)."""
    r = five_s[five_s.run_id == run_id].dropna(subset=["lat", "lon"]).sort_values("second").reset_index(drop=True)
    la, lo = np.radians(r.lat), np.radians(r.lon)
    step = 2 * 6371 * np.arcsin(np.sqrt(np.sin(la.diff() / 2) ** 2 +
                                        np.cos(la) * np.cos(la.shift()) * np.sin(lo.diff() / 2) ** 2))
    r["km"] = step.fillna(0).cumsum()
    speed = r.speed_ms.fillna(step * 1000 / r.second.diff())          # no speed in the file: from the GPS
    r["hr_s"] = r.hr.rolling(SMOOTH, center=True, min_periods=1).mean()
    r["speed_s"] = speed.rolling(SMOOTH, center=True, min_periods=1).mean()
    r["pace"] = (1000 / r.speed_s / 60).where(r.speed_s >= MIN_SPEED)
    return r


# ------------------------------------------------------------------ figures: one route map, all route maps
def coloured_route(ax, r, col, cmap, label, fast_on_top=False):
    """Route as a line coloured by `col` (colour scale: 5th-95th percentile), start o, finish s."""
    pts = np.c_[r.lon, r.lat]
    v = r[col].to_numpy()[1:]
    lo, hi = np.nanpercentile(v, [5, 95]) if np.isfinite(v).any() else (0, 1)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("lightgrey")                              # no value (e.g. standing still for pace)
    lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap=cm, norm=plt.Normalize(lo, hi), linewidth=3)
    lc.set_array(np.ma.masked_invalid(v))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.radians(r.lat.mean())))  # metres the same in both directions
    ax.plot(*pts[0], "o", color="black", ms=7, label="start")
    ax.plot(*pts[-1], "s", color="black", ms=7, label="finish")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8)
    cb = plt.colorbar(lc, ax=ax, shrink=0.8)
    cb.set_label(label)
    if fast_on_top:
        cb.ax.invert_yaxis()
        cb.ax.yaxis.set_major_formatter(PACE)


def plot(trainings, five_s, run_id):
    """Map coloured by HR, map coloured by pace, HR and pace along the distance."""
    r = route(five_s, run_id)
    if len(r) < MIN_POINTS or r.hr.notna().sum() < MIN_POINTS:
        raise ValueError(f"{run_id}: too little 5-s data for a route map "
                         f"({len(r)} rows with a position, {int(r.hr.notna().sum())} with HR)")
    t = trainings.set_index("run_id").loc[run_id]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    coloured_route(a1, r, "hr_s", "YlOrRd", "heart rate (bpm)")
    a1.set_title("Heart rate", fontweight="bold")
    coloured_route(a2, r, "pace", "viridis_r", "pace (min/km)", fast_on_top=True)
    a2.set_title("Pace", fontweight="bold")
    a3.plot(r.km, r.hr_s, color="#C0392B", lw=1.5)
    a3.set_xlabel("distance (km)")
    a3.set_ylabel("heart rate (bpm)", color="#C0392B")
    b = a3.twinx()
    b.plot(r.km, r.pace, color="#2E86AB", lw=1.2)
    if r.pace.notna().any():
        b.set_ylim(*np.nanpercentile(r.pace, [99, 1]))   # fast at the top
    b.yaxis.set_major_formatter(PACE)
    b.set_ylabel("pace (min/km)", color="#2E86AB")
    a3.set_title("Heart rate and pace along the route", fontweight="bold")
    a3.grid(alpha=0.3)
    what = f"race ({t.race})" if t.is_race == 1 else f"training for {t.race}"
    fig.suptitle(f"{t.study_id} ({t.population}) - {what}, {t.date} {t.start_time}, {r.km.iloc[-1]:.1f} km",
                 fontweight="bold")
    fig.tight_layout()
    plt.close(fig)                       # shown only when asked for, not on every call
    return fig


def save_all(trainings, five_s, run_ids, folder=ANALYSIS / "route_maps"):
    """One .png per run; runs with too little 5-s data are skipped (and listed)."""
    folder.mkdir(parents=True, exist_ok=True)
    ok = usable(five_s, run_ids)
    for i, run_id in enumerate(ok, 1):
        print(f"\r    route map {i}/{len(ok)}", end="", flush=True)
        plot(trainings, five_s, run_id).savefig(folder / f"{run_id}.png", dpi=150, bbox_inches="tight")
    skipped = [r for r in run_ids if r not in ok]
    print(f"\nroute maps -> {folder}")
    if skipped:
        print(f"skipped (less than 1 min of 5-s data with position and HR): {', '.join(skipped)}")
