"""Figures and tables for understanding results and for the report (issue #5).

Everything here only READS: run folders (runs/<run>/), the experiment logs and
best.npy. It never changes a submission. Scores shown come from the course's
grader (evaluate_submission) or the same scorer it uses (problem.evaluate).

Used by results.ipynb. The main pieces:

    score_table        the grader's scores for a submission file, plus the
                       leaderboard's number (the sum of the raw hypervolumes)
    plot_trade_off     a kangaroo's staircase: zoomed, and over the whole box
    plot_design        one mechanism + its traced curve over the kangaroo
    plot_front         every non-dominated design of a kangaroo, drawn
    read_jobs, doe_*   the DOE: per-job results by factor level
    seeds_curve        pooled hypervolume vs. number of seeds (when do more
                       replicates stop paying?)
    provenance         which jobs the designs in best.npy came from
    plot_convergence   convergence.py's curves: one job's score vs. generations
                       and vs. refinement steps
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from linkopt.archive import BEST_PATH, RUNS_DIR, _design_key, hypervolume, non_dominated
from linkopt.ga import target_curve
from linkopt.problem import evaluate
from linkopt.submission import TARGET_CURVES_PATH, load, to_entry
from LINKS.CP import REFERENCE_POINTS, evaluate_submission
from LINKS.Geometry import CurveEngine
from LINKS.Visualization import MechanismVisualizer

KANGAROOS = [
    "Kangaroo 1 (Round Body)",
    "Kangaroo 2 (No Ears, No Tail)",
    "Kangaroo 3 (Full Meme)",
]
BOX_AREA = [float(d * m) for d, m in REFERENCE_POINTS]  # 7.5, 12, 35
FACTORS = ["n_joints", "n_start", "pop_size", "n_gen", "mutation_prob", "grad_steps"]
# How each factor reads on a figure (the Config field name is what run.py takes)
FACTOR_LABELS = {
    "n_joints": "Number of Joints",
    "n_start": "Random Starting Mechanisms",
    "pop_size": "Population Size",
    "n_gen": "Generations",
    "mutation_prob": "Mutation Probability",
    "grad_steps": "Gradient Steps",
}


# --- Scores ------------------------------------------------------------------------


def score_table(path=BEST_PATH) -> dict:
    """The grader's scores for a submission file, plus the leaderboard's number.

    Returns {"overall": the grader's overall (mean of normalized), "leaderboard":
    the sum of the raw hypervolumes, "rows": one dict per kangaroo with hv, the
    normalized hv, and the share of its scoring box covered}.
    """
    scores = evaluate_submission(str(path), str(TARGET_CURVES_PATH))
    rows = []
    for t in range(len(REFERENCE_POINTS)):
        key = f"Problem {t + 1}"
        hv = scores["Score Breakdown"][key]
        rows.append(
            {
                "kangaroo": KANGAROOS[t],
                "hv": hv,
                "hv_norm": scores["Normalized Score Breakdown"][key],
                "box": BOX_AREA[t],
                "coverage": hv / BOX_AREA[t],
            }
        )
    return {
        "overall": scores["Overall Score"],
        "leaderboard": sum(r["hv"] for r in rows),
        "rows": rows,
    }


def submission_scores(submission) -> dict:
    """{target: (designs, F)} for a loaded submission; F = [distance, material]."""
    out = {}
    for t in range(len(REFERENCE_POINTS)):
        designs = list(submission[f"Problem {t + 1}"])
        F = (
            np.column_stack(evaluate(designs, target_curve(t)))
            if designs
            else np.empty((0, 2))
        )
        out[t] = (designs, F)
    return out


# --- The trade-off staircase --------------------------------------------------------


def plot_trade_off(F, target):
    """A kangaroo's designs on the distance-vs-material chart, two ways: zoomed to
    the designs, and over the whole scoring box from (0, 0)."""
    F = np.asarray(F)
    ref = REFERENCE_POINTS[target]
    hv = hypervolume(F, target)
    fig, (zoomed, full) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{KANGAROOS[target]}: Hypervolume {hv:.3f} "
        f"({hv / BOX_AREA[target]:.0%} of the Scoring Box)"
    )
    for ax, whole in ((zoomed, False), (full, True)):
        _staircase(ax, F, ref)
        if whole:
            ax.set_xlim(0, ref[1] * 1.05)
            ax.set_ylim(0, ref[0] * 1.05)
            ax.scatter(0, 0, marker="*", s=200, color="green", zorder=3, clip_on=False)
            ax.annotate(
                "Ideal Point (Unreachable)",
                (0, 0),
                xytext=(8, 8),
                textcoords="offset points",
            )
            ax.set_title("Whole Scoring Box")
        else:
            ax.set_title("Zoomed to the Designs")
    fig.tight_layout()
    return fig


def _staircase(ax, F, ref, highlight=None):
    """The shaded hypervolume (area covered up to the limits corner) + the designs.
    highlight: the index of one design to mark in green."""
    F = np.asarray(F)
    front = F[non_dominated(F)] if len(F) else F
    front = (
        front[(front[:, 0] < ref[0]) & (front[:, 1] < ref[1])] if len(front) else front
    )
    if len(front):
        order = np.argsort(front[:, 1])  # by material, left to right
        d, m = front[order, 0], front[order, 1]
        # the staircase outline: right along each design's distance, then down
        xs, ys = [m[0]], [ref[0]]
        for i in range(len(m)):
            xs += [m[i], m[i + 1] if i + 1 < len(m) else ref[1]]
            ys += [d[i], d[i]]
        xs += [ref[1]]
        ys += [ref[0]]
        ax.fill(xs, ys, color="#ff9900", alpha=0.5, label="Hypervolume (Score)")
    ax.scatter(F[:, 1], F[:, 0], s=14, color="navy", zorder=3, label="Designs")
    if highlight is not None:
        ax.scatter(
            [F[highlight, 1]],
            [F[highlight, 0]],
            s=80,
            color="limegreen",
            edgecolor="darkgreen",
            zorder=4,
            label="This Design",
        )
    ax.scatter(
        ref[1], ref[0], color="maroon", zorder=3, label="Limits (Reference Point)"
    )
    ax.set_xlabel("Material (Total Link Length)")
    ax.set_ylabel("Distance to the Kangaroo")
    _style(ax)
    ax.legend(loc="best", fontsize=8)


def _style(ax, grid_axis="both"):
    """MATLAB-like: box on, light grid behind the data."""
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.grid(True, axis=grid_axis, alpha=0.3)
    ax.set_axisbelow(True)


# --- Mechanisms ---------------------------------------------------------------------


def pick_designs(F) -> dict:
    """Indices of three designs along the front: closest fit, cheapest, and one in
    the middle of the trade-off."""
    F = np.asarray(F)
    front = non_dominated(F)
    by_material = front[np.argsort(F[front, 1])]
    return {
        "closest fit": int(front[np.argmin(F[front, 0])]),
        "middle": int(by_material[len(by_material) // 2]),
        "least material": int(front[np.argmin(F[front, 1])]),
    }


def plot_design(design, target, title=""):
    """One mechanism (left) and its traced curve over the kangaroo (right), aligned
    the way the grader aligns them (shifted and rotated, never resized)."""
    fig, (mech_ax, fit_ax) = plt.subplots(1, 2, figsize=(11, 5))
    _draw_design(design, target, mech_ax, fit_ax)
    d, m = evaluate([design], target_curve(target))
    fig.suptitle(
        f"{title or KANGAROOS[target]}: Distance {d[0]:.3f}, Material {m[0]:.2f}"
    )
    fig.tight_layout()
    return fig


def plot_front(designs, F, target, max_rows=6):
    """Designs on the front, each row: the mechanism, its fit, and where it sits on
    the staircase (green). A front with more than max_rows designs is SAMPLED: that
    many designs, evenly spaced from the closest fit to the least material."""
    F = np.asarray(F)
    front = non_dominated(F)
    front = front[np.argsort(F[front, 0])]  # closest fit first
    total = len(front)
    if total > max_rows:
        front = front[np.linspace(0, total - 1, max_rows).round().astype(int)]
    fig, axs = plt.subplots(
        len(front), 3, figsize=(15, 4.2 * len(front)), squeeze=False
    )
    for row, i in enumerate(front):
        _draw_design(designs[i], target, axs[row, 0], axs[row, 1])
        axs[row, 1].set_title(f"Distance {F[i, 0]:.3f} | Material {F[i, 1]:.2f}")
        _staircase(axs[row, 2], F, REFERENCE_POINTS[target], highlight=i)
    shown = "All" if len(front) == total else f"{len(front)} of {total}, Evenly Spaced"
    fig.suptitle(
        f"{KANGAROOS[target]}: Designs on the Front ({shown}; Closest Fit First)", y=1.0
    )
    fig.tight_layout()
    return fig


def _draw_design(design, target, mech_ax, fit_ax):
    MechanismVisualizer()(
        design["x0"],
        design["edges"],
        design["fixed_joints"],
        design["motor"],
        ax=mech_ax,
        highlight=design["target_joint"],
    )
    mech_ax.set_title(
        f"{len(design['x0'])} Joints, {len(design['edges'])} Links, "
        f"Traced Joint {design['target_joint']}"
    )
    from LINKS.Kinematics import MechanismSolver  # local: builds a solver per call

    paths = MechanismSolver(device="cpu")(
        design["x0"], design["edges"], design["fixed_joints"], design["motor"]
    )
    CurveEngine(normalize_scale=False, device="cpu").visualize_single_comparison(
        paths[design["target_joint"]], target_curve(target), ax=fit_ax
    )


# --- The DOE (per-job results) --------------------------------------------------------


def run_dirs(runs_dir=RUNS_DIR) -> list[Path]:
    """Run folders (those with a jobs.csv), oldest first."""
    return sorted(p.parent for p in Path(runs_dir).glob("*/jobs.csv"))


def read_jobs(run_dir) -> list[dict]:
    """The GA jobs of a run that finished without error, numbers as numbers."""
    rows = []
    with open(Path(run_dir) / "jobs.csv") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "ga" or r["error"]:
                continue
            row = {
                "job_id": r["job_id"],
                "kangaroo": int(r["kangaroo"]),
                "seed": int(r["seed"]),
            }
            for name in FACTORS:
                row[name] = _number(r.get(name, ""))
            for name in ("hv_ga", "hv_refined", "hv_refined_norm", "seconds"):
                row[name] = float(r[name])
            rows.append(row)
    return rows


def factors_varied(rows) -> list[str]:
    """The settings that take more than one value across these jobs."""
    return [f for f in FACTORS if len({r[f] for r in rows}) > 1]


def doe_table(rows, factor, value="hv_refined") -> dict:
    """{(kangaroo, level): {n, median, q1, q3, min, max, zeros}} over the seeds."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kangaroo"], r[factor])].append(r[value])
    table = {}
    for key, values in groups.items():
        v = np.array(values)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        table[key] = {
            "n": len(v),
            "median": med,
            "q1": q1,
            "q3": q3,
            "min": v.min(),
            "max": v.max(),
            "zeros": int((v == 0).sum()),
        }
    return table


def plot_doe_heatmap(rows, factor):
    """Kangaroo x factor level (MATLAB heatmap style). Colour: the median job's share
    of the scoring box (comparable across kangaroos). Text: the median hypervolume,
    [IQR: 25th-75th percentile, the middle half of the seeds], and how many seeds found at
    least one design within the limits (the rest score 0)."""
    table = doe_table(rows, factor)
    levels = sorted({lvl for _, lvl in table}, key=_sort_key)
    kangaroos = sorted({k for k, _ in table})
    cover = np.full((len(kangaroos), len(levels)), np.nan)
    fig, ax = plt.subplots(
        figsize=(2.6 * len(levels) + 2.5, 1.6 * len(kangaroos) + 1.6)
    )
    for i, k in enumerate(kangaroos):
        for j, lvl in enumerate(levels):
            s = table.get((k, lvl))
            if s is None:
                continue
            cover[i, j] = s["median"] / BOX_AREA[k - 1]
            text = (
                f"{s['median']:.2f}\n[{s['q1']:.2f} - {s['q3']:.2f}]\n"
                f"Seeds within Limits: {s['n'] - s['zeros']}/{s['n']}"
            )
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cover[i, j] > 0.22 else "black",
            )
    image = ax.imshow(cover, cmap="Blues", vmin=0, aspect="auto")
    fig.colorbar(image, ax=ax, label="Median Share of the Scoring Box Covered")
    # black cell borders, like MATLAB's heatmap
    ax.set_xticks(np.arange(-0.5, len(levels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(kangaroos)), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(len(levels)), [str(lvl) for lvl in levels])
    ax.set_yticks(range(len(kangaroos)), [KANGAROOS[k - 1] for k in kangaroos])
    ax.set_xlabel(_axis_label(factor))
    n = max(s["n"] for s in table.values())
    ax.set_title(
        f"Hypervolume of One Job by {FACTOR_LABELS.get(factor, factor)}: "
        f"Median [IQR] over {n} Seeds"
    )
    fig.tight_layout()
    return fig


def plot_doe_boxes(rows, factor):
    """Per kangaroo, MATLAB boxchart style: for each level, the spread of the jobs'
    own hypervolumes over the seeds (box = middle half, line = median, whiskers =
    the rest, dots = outliers)."""
    kangaroos = sorted({r["kangaroo"] for r in rows})
    fig, axs = plt.subplots(
        1, len(kangaroos), figsize=(5 * len(kangaroos), 4.6), squeeze=False
    )
    for ax, k in zip(axs[0], kangaroos):
        levels = sorted({r[factor] for r in rows if r["kangaroo"] == k}, key=_sort_key)
        data = [
            [r["hv_refined"] for r in rows if r["kangaroo"] == k and r[factor] == lvl]
            for lvl in levels
        ]
        ax.boxplot(
            data,
            tick_labels=[str(lvl) for lvl in levels],
            patch_artist=True,
            widths=0.5,
            boxprops={"facecolor": "#9ec5e8", "edgecolor": "#1f5fa0"},
            medianprops={"color": "#1f5fa0", "linewidth": 2},
            whiskerprops={"color": "#1f5fa0"},
            capprops={"color": "#1f5fa0"},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "#1f5fa0",
                "markeredgecolor": "none",
            },
        )
        n = max(len(d) for d in data)
        ax.set_title(f"{KANGAROOS[k - 1]} ({n} Seeds per Box)")
        ax.set_xlabel(_axis_label(factor))
        ax.set_ylabel("One Job's Shaded Area (Hypervolume before Pooling)")
        _style(ax, grid_axis="y")
    fig.tight_layout()
    return fig


def plot_ga_vs_refined(rows, factor):
    """Each job's hypervolume measured twice: as its GA finishes (before refining)
    and after refining. Points above the diagonal: refinement (local search) added
    something."""
    kangaroos = sorted({r["kangaroo"] for r in rows})
    fig, axs = plt.subplots(
        1, len(kangaroos), figsize=(5 * len(kangaroos), 4.8), squeeze=False
    )
    for ax, k in zip(axs[0], kangaroos):
        sub = [r for r in rows if r["kangaroo"] == k]
        top = max([r["hv_refined"] for r in sub] + [1e-9]) * 1.05
        ax.plot([0, top], [0, top], color="grey", lw=1, label="No Change")
        for lvl in sorted({r[factor] for r in sub}, key=_sort_key):
            pts = [(r["hv_ga"], r["hv_refined"]) for r in sub if r[factor] == lvl]
            ax.scatter(*zip(*pts), s=14, label=_level_label(factor, lvl))
        ax.set_xlabel("Before Refining (GA Result)")
        ax.set_ylabel("After Refining (GA + Gradient)")
        ax.set_title(KANGAROOS[k - 1])
        ax.legend(fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


def job_F(run_dir) -> dict:
    """{job id: [distance, material] of each of its designs}, scored once."""
    out = {}
    for path in sorted((Path(run_dir) / "jobs").glob("*.npy")):
        saved = np.load(path, allow_pickle=True).item()
        designs = saved["designs"]
        t = saved["job"]["target"]
        F = (
            np.column_stack(evaluate(designs, target_curve(t)))
            if designs
            else np.empty((0, 2))
        )
        out[path.stem] = F
    return out


def seeds_curve(rows, scored, factor) -> dict:
    """{(kangaroo, level): (seed counts, pooled hypervolume)}: the hypervolume of all
    designs from the first k seeds of that level, for k = 1 .. all."""
    out = {}
    for k in sorted({r["kangaroo"] for r in rows}):
        for lvl in sorted(
            {r[factor] for r in rows if r["kangaroo"] == k}, key=_sort_key
        ):
            jobs = sorted(
                (r for r in rows if r["kangaroo"] == k and r[factor] == lvl),
                key=lambda r: r["seed"],
            )
            counts, hvs, pooled = [], [], np.empty((0, 2))
            for n, r in enumerate(jobs, 1):
                pooled = np.vstack([pooled, scored.get(r["job_id"], np.empty((0, 2)))])
                counts.append(n)
                hvs.append(hypervolume(pooled, k - 1))
            out[(k, lvl)] = (counts, hvs)
    return out


def plot_seeds_curve(curve, factor):
    """Pooled hypervolume vs. number of seeds, one line per level."""
    kangaroos = sorted({k for k, _ in curve})
    fig, axs = plt.subplots(
        1, len(kangaroos), figsize=(5 * len(kangaroos), 4.5), squeeze=False
    )
    for ax, k in zip(axs[0], kangaroos):
        for (kk, lvl), (counts, hvs) in sorted(
            curve.items(), key=lambda kv: _sort_key(kv[0][1])
        ):
            if kk == k:
                ax.plot(counts, hvs, marker=".", label=_level_label(factor, lvl))
        ax.set_title(KANGAROOS[k - 1])
        ax.set_xlabel("Number of Seeds Pooled")
        ax.set_ylabel("Pooled Hypervolume")
        ax.legend(fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


# --- Where best.npy's designs came from ----------------------------------------------


def provenance(best_path=BEST_PATH, runs_dir=RUNS_DIR) -> dict:
    """{target: Counter of origin labels} for the designs in best.npy. An origin is
    "<n> Joints" for a GA job, "Refine Best", or "Not in This Computer's Runs"
    (e.g. the starter baseline, or a teammate's merged file)."""
    origin = {}
    for run_dir in run_dirs(runs_dir):
        for path in sorted((run_dir / "jobs").glob("*.npy")):
            saved = np.load(path, allow_pickle=True).item()
            job = saved["job"]
            label = (
                "Refine Best"
                if job["kind"] == "refine_best"
                else f"{job['n_joints']} Joints"
            )
            for d in saved["designs"]:
                try:
                    origin.setdefault(
                        _design_key(to_entry(d)), (label, run_dir.name, job.get("seed"))
                    )
                except Exception:  # noqa: BLE001, S112 -- a design that can't be an entry can't be in best.npy
                    continue
    best = load(best_path)
    out = {}
    for t in range(len(REFERENCE_POINTS)):
        c = Counter()
        for d in best[f"Problem {t + 1}"]:
            c[origin.get(_design_key(d), ("Not in This Computer's Runs",))[0]] += 1
        out[t] = c
    return out


def plot_provenance(prov):
    """Per kangaroo, how many of best.npy's designs came from each kind of job."""
    labels = sorted({lbl for c in prov.values() for lbl in c}, key=_sort_key)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(prov))
    for lbl in labels:
        counts = np.array([prov[t].get(lbl, 0) for t in sorted(prov)])
        ax.bar(
            [KANGAROOS[t].split(" (")[0] for t in sorted(prov)],
            counts,
            bottom=bottom,
            label=lbl,
        )
        bottom += counts
    ax.set_ylabel("Designs in best.npy")
    ax.set_title("Where the Designs in Our Submission Came From")
    ax.legend(fontsize=8)
    _style(ax, grid_axis="y")
    fig.tight_layout()
    return fig


# --- Convergence (convergence.py) ---------------------------------------------------


def read_convergence(path) -> list[dict]:
    """A convergence run's curves.csv (or its folder), numbers as numbers."""
    path = Path(path)
    if path.is_dir():
        path = path / "curves.csv"
    with open(path) as f:
        return [
            {
                "kangaroo": int(r["kangaroo"]),
                "n_joints": int(r["n_joints"]),
                "seed": int(r["seed"]),
                "stage": r["stage"],
                "x": int(r["x"]),
                "hypervolume": float(r["hypervolume"]),
                "seconds": float(r["seconds"]),
            }
            for r in csv.DictReader(f)
        ]


def plot_convergence(rows, snapshot_gen, current_steps):
    """Per kangaroo, one job's score as it runs longer. Top: the GA, after every
    generation. Bottom: refinement of the generation-`snapshot_gen` designs, by
    step count. One line per job (size, seed); dashed = the current setting."""
    kangaroos = sorted({r["kangaroo"] for r in rows})
    fig, axs = plt.subplots(
        2, len(kangaroos), figsize=(5 * len(kangaroos), 8.5), squeeze=False
    )
    jobs = sorted({(r["n_joints"], r["seed"]) for r in rows})
    colors = dict(zip(jobs, plt.cm.tab10.colors * 10))
    for col, k in enumerate(kangaroos):
        ga_ax, ref_ax = axs[0, col], axs[1, col]
        for n, s in jobs:
            for ax, stage, marker in ((ga_ax, "ga", None), (ref_ax, "refine", "o")):
                pts = sorted(
                    (r["x"], r["hypervolume"])
                    for r in rows
                    if (r["kangaroo"], r["n_joints"], r["seed"], r["stage"])
                    == (k, n, s, stage)
                )
                if pts:
                    ax.plot(
                        *zip(*pts),
                        marker=marker,
                        markersize=4,
                        color=colors[(n, s)],
                        label=f"{n} Joints, Seed {s}",
                    )
        for ax, now in ((ga_ax, snapshot_gen), (ref_ax, current_steps)):
            ax.axvline(
                now, color="grey", ls="--", lw=1, label=f"Current Setting ({now})"
            )
            ax.legend(fontsize=8)
            _style(ax)
        ga_ax.set_title(f"{KANGAROOS[k - 1]}: GA")
        ga_ax.set_xlabel("Generation")
        ga_ax.set_ylabel("Hypervolume of One Job (GA Result)")
        ref_ax.set_title(f"Refining the Generation-{snapshot_gen} Designs")
        ref_ax.set_xlabel("Refinement Steps")
        ref_ax.set_ylabel("Hypervolume after Refining (GA + Gradient)")
    fig.tight_layout()
    return fig


# --- Saving -------------------------------------------------------------------------


def save_figures(figures: dict, folder) -> list[Path]:
    """Write each {name: figure} as folder/<name>.png. Returns the paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = folder / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    return paths


def _axis_label(factor):
    """'n_joints' -> 'Number of Joints (n_joints)': readable, plus the name run.py takes."""
    label = FACTOR_LABELS.get(factor)
    return f"{label} ({factor})" if label else factor


def _level_label(factor, level):
    """One level of a factor in a legend, e.g. 'Number of Joints = 5'."""
    return f"{FACTOR_LABELS.get(factor, factor)} = {level}"


def _number(text):
    """'7' -> 7, '0.3' -> 0.3, '' or 'None' -> None (e.g. mutation_prob = pymoo's default)."""
    if text in ("", "None", "none"):
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _sort_key(level):
    """Sort levels numerically, with None (pymoo's default mutation) first."""
    if level is None:
        return (0, 0.0, "")
    if isinstance(level, (int, float)):
        return (1, float(level), "")
    digits = "".join(ch for ch in str(level) if ch.isdigit())
    return (2, float(digits) if digits else 0.0, str(level))
