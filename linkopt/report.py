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
    "Kangaroo 1 (round body)",
    "Kangaroo 2 (no ears, no tail)",
    "Kangaroo 3 (full meme)",
]
BOX_AREA = [float(d * m) for d, m in REFERENCE_POINTS]  # 7.5, 12, 35
FACTORS = ["n_joints", "n_start", "pop_size", "n_gen", "mutation_prob", "grad_steps"]


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


def plot_trade_off(F, target, leader_hv=None):
    """A kangaroo's designs on the distance-vs-material chart, two ways: zoomed to
    the designs, and over the whole scoring box from (0, 0). leader_hv (optional)
    adds the leader's hypervolume for scale."""
    F = np.asarray(F)
    ref = REFERENCE_POINTS[target]
    hv = hypervolume(F, target)
    fig, (zoomed, full) = plt.subplots(1, 2, figsize=(12, 5))
    title = f"{KANGAROOS[target]}: hypervolume {hv:.3f} ({hv / BOX_AREA[target]:.0%} of the box)"
    if leader_hv is not None:
        title += f"  |  leader {leader_hv:.3f} ({leader_hv / BOX_AREA[target]:.0%})"
    fig.suptitle(title)
    for ax, whole in ((zoomed, False), (full, True)):
        _staircase(ax, F, ref)
        if whole:
            ax.set_xlim(0, ref[1] * 1.05)
            ax.set_ylim(0, ref[0] * 1.05)
            ax.scatter(0, 0, marker="*", s=200, color="green", zorder=3, clip_on=False)
            ax.annotate(
                "utopia (0, 0)", (0, 0), xytext=(8, 8), textcoords="offset points"
            )
            ax.set_title("Whole scoring box")
        else:
            ax.set_title("Zoomed to the designs")
    fig.tight_layout()
    return fig


def _staircase(ax, F, ref):
    """The shaded hypervolume (area covered up to the limits corner) + the designs."""
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
        ax.fill(xs, ys, color="#ff9900", alpha=0.5, label="hypervolume (score)")
    ax.scatter(F[:, 1], F[:, 0], s=14, color="navy", zorder=3, label="designs")
    ax.scatter(
        ref[1], ref[0], color="maroon", zorder=3, label="limits (reference point)"
    )
    ax.set_xlabel("Material (total link length)")
    ax.set_ylabel("Distance to the kangaroo")
    ax.legend(loc="best", fontsize=8)


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
        f"{title or KANGAROOS[target]}: distance {d[0]:.3f}, material {m[0]:.2f}"
    )
    fig.tight_layout()
    return fig


def plot_front(designs, F, target, max_rows=8):
    """Every non-dominated design of a kangaroo (up to max_rows, evenly spread along
    the front): the mechanism, its fit, and where it sits on the staircase."""
    F = np.asarray(F)
    front = non_dominated(F)
    front = front[np.argsort(F[front, 0])]  # closest fit first
    if len(front) > max_rows:
        front = front[np.linspace(0, len(front) - 1, max_rows).round().astype(int)]
    fig, axs = plt.subplots(
        len(front), 3, figsize=(15, 4.2 * len(front)), squeeze=False
    )
    for row, i in enumerate(front):
        _draw_design(designs[i], target, axs[row, 0], axs[row, 1])
        axs[row, 1].set_title(f"distance {F[i, 0]:.3f} | material {F[i, 1]:.2f}")
        ax = axs[row, 2]
        ax.scatter(F[front, 1], F[front, 0], color="royalblue")
        ax.scatter([F[i, 1]], [F[i, 0]], color="tomato", s=60)
        ax.set_xlabel("Material")
        ax.set_ylabel("Distance")
    fig.suptitle(
        f"{KANGAROOS[target]}: {len(non_dominated(F))} non-dominated designs", y=1.0
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
        f"{len(design['x0'])} joints, {len(design['edges'])} links, "
        f"traced joint {design['target_joint']}"
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
    """Kangaroo x factor level. Colour: the median job's share of the scoring box
    covered (comparable across kangaroos). Text: median hypervolume, the middle half
    of the seeds [25th-75th percentile], and how many seeds found nothing."""
    table = doe_table(rows, factor)
    levels = sorted({lvl for _, lvl in table}, key=_sort_key)
    kangaroos = sorted({k for k, _ in table})
    cover = np.full((len(kangaroos), len(levels)), np.nan)
    fig, ax = plt.subplots(figsize=(2.6 * len(levels) + 2, 1.6 * len(kangaroos) + 1.5))
    for i, k in enumerate(kangaroos):
        for j, lvl in enumerate(levels):
            s = table.get((k, lvl))
            if s is None:
                continue
            cover[i, j] = s["median"] / BOX_AREA[k - 1]
            ax.text(
                j,
                i,
                f"{s['median']:.2f}\n[{s['q1']:.2f}-{s['q3']:.2f}]"
                + (f"\n{s['zeros']}/{s['n']} found nothing" if s["zeros"] else ""),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cover[i, j] < 0.25 else "black",
            )
    image = ax.imshow(cover, cmap="viridis", vmin=0, aspect="auto")
    fig.colorbar(image, ax=ax, label="median share of the scoring box")
    ax.set_xticks(range(len(levels)), [str(lvl) for lvl in levels])
    ax.set_yticks(range(len(kangaroos)), [KANGAROOS[k - 1] for k in kangaroos])
    ax.set_xlabel(factor)
    n = max(s["n"] for s in table.values())
    ax.set_title(
        f"Per-job hypervolume after refining, by {factor} ({n} seeds per cell)"
    )
    fig.tight_layout()
    return fig


def plot_doe_boxes(rows, factor):
    """Per kangaroo: the spread over the seeds for each level (box = middle half,
    line = median, dots = individual seeds)."""
    kangaroos = sorted({r["kangaroo"] for r in rows})
    fig, axs = plt.subplots(
        1, len(kangaroos), figsize=(5 * len(kangaroos), 4.5), squeeze=False
    )
    rng = np.random.default_rng(0)
    for ax, k in zip(axs[0], kangaroos):
        levels = sorted({r[factor] for r in rows if r["kangaroo"] == k}, key=_sort_key)
        data = [
            [r["hv_refined"] for r in rows if r["kangaroo"] == k and r[factor] == lvl]
            for lvl in levels
        ]
        ax.boxplot(data, tick_labels=[str(lvl) for lvl in levels], showfliers=False)
        for j, values in enumerate(data, 1):
            ax.scatter(
                j + rng.uniform(-0.15, 0.15, len(values)), values, s=10, alpha=0.6
            )
        ax.set_title(KANGAROOS[k - 1])
        ax.set_xlabel(factor)
        ax.set_ylabel("per-job hypervolume after refining")
    fig.tight_layout()
    return fig


def plot_ga_vs_refined(rows, factor):
    """Each job's hypervolume from the GA alone vs. after refinement. Points above
    the diagonal: refinement (local search) added something."""
    kangaroos = sorted({r["kangaroo"] for r in rows})
    fig, axs = plt.subplots(
        1, len(kangaroos), figsize=(5 * len(kangaroos), 4.8), squeeze=False
    )
    for ax, k in zip(axs[0], kangaroos):
        sub = [r for r in rows if r["kangaroo"] == k]
        top = max([r["hv_refined"] for r in sub] + [1e-9]) * 1.05
        ax.plot([0, top], [0, top], color="grey", lw=1, label="no change")
        for lvl in sorted({r[factor] for r in sub}, key=_sort_key):
            pts = [(r["hv_ga"], r["hv_refined"]) for r in sub if r[factor] == lvl]
            ax.scatter(*zip(*pts), s=14, label=f"{factor} = {lvl}")
        ax.set_xlabel("hypervolume, GA only")
        ax.set_ylabel("hypervolume after refining")
        ax.set_title(KANGAROOS[k - 1])
        ax.legend(fontsize=8)
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
                ax.plot(counts, hvs, marker=".", label=f"{factor} = {lvl}")
        ax.set_title(KANGAROOS[k - 1])
        ax.set_xlabel("number of seeds pooled")
        ax.set_ylabel("pooled hypervolume")
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --- Where best.npy's designs came from ----------------------------------------------


def provenance(best_path=BEST_PATH, runs_dir=RUNS_DIR) -> dict:
    """{target: Counter of origin labels} for the designs in best.npy. An origin is
    "<n> joints" for a GA job, "refine-best", or "not in this computer's runs"
    (e.g. the starter baseline, or a teammate's merged file)."""
    origin = {}
    for run_dir in run_dirs(runs_dir):
        for path in sorted((run_dir / "jobs").glob("*.npy")):
            saved = np.load(path, allow_pickle=True).item()
            job = saved["job"]
            label = (
                "refine-best"
                if job["kind"] == "refine_best"
                else f"{job['n_joints']} joints"
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
            c[origin.get(_design_key(d), ("not in this computer's runs",))[0]] += 1
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
    ax.set_ylabel("designs in best.npy")
    ax.set_title("Where the submitted designs came from")
    ax.legend(fontsize=8)
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
