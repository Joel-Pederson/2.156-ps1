"""linkopt.report: the numbers behind the figures match the grader and the logs, and
every figure renders. Uses a tiny sandbox run (2 sizes x 2 seeds), never the real files."""

import json
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # draw to files, no windows

import matplotlib.pyplot as plt
import numpy as np
import pytest
from conftest import BEST_PATH

import run
from linkopt import report
from linkopt.archive import hypervolume, non_dominated
from linkopt.submission import TARGET_CURVES_PATH, load
from LINKS.CP import evaluate_submission


@pytest.fixture(scope="module")
def sandbox_run(tmp_path_factory):
    """A smoke run with n_joints 5 and 7, seeds 0-1, into a sandbox copy of best.npy."""
    root = tmp_path_factory.mktemp("report")
    subs = root / "submissions"
    subs.mkdir()
    shutil.copy2(BEST_PATH, subs / "best.npy")
    shutil.copy2(BEST_PATH.with_name("best_score.json"), subs / "best_score.json")
    logs = root / "logs"
    logs.mkdir()
    flags = [
        "--preset",
        "smoke",
        "--n-joints",
        "5",
        "7",
        "--seeds",
        "0-1",
        "--workers",
        "0",
    ]
    sandbox = [
        "--best",
        str(subs / "best.npy"),
        "--runs-dir",
        str(root / "runs"),
        "--log-dir",
        str(logs),
    ]
    assert run.main(flags + sandbox) == 0
    (run_dir,) = report.run_dirs(root / "runs")
    yield {"run": run_dir, "best": subs / "best.npy", "runs": root / "runs"}
    plt.close("all")


def test_score_table_matches_the_grader(sandbox_run):
    table = report.score_table(sandbox_run["best"])
    grader = evaluate_submission(str(sandbox_run["best"]), str(TARGET_CURVES_PATH))
    assert table["overall"] == grader["Overall Score"]
    raw = [grader["Score Breakdown"][f"Problem {t + 1}"] for t in range(3)]
    assert [r["hv"] for r in table["rows"]] == raw
    assert table["leaderboard"] == pytest.approx(sum(raw))  # the leaderboard's sum
    for r, area in zip(table["rows"], (7.5, 12.0, 35.0)):
        assert r["coverage"] == pytest.approx(r["hv"] / area)


def test_doe_reads_the_jobs_and_finds_the_factor(sandbox_run):
    rows = report.read_jobs(sandbox_run["run"])
    assert len(rows) == 3 * 2 * 2  # kangaroos x sizes x seeds
    assert report.factors_varied(rows) == ["n_joints"]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kangaroo"], r["n_joints"])].append(r["hv_refined"])
    table = report.doe_table(rows, "n_joints")
    for key, values in groups.items():
        assert table[key]["median"] == pytest.approx(np.median(values))
        assert table[key]["n"] == 2


def test_seeds_curve_pools_the_jobs(sandbox_run):
    rows = report.read_jobs(sandbox_run["run"])
    scored = report.job_F(sandbox_run["run"])
    curve = report.seeds_curve(rows, scored, "n_joints")
    for (k, n), (counts, hvs) in curve.items():
        assert counts == [1, 2]
        assert hvs[1] >= hvs[0]  # pooling more seeds never loses area
        jobs = [r["job_id"] for r in rows if r["kangaroo"] == k and r["n_joints"] == n]
        everything = np.vstack([scored[j] for j in jobs])
        assert hvs[-1] == pytest.approx(hypervolume(everything, k - 1))


def test_provenance_accounts_for_every_design(sandbox_run):
    prov = report.provenance(sandbox_run["best"], sandbox_run["runs"])
    best = load(sandbox_run["best"])
    for t in range(3):
        assert sum(prov[t].values()) == len(best[f"Problem {t + 1}"])


def test_picked_designs_are_on_the_front():
    best = load(BEST_PATH)
    _, F = report.submission_scores(best)[2]
    picks = report.pick_designs(F)
    front = set(non_dominated(F))
    assert set(picks.values()) <= front
    assert picks["closest fit"] == int(np.argmin(F[:, 0]))
    assert picks["least material"] == int(np.argmin(F[:, 1]))


def test_every_figure_renders_and_saves(sandbox_run, tmp_path):
    rows = report.read_jobs(sandbox_run["run"])
    scored = report.submission_scores(load(sandbox_run["best"]))
    designs, F = scored[2]
    figures = {
        "tradeoff": report.plot_trade_off(F, 2),
        "design": report.plot_design(designs[report.pick_designs(F)["middle"]], 2),
        "front": report.plot_front(designs, F, 2, max_rows=3),
        "front_one": report.plot_front(designs[:1], F[:1], 2),  # a single design
        "heatmap": report.plot_doe_heatmap(rows, "n_joints"),
        "boxes": report.plot_doe_boxes(rows, "n_joints"),
        "ga_vs_refined": report.plot_ga_vs_refined(rows, "n_joints"),
        "seeds": report.plot_seeds_curve(
            report.seeds_curve(rows, report.job_F(sandbox_run["run"]), "n_joints"),
            "n_joints",
        ),
        "provenance": report.plot_provenance(
            report.provenance(sandbox_run["best"], sandbox_run["runs"])
        ),
    }
    paths = report.save_figures(figures, tmp_path / "figures")
    assert [p.name for p in paths] == [f"{name}.png" for name in figures]
    assert all(p.stat().st_size > 1000 for p in paths)
    plt.close("all")


def test_score_history_rescored_from_the_backups(tmp_path):
    """Each history entry is matched to its backup and re-scored; a file that doesn't
    match its entry's recorded score is left out (hv None)."""
    subs = tmp_path / "submissions"
    backups = tmp_path / "runs" / "best_backups"
    subs.mkdir()
    backups.mkdir(parents=True)
    shutil.copy2(BEST_PATH, subs / "best.npy")
    shutil.copy2(BEST_PATH, backups / "best-20260101-000000.npy")
    shutil.copy2(BEST_PATH, backups / "best-20260102-000000.npy")
    grader = evaluate_submission(str(BEST_PATH), str(TARGET_CURVES_PATH))
    score = grader["Overall Score"]
    history = [
        {"date": "2026-01-01", "source": "starter", "overall_score": score},
        {"date": "2026-01-02 10:00", "source": "run.py A", "overall_score": score - 1},
        {"date": "2026-01-03 10:00", "source": "run.py B", "overall_score": score},
    ]
    (subs / "best_score.json").write_text(json.dumps({"history": history}))
    hist = report.score_history(subs / "best.npy")
    raw = [grader["Score Breakdown"][f"Problem {t + 1}"] for t in range(3)]
    assert [h["hv"] for h in hist] == [raw, None, raw]  # the middle one doesn't match
    assert hist[-1]["total"] == pytest.approx(sum(raw))
    fig = report.plot_score_history(hist, {"starter": "Start", "run.py B": "B"})
    assert len(fig.axes[0].patches) == 2 * 3  # 2 bars x 3 kangaroos
    plt.close("all")


def test_score_per_hour_pools_each_setting_over_its_hours(sandbox_run):
    rows = report.read_jobs(sandbox_run["run"])
    scored = report.job_F(sandbox_run["run"])
    curves = report.pooled_vs_hours(rows, scored)
    assert len(curves) == 3 * 2  # kangaroos x sizes; one (n_gen, grad_steps) setting
    for (k, n, _), curve in curves.items():
        hours = [h for h, _ in curve]
        assert hours == sorted(hours) and len(curve) == 2  # seeds 0-1
        jobs = [r for r in rows if r["kangaroo"] == k and r["n_joints"] == n]
        assert hours[-1] == pytest.approx(sum(r["seconds"] for r in jobs) / 3600)
        everything = np.vstack([scored[r["job_id"]] for r in jobs])
        assert curve[-1][1] == pytest.approx(hypervolume(everything, k - 1))
    report.plot_score_per_hour(curves)
    report.plot_interaction(rows, x_factor="n_joints", line_factor="n_gen")
    plt.close("all")


def test_make_report_figures_writes_every_figure(sandbox_run, tmp_path):
    run = sandbox_run["run"]
    paths = report.make_report_figures(
        tmp_path / "figs",
        doe_runs={"scan": run, "missing": tmp_path / "nowhere"},
        knob_sweep=run,
        best_path=sandbox_run["best"],
        runs_dir=sandbox_run["runs"],
    )
    names = {p.stem for p in paths}
    assert {"final_tradeoff_k3", "final_front_k1", "final_score_progression"} <= names
    assert {"scan_heatmap", "scan_seeds_curve", "knobsweep_interaction"} <= names
    assert not any(n.startswith("missing") for n in names)  # skipped, not an error
    assert (tmp_path / "figs" / "final_scores.txt").exists()
    assert all(p.stat().st_size > 1000 for p in paths)
