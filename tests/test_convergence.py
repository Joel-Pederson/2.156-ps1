"""convergence.py: one long run contains every shorter run. Its GA score at a
generation, and its refinement score at a step count, must equal what a run.py job
with that n_gen (or grad_steps) logs. Tiny settings; sandbox folders only."""

import csv
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # draw to files, no windows

import pytest

import convergence
from linkopt import pipeline
from linkopt.config import preset

# Just big enough for the GA to get inside Kangaroo 3's limits in a few generations
# (at the smoke preset's 16 designs it finds nothing, and every score would be 0).
CFG = preset("smoke", n_start=32, pop_size=32, n_workers=0)
JOB = (2, 5, 1)  # Kangaroo 3, 5 joints, seed 1
SNAPSHOT, N_GEN, STEPS = 6, 8, (0, 5, 10)


@pytest.fixture(scope="module")
def rows():
    return convergence.measure(*JOB, CFG, N_GEN, SNAPSHOT, STEPS)


def run_py_job(steps):
    """What run.py runs (and logs) with n_gen = SNAPSHOT and grad_steps = steps."""
    cfg = replace(CFG, n_gen=SNAPSHOT, grad_steps=steps)
    return pipeline.run_job(pipeline.Job("ga", *JOB), cfg), cfg


def test_one_score_per_generation_and_step_count(rows):
    assert [r["x"] for r in rows if r["stage"] == "ga"] == list(range(1, N_GEN + 1))
    assert [r["x"] for r in rows if r["stage"] == "refine"] == list(STEPS)


@pytest.mark.parametrize("steps", STEPS)
def test_scores_equal_a_run_py_job_with_those_settings(rows, steps):
    job, _ = run_py_job(steps)
    assert job.hv_ga > 0  # only a real test if the GA found designs
    curve = {(r["stage"], r["x"]): r["hypervolume"] for r in rows}
    assert curve[("ga", SNAPSHOT)] == pytest.approx(job.hv_ga, rel=1e-9)
    assert curve[("refine", steps)] == pytest.approx(job.hv_refined, rel=1e-9)


def test_run_saves_curves_figure_and_checks_itself(tmp_path, capsys):
    flags = ["--preset", "smoke", "--targets", "2", "--n-joints", "5", "--seeds", "1"]
    flags += ["--n-gen", str(N_GEN), "--snapshot-gen", str(SNAPSHOT), "--workers", "0"]
    flags += ["--grad-steps", *map(str, STEPS)]
    flags += ["--n-start", "32", "--pop-size", "32"]  # as CFG
    flags += ["--runs-dir", str(tmp_path / "runs")]
    assert convergence.main(flags) == 0

    (out,) = (tmp_path / "runs").glob("convergence-*")
    with open(out / "curves.csv") as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == N_GEN + len(STEPS)
    assert (out / "convergence.png").stat().st_size > 1000
    assert (out / "config.json").exists()
    printed = capsys.readouterr().out
    # the smoke preset's grad_steps is 10: GA at the snapshot + refined at 10 steps
    assert printed.count("match   ") == 2
    assert "MISMATCH" not in printed


def test_bad_settings_are_refused(tmp_path):
    flags = ["--n-gen", "100", "--snapshot-gen", "150", "--runs-dir", str(tmp_path)]
    assert convergence.main(flags) == 2
    assert not list(tmp_path.iterdir())  # nothing created
