"""run.py + linkopt.pipeline: the right jobs, saved as they finish, resumable, and
pooled into best.npy only when the score improves."""

import csv
import json
import shutil
from pathlib import Path

import pytest
from conftest import BEST_PATH

import run
from linkopt import pipeline
from linkopt.config import preset
from linkopt.ga import target_curve
from linkopt.pipeline import Job, make_jobs
from linkopt.problem import evaluate
from linkopt.submission import load, validate
from LINKS.CP import REFERENCE_POINTS


@pytest.fixture
def sandbox(tmp_path):
    """A private copy of best.npy + an empty runs folder: the real ones are never touched."""
    subs = tmp_path / "submissions"
    subs.mkdir()
    shutil.copy2(BEST_PATH, subs / "best.npy")
    shutil.copy2(BEST_PATH.with_name("best_score.json"), subs / "best_score.json")
    runs = tmp_path / "runs"
    runs.mkdir()
    return {"best": subs / "best.npy", "runs": runs}


def args_for(sandbox, *flags):
    return [*flags, "--best", str(sandbox["best"]), "--runs-dir", str(sandbox["runs"])]


def only_run_dir(sandbox) -> Path:
    dirs = [p for p in sandbox["runs"].iterdir() if p.is_dir() and p.name[0].isdigit()]
    assert len(dirs) == 1
    return dirs[0]


def rows(run_dir):
    with open(run_dir / "jobs.csv") as f:
        return list(csv.DictReader(f))


# --- Settings and jobs ------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "seeds"),
    [
        (["0-3"], (0, 1, 2, 3)),
        (["0-2", "7"], (0, 1, 2, 7)),
        (["1,3"], (1, 3)),
        (["5"], (5,)),
    ],
)
def test_seed_ranges(values, seeds):
    assert run.parse_seeds(values) == seeds


def test_flags_override_the_preset():
    cfg = run.config_from_args(
        run.parse_args(
            [
                "--preset",
                "smoke",
                "--seeds",
                "0-4",
                "--n-gen",
                "7",
                "--mutation-prob",
                "none",
                "--step-sizes",
                "1e-4",
                "--targets",
                "2",
                "--workers",
                "0",
            ]
        )
    )
    assert cfg.seeds == (0, 1, 2, 3, 4) and cfg.n_gen == 7 and cfg.mutation_prob is None
    assert cfg.step_sizes == (1e-4,) and cfg.targets == (2,) and cfg.n_workers == 0
    assert (
        cfg.pop_size == preset("smoke").pop_size
    )  # untouched settings keep the preset's


def test_bad_settings_stop_before_anything_runs(sandbox, capsys):
    assert run.main(args_for(sandbox, "--n-joints", "21")) == 2
    assert "invalid settings" in capsys.readouterr().out
    assert not any(sandbox["runs"].iterdir())


def test_jobs_cover_every_combination():
    cfg = preset("quick", targets=(0, 2), n_joints=(6, 8), seeds=(0, 1, 2))
    jobs = make_jobs(cfg)
    assert len(jobs) == 2 * 2 * 3
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert {(j.target, j.n_joints, j.seed) for j in jobs} == {
        (t, n, s) for t in (0, 2) for n in (6, 8) for s in (0, 1, 2)
    }
    extra = make_jobs(cfg, refine_best=True)[len(jobs) :]
    assert [j.kind for j in extra] == ["refine_best", "refine_best"]


def test_dry_run_runs_and_writes_nothing(sandbox, capsys):
    assert (
        run.main(args_for(sandbox, "--preset", "quick", "--seeds", "0-2", "--dry-run"))
        == 0
    )
    out = capsys.readouterr().out
    assert "9 GA jobs" in out and "dry run" in out
    assert not any(sandbox["runs"].iterdir())


# --- Running, saving, resuming ----------------------------------------------------


def test_a_crashed_job_doesnt_stop_the_others(sandbox, monkeypatch):
    real = pipeline.run_ga

    def flaky(target, n_joints, seed, cfg, **kw):
        if seed == 1:
            raise RuntimeError("simulated crash")
        return real(target, n_joints, seed, cfg, **kw)

    monkeypatch.setattr(pipeline, "run_ga", flaky)
    jobs = [Job("ga", 2, 7, 0), Job("ga", 2, 7, 1), Job("ga", 2, 7, 2)]
    run_dir = sandbox["runs"] / "r"
    results = pipeline.run_jobs(jobs, preset("smoke"), run_dir, workers=0)
    assert [bool(r.error) for r in results] == [False, True, False]
    assert pipeline.finished_job_ids(run_dir) == {
        "k3-7j-s0",
        "k3-7j-s2",
    }  # crashed: not saved
    assert "simulated crash" in rows(run_dir)[1]["error"]


def test_ctrl_c_keeps_finished_jobs_and_resume_finishes(sandbox, monkeypatch, capsys):
    real = pipeline.run_job
    calls = {"n": 0}

    def interrupted_second_job(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "run_job", interrupted_second_job)
    flags = ("--preset", "smoke", "--targets", "2", "--seeds", "0-2", "--workers", "0")
    assert run.main(args_for(sandbox, *flags)) == 1  # stopped
    run_dir = only_run_dir(sandbox)
    assert pipeline.finished_job_ids(run_dir) == {"k3-7j-s0"}
    assert "STOPPED" in (run_dir / "summary.txt").read_text()
    assert (run_dir / "submission.npy").exists()  # the finished job was pooled

    monkeypatch.setattr(pipeline, "run_job", real)
    capsys.readouterr()
    assert run.main(["--resume", str(run_dir), "--best", str(sandbox["best"])]) == 0
    assert "1 jobs already done, 2 to run" in capsys.readouterr().out
    assert pipeline.finished_job_ids(run_dir) == {"k3-7j-s0", "k3-7j-s1", "k3-7j-s2"}
    assert len(rows(run_dir)) == 3


def test_no_update_best_leaves_it_alone(sandbox):
    before = sandbox["best"].read_bytes()
    flags = ("--preset", "smoke", "--targets", "2", "--workers", "0", "--refine-best")
    assert run.main(args_for(sandbox, *flags, "--no-update-best")) == 0
    assert sandbox["best"].read_bytes() == before
    assert "not updated" in (only_run_dir(sandbox) / "summary.txt").read_text()


def test_refine_best_improves_the_best(sandbox):
    old = json.loads(sandbox["best"].with_name("best_score.json").read_text())[
        "overall_score"
    ]
    flags = ("--preset", "smoke", "--targets", "1", "--workers", "0", "--refine-best")
    assert run.main(args_for(sandbox, *flags)) == 0
    new = json.loads(sandbox["best"].with_name("best_score.json").read_text())
    assert new["overall_score"] > old
    assert new["history"][-1]["source"].startswith("run.py ")


@pytest.mark.slow
def test_smoke_run_end_to_end_in_parallel(sandbox):
    """The whole pipeline on 2 worker processes: every output file, a valid
    submission, and a best that didn't get worse."""
    old = json.loads(sandbox["best"].with_name("best_score.json").read_text())[
        "overall_score"
    ]
    assert run.main(args_for(sandbox, "--preset", "smoke", "--workers", "2")) == 0
    run_dir = only_run_dir(sandbox)
    for name in (
        "config.json",
        "jobs.csv",
        "submission.npy",
        "scores.json",
        "summary.txt",
    ):
        assert (run_dir / name).exists(), name
    assert len(rows(run_dir)) == 3 and pipeline.finished_job_ids(run_dir)

    config = json.loads((run_dir / "config.json").read_text())
    assert config["config"]["n_workers"] == 2 and config["git_commit"]

    own = load(run_dir / "submission.npy")
    validate(own, strict=True)
    for t in range(3):
        if own[f"Problem {t + 1}"]:
            d, m = evaluate(own[f"Problem {t + 1}"], target_curve(t))
            assert (d < REFERENCE_POINTS[t][0]).all() and (
                m < REFERENCE_POINTS[t][1]
            ).all()

    validate(load(sandbox["best"]), strict=True)
    new = json.loads(sandbox["best"].with_name("best_score.json").read_text())[
        "overall_score"
    ]
    assert new >= old
