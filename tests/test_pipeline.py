"""run.py + linkopt.pipeline: the right jobs, saved as they finish, resumable,
pooled into best.npy only when the score improves, and (increment 5c) sweeps plus
the experiment logs."""

import csv
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import BEST_PATH

import run
from linkopt import experiments, pipeline
from linkopt.config import preset
from linkopt.ga import target_curve
from linkopt.pipeline import Job, make_jobs
from linkopt.problem import evaluate
from linkopt.submission import load, validate
from LINKS.CP import REFERENCE_POINTS, SCORE_NORMALIZERS


@pytest.fixture
def sandbox(tmp_path):
    """Private copies of best.npy, the runs folder and the experiment logs: the real
    ones are never touched."""
    subs = tmp_path / "submissions"
    subs.mkdir()
    shutil.copy2(BEST_PATH, subs / "best.npy")
    shutil.copy2(BEST_PATH.with_name("best_score.json"), subs / "best_score.json")
    runs = tmp_path / "runs"
    runs.mkdir()
    return {"best": subs / "best.npy", "runs": runs, "logs": tmp_path / "logs"}


def sandbox_flags(sandbox):
    return [
        "--best",
        str(sandbox["best"]),
        "--runs-dir",
        str(sandbox["runs"]),
        "--log-dir",
        str(sandbox["logs"]),
    ]


def args_for(sandbox, *flags):
    return [*flags, *sandbox_flags(sandbox)]


def only_run_dir(sandbox) -> Path:
    dirs = [p for p in sandbox["runs"].iterdir() if p.is_dir() and p.name[0].isdigit()]
    assert len(dirs) == 1
    return dirs[0]


def rows(run_dir, name="jobs.csv"):
    with open(Path(run_dir) / name) as f:
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
    crashed = rows(run_dir)[1]
    assert "simulated crash" in crashed["error"]
    # Its results are empty, not 0: an average can't mistake a crash for a bad job.
    assert (
        crashed["hv_refined"] == crashed["hv_refined_norm"] == crashed["designs"] == ""
    )


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
    assert run.main(["--resume", str(run_dir), *sandbox_flags(sandbox)]) == 0
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


# --- Sweeps and the experiment logs (increment 5c) ---------------------------------

SWEEP_FLAGS = (
    "--preset",
    "smoke",
    "--targets",
    "2",
    "--seeds",
    "0-1",
    "--sweep",
    "mutation_prob=0.3,0.7",
    "--workers",
    "0",
)
SWEEP_IDS = {f"k3-7j-s{s}-mut{m}" for s in (0, 1) for m in (0.3, 0.7)}


def test_sweep_values_are_read_with_each_settings_type():
    sweep = run.parse_sweep(["mutation_prob=none, 0.3", "n-gen=5,10"])
    assert sweep == (("mutation_prob", (None, 0.3)), ("n_gen", (5, 10)))
    assert all(type(v) is int for v in sweep[1][1])


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--sweep", "n_gen=0,5"], "n_gen must be at least 1"),
        (["--sweep", "mutation_prob=0.3,1.5"], "mutation_prob must be in [0, 1]"),
        (["--sweep", "n_gen=5.5"], "whole numbers"),
        (["--sweep", "mutation_prob=0.3,0.30"], "lists a value twice"),
        (["--sweep", "n_gen=5", "--sweep", "n_gen=6"], "swept twice"),
        (["--sweep", "n_gen="], "has no values"),
        (["--sweep", "n_gen"], "expected NAME=V1,V2"),
        (["--sweep", "step_sizes=1e-4"], "not sweepable"),
        (["--sweep", "seeds=0,1"], "not sweepable"),
        (["--sweep", "colour=red"], "unknown setting"),
        (["--n-gen", "5", "--sweep", "n_gen=5,6"], "both --n-gen and --sweep"),
    ],
)
def test_bad_sweeps_stop_before_anything_runs(sandbox, capsys, flags, message):
    assert run.main(args_for(sandbox, "--preset", "smoke", *flags)) == 2
    out = capsys.readouterr().out
    assert "invalid settings" in out and message in out
    assert not any(sandbox["runs"].iterdir())
    assert not sandbox["logs"].exists()


def test_sweep_jobs_cover_every_combination_seeds_outermost():
    cfg = preset("quick", targets=(0, 2), seeds=(0, 1))
    sweep = (("mutation_prob", (None, 0.5)), ("n_gen", (10, 20)))
    jobs = make_jobs(cfg, sweep=sweep)
    assert len(jobs) == 2 * 1 * 2 * 2 * 2  # kangaroos x sizes x seeds x 2 x 2
    ids = [j.job_id for j in jobs]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9._-]+", i) for i in ids)  # safe filenames
    assert "k3-7j-s1-mutnone-gen20" in ids
    assert [j.seed for j in jobs] == [0] * 8 + [1] * 8  # whole replicates first
    ran_with = {(j.config(cfg).mutation_prob, j.config(cfg).n_gen) for j in jobs}
    assert ran_with == {(m, g) for m in (None, 0.5) for g in (10, 20)}
    # No sweep: the same ids as before sweeps existed, seeds outermost.
    assert [j.job_id for j in make_jobs(cfg)] == [
        "k1-7j-s0",
        "k3-7j-s0",
        "k1-7j-s1",
        "k3-7j-s1",
    ]


def test_the_time_estimate_uses_each_jobs_own_settings():
    cfg = preset("quick")
    few, many = (Job("ga", 0, 7, 0, (("n_gen", g),)) for g in (10, 100))
    assert pipeline.estimate_seconds([many], cfg, 1) > pipeline.estimate_seconds(
        [few], cfg, 1
    )


def test_dry_run_lists_every_sweep_job_and_writes_nothing(sandbox, capsys):
    assert run.main(args_for(sandbox, *SWEEP_FLAGS, "--dry-run")) == 0
    out = capsys.readouterr().out
    assert "1 kangaroo(s) x 1 size(s) x 2 seed(s) x 2 mutation_prob = 4 GA jobs" in out
    cfg = run.config_from_args(run.parse_args(list(SWEEP_FLAGS)))
    expected = make_jobs(cfg, sweep=run.parse_sweep(["mutation_prob=0.3,0.7"]))
    listed = [
        line[len("  todo  ") :]
        for line in out.splitlines()
        if line.startswith("  todo  ")
    ]
    assert listed == [j.describe() for j in expected]
    assert not any(sandbox["runs"].iterdir()) and not sandbox["logs"].exists()


def test_a_sweep_logs_every_job_with_the_settings_it_ran_with(sandbox, monkeypatch):
    real, ran = pipeline.run_ga, []

    def recording(target, n_joints, seed, cfg, **kw):
        ran.append((seed, cfg.mutation_prob))
        return real(target, n_joints, seed, cfg, **kw)

    monkeypatch.setattr(pipeline, "run_ga", recording)
    assert run.main(args_for(sandbox, *SWEEP_FLAGS)) == 0
    assert sorted(ran) == [(0, 0.3), (0, 0.7), (1, 0.3), (1, 0.7)]  # its own value

    run_dir = only_run_dir(sandbox)
    assert pipeline.finished_job_ids(run_dir) == SWEEP_IDS
    config = json.loads((run_dir / "config.json").read_text())
    assert config["sweep"] == {"mutation_prob": [0.3, 0.7]}

    logged = rows(sandbox["logs"], experiments.JOBS_LOG)
    assert len(rows(run_dir)) == len(logged) == 4
    refining_helped = 0
    for row in logged:
        assert row["run_id"] == run_dir.name and row["git_commit"]
        assert row["job_id"].endswith("-mut" + row["mutation_prob"])
        assert row["n_gen"] == str(preset("smoke").n_gen)  # not swept: the preset's
        norm = float(row["hv_refined"]) / SCORE_NORMALIZERS[2]
        assert float(row["hv_refined_norm"]) == pytest.approx(norm, abs=1e-6)
        wins = [int(w.split(":")[1]) for w in row["step_size_wins"].split()]
        assert len(wins) == 3 and sum(wins) <= int(row["designs"])
        if float(row["hv_refined"]) > float(row["hv_ga"]):  # refining helped, so
            assert sum(wins) >= 1  # some step size must have won a design
            refining_helped += 1
        parts = float(row["seconds_ga"]) + float(row["seconds_refine"])
        assert parts <= float(row["seconds"]) + 0.2
    assert refining_helped  # (Kangaroo 3, seed 0 improves: 0.837 -> 1.409)
    assert sum(float(row["seconds_ga"]) for row in logged) > 0

    (run_row,) = rows(sandbox["logs"], experiments.RUNS_LOG)
    scores = json.loads((run_dir / "scores.json").read_text())
    assert run_row["run_id"] == run_dir.name
    assert run_row["sweep"] == "mutation_prob=0.3,0.7"
    assert (run_row["jobs_planned"], run_row["jobs_done"]) == ("4", "4")
    assert (run_row["stopped"], run_row["resumed"]) == ("False", "False")
    assert float(run_row["run_score"]) == pytest.approx(
        scores["Overall Score"], abs=1e-6
    )
    assert float(run_row["best_after"]) >= float(run_row["best_before"])


def test_resuming_a_sweep_runs_only_the_missing_jobs(sandbox, monkeypatch, capsys):
    real, calls = pipeline.run_job, {"n": 0}

    def interrupted_second_job(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "run_job", interrupted_second_job)
    assert run.main(args_for(sandbox, *SWEEP_FLAGS)) == 1  # stopped
    run_dir = only_run_dir(sandbox)
    assert pipeline.finished_job_ids(run_dir) == {"k3-7j-s0-mut0.3"}

    monkeypatch.setattr(pipeline, "run_job", real)
    capsys.readouterr()
    assert run.main(["--resume", str(run_dir), *sandbox_flags(sandbox)]) == 0
    assert "1 jobs already done, 3 to run" in capsys.readouterr().out
    assert pipeline.finished_job_ids(run_dir) == SWEEP_IDS
    logged = sorted(r["job_id"] for r in rows(sandbox["logs"], experiments.JOBS_LOG))
    assert logged == sorted(SWEEP_IDS)  # each job once
    first, second = rows(sandbox["logs"], experiments.RUNS_LOG)
    assert (first["stopped"], first["resumed"], first["jobs_done"]) == (
        "True",
        "False",
        "1",
    )
    assert (second["stopped"], second["resumed"], second["jobs_done"]) == (
        "False",
        "True",
        "4",
    )


def test_refine_best_jobs_follow_a_grad_steps_sweep_only():
    cfg = preset("quick", targets=(0, 2))
    sweep = (("grad_steps", (5, 10)), ("mutation_prob", (None, 0.5)))
    extra = [j for j in make_jobs(cfg, refine_best=True, sweep=sweep) if j.kind != "ga"]
    # One per kangaroo per grad_steps value; mutation (a GA setting) doesn't apply.
    assert [j.job_id for j in extra] == [
        "k1-refine-best-steps5",
        "k1-refine-best-steps10",
        "k3-refine-best-steps5",
        "k3-refine-best-steps10",
    ]
    assert [j.config(cfg).grad_steps for j in extra] == [5, 10, 5, 10]
    only_ga = (("mutation_prob", (None, 0.5)),)
    extra = [
        j for j in make_jobs(cfg, refine_best=True, sweep=only_ga) if j.kind != "ga"
    ]
    assert [j.job_id for j in extra] == ["k1-refine-best", "k3-refine-best"]


def test_resume_refuses_settings_it_would_ignore(sandbox, capsys):
    flags = ["--resume", str(sandbox["runs"] / "any"), *sandbox_flags(sandbox)]
    assert run.main([*flags, "--sweep", "mutation_prob=0.3,0.7", "--seeds", "0-2"]) == 2
    out = capsys.readouterr().out
    assert "--seeds, --sweep would be ignored; only --workers can change" in out
    assert run.main([*flags, "--grad-steps", "0"]) == 2  # 0 is a value, not "unset"
    assert "--grad-steps would be ignored" in capsys.readouterr().out


def test_a_run_folder_from_older_code_isnt_resumed(sandbox, capsys):
    run_dir = sandbox["runs"] / "20260101-000000"
    run_dir.mkdir()
    config = {"config": preset("smoke").to_dict(), "refine_best": False}
    (run_dir / "config.json").write_text(json.dumps({**config, "update_best": True}))
    (run_dir / "jobs.csv").write_text("job_id,kind,designs\nk1-7j-s0,ga,3\n")
    assert run.main(["--resume", str(run_dir), *sandbox_flags(sandbox)]) == 2
    assert "made by older code" in capsys.readouterr().out


def test_the_runs_own_data_files_dont_make_the_commit_dirty(tmp_path, monkeypatch):
    def git(*cmd):
        subprocess.run(["git", *cmd], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "submissions").mkdir()
    for name in ("code.py", "experiments_jobs.csv", "submissions/best.npy"):
        (tmp_path / name).write_text("v1\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    (tmp_path / "experiments_jobs.csv").write_text("v1\nrow\n")  # what runs change
    (tmp_path / "submissions/best.npy").write_text("v2\n")
    assert not run._git_commit().endswith("-dirty")
    (tmp_path / "code.py").write_text("v2\n")  # a real code change
    assert run._git_commit().endswith("-dirty")


def test_two_runs_started_in_the_same_second_get_their_own_folders(tmp_path):
    first = run._new_run_dir(tmp_path / "runs" / "20261001-221500")
    second = run._new_run_dir(tmp_path / "runs" / "20261001-221500")
    assert (first.name, second.name) == ("20261001-221500", "20261001-221500-2")
    assert first.is_dir() and second.is_dir()


def test_a_log_with_other_columns_stops_the_run(sandbox, capsys):
    sandbox["logs"].mkdir()
    (sandbox["logs"] / experiments.JOBS_LOG).write_text("job_id,hv\nk1-7j-s0,0.5\n")
    assert run.main(args_for(sandbox, "--preset", "smoke")) == 2
    assert "Rename it" in capsys.readouterr().out
    assert not any(sandbox["runs"].iterdir())


def test_rows_are_added_even_if_a_merge_dropped_the_final_newline(tmp_path):
    path = tmp_path / "log.csv"
    pipeline.append_row(path, ["a", "b"], {"a": 1, "b": 2})
    path.write_text(path.read_text().rstrip("\n"))
    pipeline.append_row(path, ["a", "b"], {"a": 3})
    assert path.read_text() == "a,b\n1,2\n3,\n"


def test_the_committed_logs_have_the_current_columns():
    experiments.check()  # reads the real logs; writes nothing


@pytest.mark.slow
def test_a_real_ctrl_c_with_workers_stops_cleanly(sandbox):
    """A real Ctrl+C (SIGINT) to run.py with worker processes: running jobs are
    stopped, the finished ones pooled, and the stopped run is logged."""
    cmd = [sys.executable, "run.py", "--preset", "smoke", "--targets", "2"]
    cmd += ["--seeds", "0-19", "--workers", "2", *sandbox_flags(sandbox)]
    proc = subprocess.Popen(
        cmd,
        cwd=Path(run.__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.time() + 180
    while not list(sandbox["runs"].glob("*/jobs/*.npy")):  # a job has finished
        assert time.time() < deadline and proc.poll() is None, "no job finished"
        time.sleep(0.2)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=120)

    assert proc.returncode == 1, out[-2000:]  # 1 = stopped early
    assert "Traceback" not in out, out[-2000:]
    run_dir = only_run_dir(sandbox)
    assert "STOPPED" in (run_dir / "summary.txt").read_text()
    assert (run_dir / "submission.npy").exists()  # the finished jobs were pooled
    assert 1 <= len(pipeline.finished_job_ids(run_dir)) < 20  # the rest were stopped
    (run_row,) = rows(sandbox["logs"], experiments.RUNS_LOG)
    assert run_row["stopped"] == "True"
