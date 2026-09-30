"""Run many jobs (GA -> refinement) in parallel, save each as it finishes, then pool.

A *job* is one GA run: one kangaroo, one mechanism size, one seed, followed by
refinement of its designs. A run of `run.py` is a list of jobs:

    make_jobs     every (kangaroo, size, seed) combination in the settings, plus
                  optional "refine the current best" jobs
    run_jobs      runs them on several worker processes; each finished job is
                  saved to runs/<run>/jobs/<job id>.npy straight away, and a line
                  is added to runs/<run>/jobs.csv
    pool_run      pools every saved job's designs into this run's own submission,
                  and (unless told not to) into submissions/best.npy via
                  archive.update_best, which only saves an improvement

Because each job is saved the moment it finishes, a stopped run (Ctrl+C, a crash,
a closed lid) keeps its finished jobs, and `run.py --resume` runs only the rest. A
job with the same settings and seed always gives the same result, so resuming
gives the same answer as an uninterrupted run.
"""

import csv
import json
import multiprocessing
import signal
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from linkopt.archive import hypervolume, select, update_best
from linkopt.config import Config
from linkopt.ga import run_ga
from linkopt.refine import refine
from linkopt.submission import build_submission, load, problem_key, save
from LINKS.CP import N_PROBLEMS

JOB_COLUMNS = [  # one row per job in jobs.csv
    "job_id",
    "kind",
    "kangaroo",
    "n_joints",
    "seed",
    "designs",
    "hv_ga",
    "hv_refined",
    "seconds",
    "error",
]


# --- Jobs -----------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    """One unit of work. kind = "ga" (GA, then refine its designs) or
    "refine_best" (refine the designs already in best.npy; no GA)."""

    kind: str
    target: int  # kangaroo (0 = Kangaroo 1)
    n_joints: int = 0  # mechanism size (ga jobs)
    seed: int = 0  # random seed (ga jobs)

    @property
    def job_id(self) -> str:
        if self.kind == "refine_best":
            return f"k{self.target + 1}-refine-best"
        return f"k{self.target + 1}-{self.n_joints}j-s{self.seed}"

    def describe(self) -> str:
        if self.kind == "refine_best":
            return f"Kangaroo {self.target + 1}, refining the current best"
        return f"Kangaroo {self.target + 1}, {self.n_joints} joints, seed {self.seed}"


def make_jobs(cfg: Config, refine_best: bool = False) -> list[Job]:
    """Every (kangaroo, size, seed) combination in cfg, in that order."""
    jobs = [
        Job("ga", t, n, s) for t in cfg.targets for n in cfg.n_joints for s in cfg.seeds
    ]
    if refine_best:
        jobs += [Job("refine_best", t) for t in cfg.targets]
    return jobs


@dataclass
class JobResult:
    """What one job produced (also saved to disk as jobs/<job id>.npy)."""

    job: Job
    designs: list = field(default_factory=list)  # GA designs + refined versions
    hv_ga: float = 0.0  # hypervolume of the GA's designs alone
    hv_refined: float = 0.0  # hypervolume after adding the refined versions
    seconds: float = 0.0
    error: str = ""  # traceback if the job crashed (the run carries on)


def run_job(job: Job, cfg: Config, best_designs=None) -> JobResult:
    """Run one job, in a worker process. Never raises: a crash is recorded in
    `error` so the other jobs carry on."""
    start = time.perf_counter()
    try:
        if job.kind == "refine_best":
            # No GA: refine the designs already in best.npy for this kangaroo. Only
            # the ones that improved are new (the originals are already in best.npy).
            refined = refine(best_designs or [], job.target, cfg)
            designs = refined.moved()
            hv_ga = hypervolume(refined.F_before, job.target)
            F_all = np.vstack([refined.F_before, refined.F_after[refined.steps > 0]])
        else:
            ga = run_ga(job.target, job.n_joints, job.seed, cfg)
            refined = refine(ga.designs, job.target, cfg)
            designs = ga.designs + refined.moved()  # keep both versions
            hv_ga = hypervolume(ga.F, job.target)
            F_all = np.vstack([ga.F, refined.F_after[refined.steps > 0]])
        return JobResult(
            job=job,
            designs=designs,
            hv_ga=hv_ga,
            hv_refined=hypervolume(F_all, job.target),
            seconds=time.perf_counter() - start,
        )
    except Exception:  # noqa: BLE001 -- record any crash, don't stop the run
        return JobResult(
            job=job, seconds=time.perf_counter() - start, error=traceback.format_exc()
        )


# --- Running many jobs ----------------------------------------------------------


def run_jobs(jobs, cfg, run_dir, workers, best=None, on_result=None):
    """Run `jobs`, saving each result as it finishes. Returns the results.

    workers = 0 runs every job here, in this process (slower, but the debugger can
    step into it). best: the current best submission (needed by refine_best jobs).
    on_result(result, done, total) is called after each job is saved.

    Ctrl+C: jobs not yet started are cancelled, running ones are stopped, and
    KeyboardInterrupt is raised after the finished jobs are safely on disk.
    """
    run_dir = Path(run_dir)
    (run_dir / "jobs").mkdir(parents=True, exist_ok=True)

    def best_for(job):
        if job.kind != "refine_best" or best is None:
            return None
        return list(best[problem_key(job.target)])

    results, total = [], len(jobs)

    def finished(result):
        save_job(result, run_dir)
        results.append(result)
        if on_result:
            on_result(result, len(results), total)

    if workers == 0:
        for job in jobs:
            finished(run_job(job, cfg, best_for(job)))
        return results

    # spawn: fresh worker processes (JAX isn't safe to fork). Workers ignore Ctrl+C;
    # this process handles it, so only it decides what to keep.
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_ignore_ctrl_c,
    )
    pending = {pool.submit(run_job, job, cfg, best_for(job)) for job in jobs}
    try:
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                finished(future.result())
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()  # stop jobs that were mid-run
        raise
    pool.shutdown(wait=True)
    return results


def _ignore_ctrl_c():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def save_job(result: JobResult, run_dir) -> None:
    """jobs/<job id>.npy (its designs) + a row in jobs.csv. A crashed job gets a
    csv row but no .npy, so --resume will run it again."""
    run_dir = Path(run_dir)
    if not result.error:
        np.save(
            run_dir / "jobs" / f"{result.job.job_id}.npy",
            {"job": asdict(result.job), "designs": result.designs},
            allow_pickle=True,
        )
    csv_path = run_dir / "jobs.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOB_COLUMNS)
        if new_file:
            writer.writeheader()
        j = result.job
        writer.writerow(
            {
                "job_id": j.job_id,
                "kind": j.kind,
                "kangaroo": j.target + 1,
                "n_joints": j.n_joints or "",
                "seed": j.seed if j.kind == "ga" else "",
                "designs": len(result.designs),
                "hv_ga": round(result.hv_ga, 6),
                "hv_refined": round(result.hv_refined, 6),
                "seconds": round(result.seconds, 1),
                "error": result.error.strip().splitlines()[-1] if result.error else "",
            }
        )


def finished_job_ids(run_dir) -> set[str]:
    """Ids of the jobs already saved in run_dir (the ones --resume skips)."""
    return {p.stem for p in (Path(run_dir) / "jobs").glob("*.npy")}


# --- Pooling a run --------------------------------------------------------------


@dataclass
class RunOutcome:
    run_scores: dict  # the grader's scores for this run's designs alone
    best: object  # archive.UpdateResult, or None if best.npy wasn't updated


def pool_run(run_dir, update=True, best_path=None) -> RunOutcome:
    """Pool every job saved in run_dir: this run's own submission (submission.npy +
    scores.json in run_dir), then (if update) into best.npy via update_best."""
    run_dir = Path(run_dir)
    designs = {t: [] for t in range(N_PROBLEMS)}
    for path in sorted((run_dir / "jobs").glob("*.npy")):
        saved = np.load(path, allow_pickle=True).item()
        designs[saved["job"]["target"]] += saved["designs"]

    # This run on its own (so runs can be compared), scored by the grader.
    own = {t: select(d, t).designs for t, d in designs.items()}
    run_scores = save(build_submission(own), run_dir / "submission.npy")
    (run_dir / "scores.json").write_text(json.dumps(run_scores, indent=2) + "\n")

    best = None
    if update:
        kwargs = {"best_path": best_path} if best_path else {}
        best = update_best(designs, source=f"run.py {run_dir.name}", **kwargs)
    return RunOutcome(run_scores=run_scores, best=best)


def load_best(best_path):
    """The current best submission, or None if there isn't one yet."""
    return load(best_path) if Path(best_path).exists() else None


# --- Rough time estimate ---------------------------------------------------------

# Measured on an M1 Max (increment 5b benchmark: 8 Kangaroo 3 jobs, 7 joints): the GA
# costs ~2.5 ms per design it evaluates, and refinement ~12 ms per step per step size.
# Workers: 1 -> 55 s, 3 -> 30 s, 4 -> 29 s, 6 -> 29 s. Each job keeps ~2 cores busy,
# so beyond 3 workers the 8 performance cores are full and there's no more speedup.
# Real times vary with mechanism size; this is for planning.
SECONDS_PER_GA_EVAL = 2.5e-3
SECONDS_PER_REFINE_STEP = 0.012
SPEEDUP_PER_EXTRA_WORKER = 0.5  # 3 workers ~ 2x faster than 1
MAX_USEFUL_WORKERS = 3
STARTUP_SECONDS = 10  # starting workers (each compiles JAX) + pooling at the end


def estimate_seconds(jobs, cfg: Config, workers: int) -> float:
    """Rough wall-clock time for `jobs` with `workers` processes."""
    per_ga_job = (cfg.n_start + cfg.pop_size * cfg.n_gen) * SECONDS_PER_GA_EVAL
    per_refine = cfg.grad_steps * len(cfg.step_sizes) * SECONDS_PER_REFINE_STEP
    total = sum(
        per_refine if job.kind == "refine_best" else per_ga_job + per_refine
        for job in jobs
    )
    useful = min(max(workers, 1), MAX_USEFUL_WORKERS, max(len(jobs), 1))
    return total / (1 + (useful - 1) * SPEEDUP_PER_EXTRA_WORKER) + STARTUP_SECONDS
