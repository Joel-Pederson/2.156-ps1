"""Run many jobs (GA -> refinement) in parallel, save each as it finishes, then pool.

A *job* is one GA run: one kangaroo, one mechanism size, one seed (and, in a
sweep, one value of each swept setting), followed by refinement of its designs. A
run of `run.py` is a list of jobs:

    make_jobs     every (seed, kangaroo, size, swept values) combination in the
                  settings, plus optional "refine the current best" jobs
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
import itertools
import json
import multiprocessing
import signal
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from linkopt.archive import hypervolume, select, update_best
from linkopt.config import Config
from linkopt.ga import run_ga
from linkopt.refine import refine, refine_material
from linkopt.submission import build_submission, load, problem_key, save
from LINKS.CP import N_PROBLEMS, SCORE_NORMALIZERS

# Settings that `run.py --sweep` can vary, with the short tag each gets in a job id
# (k2-7j-s1-mut0.3). targets / n_joints / seeds are already lists of their own, and
# step_sizes is a list per job (each job tries every size and logs which one won).
SWEEPABLE = {
    "n_start": "start",
    "pop_size": "pop",
    "n_gen": "gen",
    "mutation_prob": "mut",
    "grad_steps": "steps",
}

SETTING_COLUMNS = [
    "n_start",
    "pop_size",
    "n_gen",
    "mutation_prob",
    "grad_steps",
    "step_sizes",
]
GA_SETTINGS = {"n_start", "pop_size", "n_gen", "mutation_prob"}  # unused by refine_best

JOB_COLUMNS = [  # one row per job in jobs.csv (and in the experiments_jobs.csv log)
    "job_id",
    "kind",
    "kangaroo",
    "n_joints",
    "seed",
    *SETTING_COLUMNS,  # every setting the job ran with, swept or not
    "designs",
    "hv_ga",
    "hv_refined",
    "hv_refined_norm",  # hv_refined / the kangaroo's normalizer: the grade's scale
    "step_size_wins",  # "0.0004:12 0.0001:5 3e-05:0": designs each size refined best
    "seconds_ga",
    "seconds_refine",
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
    # This job's swept values, as (name, value) pairs, e.g. (("mutation_prob", 0.3),).
    # Empty = the run's settings as they are.
    settings: tuple = ()

    @property
    def job_id(self) -> str:
        """Unique within a run, and a safe filename: k2-7j-s1, k2-7j-s1-mut0.3."""
        tags = "".join(f"-{SWEEPABLE[n]}{format_value(v)}" for n, v in self.settings)
        if self.kind == "refine_best":
            return f"k{self.target + 1}-refine-best{tags}"
        return f"k{self.target + 1}-{self.n_joints}j-s{self.seed}{tags}"

    def describe(self) -> str:
        if self.kind == "refine_best":
            text = f"Kangaroo {self.target + 1}, refining the current best"
        else:
            text = (
                f"Kangaroo {self.target + 1}, {self.n_joints} joints, seed {self.seed}"
            )
        return text + "".join(f", {n}={format_value(v)}" for n, v in self.settings)

    def config(self, cfg: Config) -> Config:
        """The settings this job runs with: the run's, plus its swept values."""
        return replace(cfg, **dict(self.settings)) if self.settings else cfg


def format_value(value) -> str:
    """A setting's value as text, for job ids and logs: 0.3 -> '0.3', None -> 'none'."""
    return "none" if value is None else repr(value)


def format_sweep(sweep) -> str:
    """(("mutation_prob", (None, 0.3)),) -> 'mutation_prob=none,0.3'."""
    return " ".join(
        f"{name}=" + ",".join(format_value(v) for v in values) for name, values in sweep
    )


def make_jobs(cfg: Config, refine_best: bool = False, sweep=()) -> list[Job]:
    """Every (seed, kangaroo, size, swept values) combination, seeds outermost.

    Seeds outermost: a run stopped halfway has whole replicates (every kangaroo,
    size and swept value for seeds 0, 1, ...), not all of Kangaroo 1 and none of
    Kangaroo 3, so it's still a balanced experiment.

    sweep: ((name, (value, value, ...)), ...), e.g. (("mutation_prob", (0.3, 0.5)),).
    Every combination is checked here (built as a Config), so a bad value stops
    the run before anything starts.
    """
    names = [name for name, _ in sweep]
    for name, values in sweep:
        if name not in SWEEPABLE:
            raise ValueError(f"can't sweep {name!r}; sweepable: {', '.join(SWEEPABLE)}")
        if not values:
            raise ValueError(f"sweep {name} has no values")
        if len(set(values)) != len(values):
            raise ValueError(
                f"sweep {name} lists a value twice: {format_sweep([(name, values)])}"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"a setting is swept twice: {format_sweep(sweep)}")

    combos = [
        tuple(zip(names, values)) for values in itertools.product(*dict(sweep).values())
    ]
    for combo in combos:
        try:
            replace(cfg, **dict(combo))
        except ValueError as e:
            raise ValueError(
                f"sweep {format_sweep((n, (v,)) for n, v in combo)}: {e}"
            ) from None

    jobs = [
        Job("ga", t, n, s, combo)
        for s in cfg.seeds
        for t in cfg.targets
        for n in cfg.n_joints
        for combo in combos
    ]
    if refine_best:
        # No GA, so only the swept refinement settings (grad_steps) apply: one
        # refine-best job per kangaroo per value of those.
        refine_combos = dict.fromkeys(
            tuple((n, v) for n, v in combo if n not in GA_SETTINGS) for combo in combos
        )
        jobs += [
            Job("refine_best", t, settings=c)
            for t in cfg.targets
            for c in refine_combos
        ]
    return jobs


@dataclass
class JobResult:
    """What one job produced (also saved to disk as jobs/<job id>.npy)."""

    job: Job
    designs: list = field(default_factory=list)  # GA designs + refined versions
    hv_ga: float = 0.0  # hypervolume of the GA's designs alone
    hv_refined: float = 0.0  # hypervolume after adding the refined versions
    step_size_wins: dict = field(default_factory=dict)  # step size -> designs it won
    seconds_ga: float = 0.0
    # The first job in each worker also includes compiling JAX (GA and refinement).
    seconds_refine: float = 0.0
    seconds: float = 0.0
    error: str = ""  # traceback if the job crashed (the run carries on)


def run_job(job: Job, cfg: Config, best_designs=None) -> JobResult:
    """Run one job, in a worker process. Never raises: a crash is recorded in
    `error` so the other jobs carry on."""
    start = time.perf_counter()
    try:
        cfg = job.config(cfg)  # the run's settings + this job's swept values
        seconds_ga = 0.0
        if job.kind == "refine_best":
            # No GA: refine the designs already in best.npy for this kangaroo. Only
            # the ones that improved are new (the originals are already in best.npy).
            refined = refine(best_designs or [], job.target, cfg)
            designs = refined.moved()
            hv_ga = hypervolume(refined.F_before, job.target)
            F_all = np.vstack([refined.F_before, refined.F_after[refined.steps > 0]])
            source = best_designs or []
        else:
            ga = run_ga(job.target, job.n_joints, job.seed, cfg)
            seconds_ga = ga.seconds
            refined = refine(ga.designs, job.target, cfg)
            designs = ga.designs + refined.moved()  # keep both versions
            hv_ga = hypervolume(ga.F, job.target)
            F_all = np.vstack([ga.F, refined.F_after[refined.steps > 0]])
            source = ga.designs
        if cfg.refine_material:
            # The same designs walked downhill in material instead: the cheap end of
            # the front, which refining for distance alone never reaches.
            cheaper, F_cheaper = refine_material(source, job.target, cfg)
            designs += cheaper
            F_all = np.vstack([F_all, F_cheaper])
        return JobResult(
            job=job,
            designs=designs,
            hv_ga=hv_ga,
            hv_refined=hypervolume(F_all, job.target),
            # refined.step_size holds, per design, the size that gave its best (0 if
            # none improved it), so this counts the designs each size won.
            step_size_wins={
                s: int(np.sum(refined.step_size == s)) for s in cfg.step_sizes
            },
            seconds_ga=seconds_ga,
            seconds_refine=refined.seconds,
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
        save_job(result, cfg, run_dir)
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
        # Take the worker processes first: shutdown() forgets them (sets them to None).
        processes = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()  # stop jobs that were mid-run
        raise
    pool.shutdown(wait=True)
    return results


def _ignore_ctrl_c():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def save_job(result: JobResult, cfg: Config, run_dir) -> None:
    """jobs/<job id>.npy (its designs) + a row in jobs.csv. A crashed job gets a
    csv row but no .npy, so --resume will run it again."""
    run_dir = Path(run_dir)
    if not result.error:
        np.save(
            run_dir / "jobs" / f"{result.job.job_id}.npy",
            {"job": asdict(result.job), "designs": result.designs},
            allow_pickle=True,
        )
    append_row(run_dir / "jobs.csv", JOB_COLUMNS, job_row(result, cfg))


def job_row(result: JobResult, cfg: Config) -> dict:
    """One jobs.csv row: the job, every setting it ran with, and what it produced.

    A crashed job's results are left empty (not 0), so an average over the log
    can't mistake a crash for a job that found nothing; its error column says why.
    """
    j = result.job
    ran_with = j.config(cfg)
    ga = j.kind == "ga"
    settings = {
        name: format_value(getattr(ran_with, name))
        if ga or name not in GA_SETTINGS
        else ""
        for name in SETTING_COLUMNS
    }
    settings["step_sizes"] = " ".join(f"{s:g}" for s in ran_with.step_sizes)
    row = {
        "job_id": j.job_id,
        "kind": j.kind,
        "kangaroo": j.target + 1,
        "n_joints": j.n_joints if ga else "",
        "seed": j.seed if ga else "",
        **settings,
        "seconds": round(result.seconds, 1),
    }
    if result.error:
        return {**row, "error": result.error.strip().splitlines()[-1]}
    return {
        **row,
        "designs": len(result.designs),
        "hv_ga": round(result.hv_ga, 6),
        "hv_refined": round(result.hv_refined, 6),
        "hv_refined_norm": round(result.hv_refined / SCORE_NORMALIZERS[j.target], 6),
        "step_size_wins": " ".join(
            f"{s:g}:{n}" for s, n in result.step_size_wins.items()
        ),
        "seconds_ga": round(result.seconds_ga, 1),
        "seconds_refine": round(result.seconds_refine, 1),
    }


def header_matches(path, columns) -> bool:
    """True if the CSV file is missing, empty, or its header is exactly `columns`
    (appending to a file with other columns would misalign every value)."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return True
    with open(path, newline="") as f:
        return next(csv.reader(f), []) == list(columns)


def append_row(path, columns, row) -> None:
    """Add one row to a CSV file, writing the header first if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        if not new_file and _last_byte(path) != b"\n":
            f.write("\n")  # a hand-edited or merged file that lost its final newline
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _last_byte(path) -> bytes:
    with open(path, "rb") as f:
        f.seek(-1, 2)
        return f.read(1)


def finished_job_ids(run_dir) -> set[str]:
    """Ids of the jobs already saved in run_dir (the ones --resume skips)."""
    return {p.stem for p in (Path(run_dir) / "jobs").glob("*.npy")}


# --- Pooling a run --------------------------------------------------------------


@dataclass
class RunOutcome:
    run_scores: dict  # the grader's scores for this run's designs alone
    best: object  # archive.UpdateResult, or None if best.npy wasn't updated


def pool_run(run_dir, update=True, best_path=None, on_step=None) -> RunOutcome:
    """Pool every job saved in run_dir: this run's own submission (submission.npy +
    scores.json in run_dir), then (if update) into best.npy via update_best.

    on_step(label, done, total), if given, is called as each step starts (reading a
    job file, scoring a kangaroo's designs, grading, updating best.npy) and once at
    the end with done == total: run.py's pooling bar. A big run pools for minutes."""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "jobs").glob("*.npy"))
    total = len(paths) + N_PROBLEMS + 1 + (N_PROBLEMS + 2 if update else 0)
    done = 0

    def step(label):
        nonlocal done
        if on_step:
            on_step(label, done, total)
        done += 1

    designs = {t: [] for t in range(N_PROBLEMS)}
    for path in paths:
        step("reading job files")
        saved = np.load(path, allow_pickle=True).item()
        designs[saved["job"]["target"]] += saved["designs"]

    # This run on its own (so runs can be compared), scored by the grader.
    own = {}
    for t, d in designs.items():
        step(f"scoring Kangaroo {t + 1}'s {len(d):,} designs")
        own[t] = select(d, t).designs
    step("grading this run's submission")
    run_scores = save(build_submission(own), run_dir / "submission.npy")
    (run_dir / "scores.json").write_text(json.dumps(run_scores, indent=2) + "\n")

    best = None
    if update:
        kwargs = {"best_path": best_path} if best_path else {}
        best = update_best(
            designs, source=f"run.py {run_dir.name}", on_step=step, **kwargs
        )
    if on_step:
        on_step("done", total, total)
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
    total = 0.0
    for job in jobs:
        c = job.config(cfg)  # a swept job may have more generations, steps, ...
        total += c.grad_steps * len(c.step_sizes) * SECONDS_PER_REFINE_STEP
        if job.kind == "ga":
            total += (c.n_start + c.pop_size * c.n_gen) * SECONDS_PER_GA_EVAL
    useful = min(max(workers, 1), MAX_USEFUL_WORKERS, max(len(jobs), 1))
    return total / (1 + (useful - 1) * SPEEDUP_PER_EXTRA_WORKER) + STARTUP_SECONDS
