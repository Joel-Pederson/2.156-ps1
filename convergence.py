"""Convergence mini test: how long should the GA and refinement run?

One long run contains every shorter run. With the same seed, a 600-generation GA
passes through exactly the states a 150-generation GA ends in, so recording the
score after EVERY generation answers "would more generations help?" for every
n_gen at once. Refinement is measured the same way, by refining the GA's
designs at one generation (--snapshot-gen, default: the preset's n_gen) for
several step counts.

    python convergence.py                       # 3 kangaroos x 5 joints x seeds 0-2 (~45 min)
    python convergence.py --n-joints 5 6 --seeds 0-4
    python convergence.py --dry-run             # list the jobs; run nothing

Measurement only: it never touches submissions/best.npy or the experiment logs.
Each run is saved in runs/convergence-<date-time>/: config.json (settings,
command, git commit), curves.csv (one row per generation and per step count)
and convergence.png (the curves; the dashed line is the current setting).

At the end it checks itself: it runs its first job once more the ordinary run.py
way (n_gen = the snapshot generation, the preset's grad_steps). Its GA score and
refined score must equal the curves at those points, or the measurement is wrong.
(Earlier logged jobs aren't used for this: a change to the course simulator, like
the LINKS sync of 2026-10-03, makes old scores impossible to reproduce.)
"""

import argparse
import csv
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save the figure to a file; no window

import numpy as np

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from linkopt import pipeline, report
from linkopt.archive import RUNS_DIR, hypervolume
from linkopt.config import preset
from linkopt.ga import run_ga
from linkopt.refine import refine
from run import _duration, _git_commit, _new_run_dir, parse_seeds

CURVE_COLUMNS = ["kangaroo", "n_joints", "seed", "stage", "x", "hypervolume", "seconds"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preset", default="full", help="base settings (default: full)"
    )
    parser.add_argument("--targets", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-joints", type=int, nargs="+", default=[5])
    parser.add_argument("--seeds", nargs="+", default=["0-2"], help="e.g. 0-2")
    parser.add_argument(
        "--n-gen", type=int, default=600, help="generations to run (default 600)"
    )
    parser.add_argument(
        "--snapshot-gen",
        type=int,
        help="the generation whose designs are refined (default: the preset's n_gen)",
    )
    parser.add_argument(
        "--grad-steps",
        type=int,
        nargs="+",
        default=[0, 250, 500, 1000, 2000, 3000],
        help="refinement step counts to measure (0 = the GA's designs alone)",
    )
    parser.add_argument("--n-start", type=int, help="default: the preset's")
    parser.add_argument("--pop-size", type=int, help="default: the preset's")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--dry-run", action="store_true", help="list the jobs; run nothing"
    )
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def measure(target, n_joints, seed, cfg, n_gen, snapshot_gen, grad_steps) -> list:
    """One job: the GA's score after every generation up to n_gen, then the score
    after refining the generation-`snapshot_gen` designs for each step count.

    Each score is exactly what a run.py job with that n_gen (or grad_steps) would
    log: hv_ga for the GA, hv_refined (the GA's designs + their refined versions)
    for refinement. Returns rows for curves.csv.
    """
    rows, snapshot = [], {}
    label = {"kangaroo": target + 1, "n_joints": n_joints, "seed": seed}
    start = time.perf_counter()

    def record(algorithm):
        # What the GA would return if it stopped now (pymoo's res.opt): the designs
        # inside both limits that nothing beats, or none yet.
        opt = algorithm.opt
        found = opt is not None and len(opt) > 0 and np.any(opt.get("FEAS"))
        F = opt.get("F") if found else np.empty((0, 2))
        gen = algorithm.n_gen
        rows.append(
            dict(
                label,
                stage="ga",
                x=gen,
                hypervolume=hypervolume(F, target),
                seconds=time.perf_counter() - start,
            )
        )
        if gen == snapshot_gen:  # keep these designs to refine below
            to_mech = algorithm.problem.to_mech
            snapshot["designs"] = [to_mech(x) for x in opt.get("X")] if found else []
            snapshot["F"] = F

    run_ga(target, n_joints, seed, replace(cfg, n_gen=n_gen), callback=record)

    # Refinement, as pipeline.run_job does it: keep both versions of each design.
    for steps in grad_steps:
        refined = refine(snapshot["designs"], target, replace(cfg, grad_steps=steps))
        F_all = np.vstack([snapshot["F"], refined.F_after[refined.steps > 0]])
        rows.append(
            dict(
                label,
                stage="refine",
                x=steps,
                hypervolume=hypervolume(F_all, target),
                seconds=refined.seconds,
            )
        )
    return rows


def rough_seconds(n_jobs, cfg, n_gen, grad_steps, workers) -> float:
    """Wall-clock estimate with pipeline's per-evaluation and per-step costs."""
    ga = (cfg.n_start + cfg.pop_size * n_gen) * pipeline.SECONDS_PER_GA_EVAL
    steps = sum(grad_steps) * len(cfg.step_sizes) * pipeline.SECONDS_PER_REFINE_STEP
    useful = min(max(workers, 1), pipeline.MAX_USEFUL_WORKERS, max(n_jobs, 1))
    speedup = 1 + (useful - 1) * pipeline.SPEEDUP_PER_EXTRA_WORKER
    return n_jobs * (ga + steps) / speedup + pipeline.STARTUP_SECONDS


def verify(rows, target, n_joints, seed, cfg, snapshot_gen) -> list:
    """Run one job the ordinary run.py way (n_gen = snapshot_gen, cfg.grad_steps) and
    compare: its hv_ga must equal the GA curve at snapshot_gen, and its hv_refined the
    refinement curve at cfg.grad_steps (if that step count was measured).
    Returns (what, measured here, run.py's) pairs."""
    job = pipeline.run_job(
        pipeline.Job("ga", target, n_joints, seed), replace(cfg, n_gen=snapshot_gen)
    )
    if job.error:
        raise RuntimeError(f"the check job failed: {job.error}")
    curve = {
        (r["stage"], r["x"]): r["hypervolume"]
        for r in rows
        if (r["kangaroo"], r["n_joints"], r["seed"]) == (target + 1, n_joints, seed)
    }
    checks = [
        (f"GA at generation {snapshot_gen}", curve[("ga", snapshot_gen)], job.hv_ga)
    ]
    if ("refine", cfg.grad_steps) in curve:
        checks.append(
            (
                f"refined, {cfg.grad_steps} steps",
                curve[("refine", cfg.grad_steps)],
                job.hv_refined,
            )
        )
    return checks


def main(argv=None) -> int:
    args = parse_args(argv)
    clock = time.perf_counter()
    command = "python convergence.py " + " ".join(
        argv if argv is not None else sys.argv[1:]
    )
    try:
        cfg = preset(
            args.preset,
            targets=tuple(args.targets),
            n_joints=tuple(args.n_joints),
            seeds=parse_seeds(args.seeds),
            n_workers=args.workers,
            **{
                name: getattr(args, name)
                for name in ("n_start", "pop_size")
                if getattr(args, name) is not None
            },
        )
        snapshot_gen = args.snapshot_gen or cfg.n_gen
        replace(cfg, n_gen=args.n_gen)  # Config checks the values
        for steps in args.grad_steps:
            replace(cfg, grad_steps=steps)
        if not 1 <= snapshot_gen <= args.n_gen:
            raise ValueError(
                f"--snapshot-gen must be between 1 and --n-gen ({args.n_gen}), "
                f"got {snapshot_gen}"
            )
    except ValueError as e:
        print(f"invalid settings: {e}")
        return 2

    jobs = [(t, n, s) for s in cfg.seeds for t in cfg.targets for n in cfg.n_joints]
    grad_steps = sorted(set(args.grad_steps))
    seconds = rough_seconds(len(jobs), cfg, args.n_gen, grad_steps, cfg.n_workers)
    print(
        f"{len(jobs)} jobs: {len(cfg.targets)} kangaroo(s) x {len(cfg.n_joints)} "
        f"size(s) x {len(cfg.seeds)} seed(s); GA to generation {args.n_gen}, refining "
        f"the generation-{snapshot_gen} designs for {grad_steps} steps"
    )
    print(f"rough time: ~{_duration(seconds)} with {cfg.n_workers} worker(s)")
    if args.dry_run:
        for t, n, s in jobs:
            print(f"  Kangaroo {t + 1}, {n} joints, seed {s}")
        print("dry run: nothing was run")
        return 0

    started = datetime.now()  # noqa: DTZ005 (local time)
    out = _new_run_dir(
        Path(args.runs_dir) / f"convergence-{started.strftime('%Y%m%d-%H%M%S')}"
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "command": command,
                "git_commit": _git_commit(),
                "started": started.isoformat(timespec="seconds"),
                "n_gen": args.n_gen,
                "snapshot_gen": snapshot_gen,
                "grad_steps": grad_steps,
                "config": cfg.to_dict(),
            },
            indent=2,
        )
    )
    print(f"saving to {out}")

    # Each job's rows are written the moment it finishes, so Ctrl+C keeps them.
    rows = []
    with open(out / "curves.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        writer.writeheader()

        def finished(job_rows, done):
            writer.writerows(job_rows)
            f.flush()
            rows.extend(job_rows)
            print(f"[{done}/{len(jobs)}] {_summary(job_rows, snapshot_gen)}")

        task = (cfg, args.n_gen, snapshot_gen, grad_steps)
        try:
            _run_all(jobs, task, cfg.n_workers, finished)
        except KeyboardInterrupt:
            print(f"stopped: {len(rows)} rows from the finished jobs are in {out}")
            return 130

    fig = report.plot_convergence(rows, snapshot_gen, cfg.grad_steps)
    fig.savefig(out / "convergence.png", dpi=150, bbox_inches="tight")

    t, n, s0 = jobs[0]
    print(
        f"\nCheck: Kangaroo {t + 1}, {n} joints, seed {s0} once more, the run.py way "
        f"(n_gen {snapshot_gen}, grad_steps {cfg.grad_steps}):"
    )
    for what, measured, run_py in verify(rows, t, n, s0, cfg, snapshot_gen):
        ok = abs(measured - run_py) <= 1e-9 * max(1.0, abs(run_py))
        print(
            f"  {'match   ' if ok else 'MISMATCH'} {what}: {measured:.6f} "
            f"(run.py: {run_py:.6f})"
        )
    print(f"\ndone in {_duration(time.perf_counter() - clock)}: {out}")
    return 0


def _run_all(jobs, task, workers, finished):
    """measure() every job, calling finished(rows, done) as each one ends. Ctrl+C
    stops the running jobs at once, as in pipeline.run_jobs: workers ignore Ctrl+C,
    and this process terminates them."""
    if workers == 0:
        for i, (t, n, s) in enumerate(jobs, 1):
            finished(measure(t, n, s, *task), i)
        return
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),  # JAX isn't safe to fork
        initializer=pipeline._ignore_ctrl_c,
    )
    try:
        futures = [pool.submit(measure, t, n, s, *task) for t, n, s in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            finished(future.result(), i)
    except KeyboardInterrupt:
        processes = list((getattr(pool, "_processes", None) or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
        raise
    pool.shutdown(wait=True)


def _summary(job_rows, snapshot_gen) -> str:
    """One line per finished job: its score at a few generations and step counts."""
    r0 = job_rows[0]
    ga = {r["x"]: r["hypervolume"] for r in job_rows if r["stage"] == "ga"}
    ref = {r["x"]: r["hypervolume"] for r in job_rows if r["stage"] == "refine"}
    last = max(ga)
    return (
        f"Kangaroo {r0['kangaroo']}, {r0['n_joints']} joints, seed {r0['seed']}: "
        f"GA {ga.get(snapshot_gen, 0):.3f} at generation {snapshot_gen}, "
        f"{ga[last]:.3f} at {last}; refined "
        + ", ".join(f"{h:.3f} at {x} steps" for x, h in sorted(ref.items()))
    )


if __name__ == "__main__":
    sys.exit(main())
