"""Run the pipeline: GA -> refinement for every kangaroo x mechanism size x seed, in
parallel; then pool the designs into submissions/best.npy if the score improves.

    python run.py --preset smoke                          # does it work? (~1 min)
    python run.py --preset quick --dry-run                # list jobs + rough time
    python run.py --preset quick --seeds 0-49             # 50 replicates
    python run.py --preset full --n-joints 10 12 --n-gen 200
    caffeinate -i python run.py --preset full             # overnight (Mac stays awake)
    python run.py --resume runs/20261001-221500           # finish a stopped run

Settings: start from --preset, then override any of them (see the README's
"Choosing settings" table). --targets are kangaroo indices: 0 = Kangaroo 1.

Each run is saved in runs/<date-time>/: config.json (settings, command, git
commit), jobs/ (each job's designs, saved as it finishes), jobs.csv (one row per
job), submission.npy + scores.json (this run alone), summary.txt.

Ctrl+C once: stop, keep and pool the finished jobs (resume later with --resume).
Ctrl+C again: quit immediately (finished jobs are still saved in the run folder).
"""

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from linkopt import pipeline
from linkopt.archive import BEST_PATH, RUNS_DIR
from linkopt.config import Config, preset

SETTING_FLAGS = {  # command-line flag -> Config field, for the simple settings
    "n_start": int,
    "pop_size": int,
    "n_gen": int,
    "grad_steps": int,
}


def parse_seeds(values) -> tuple[int, ...]:
    """'0-49' -> 0..49; '1 3 5' -> 1, 3, 5; mixes like '0-2 7' work too."""
    seeds = []
    for value in values:
        for part in str(value).split(","):
            if "-" in part.strip("-"):
                lo, hi = part.split("-")
                seeds += range(int(lo), int(hi) + 1)
            elif part:
                seeds.append(int(part))
    return tuple(seeds)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--preset", default="quick", choices=["smoke", "quick", "full"])
    parser.add_argument("--targets", type=int, nargs="+", help="kangaroos: 0 1 2")
    parser.add_argument("--n-joints", type=int, nargs="+", help="mechanism sizes")
    parser.add_argument("--seeds", nargs="+", help="e.g. 0-49, or 0 1 2")
    for name, kind in SETTING_FLAGS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--mutation-prob", help="a number in [0, 1], or none")
    parser.add_argument("--step-sizes", type=float, nargs="+")
    parser.add_argument(
        "--workers", type=int, help="parallel processes (0 = in this one)"
    )
    parser.add_argument(
        "--refine-best", action="store_true", help="also refine best.npy's designs"
    )
    parser.add_argument(
        "--no-update-best", action="store_true", help="don't touch best.npy"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the jobs; run nothing"
    )
    parser.add_argument("--resume", metavar="RUN_DIR", help="finish a stopped run")
    parser.add_argument("--best", default=str(BEST_PATH), help=argparse.SUPPRESS)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def config_from_args(args) -> Config:
    """The preset with every given flag applied (Config rejects bad values)."""
    overrides = {}
    if args.targets is not None:
        overrides["targets"] = tuple(args.targets)
    if args.n_joints is not None:
        overrides["n_joints"] = tuple(args.n_joints)
    if args.seeds is not None:
        overrides["seeds"] = parse_seeds(args.seeds)
    for name in SETTING_FLAGS:
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    if args.mutation_prob is not None:
        text = args.mutation_prob.strip().lower()
        overrides["mutation_prob"] = None if text == "none" else float(text)
    if args.step_sizes is not None:
        overrides["step_sizes"] = tuple(args.step_sizes)
    if args.workers is not None:
        overrides["n_workers"] = args.workers
    return preset(args.preset, **overrides)


def main(argv=None) -> int:
    args = parse_args(argv)

    # 1. Settings: a new run (from flags), or a stopped run to finish (--resume).
    if args.resume:
        run_dir = Path(args.resume)
        saved = json.loads((run_dir / "config.json").read_text())
        cfg = Config(**saved["config"])
        refine_best, update = saved["refine_best"], saved["update_best"]
        if args.workers is not None:  # the only thing a resume may change
            cfg = replace(cfg, n_workers=args.workers)
    else:
        try:
            cfg = config_from_args(args)
        except ValueError as e:
            print(f"invalid settings: {e}")
            return 2
        refine_best, update = args.refine_best, not args.no_update_best
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 (local time)
        run_dir = Path(args.runs_dir) / stamp

    # 2. The jobs (skipping any a resumed run already finished).
    jobs = pipeline.make_jobs(cfg, refine_best=refine_best)
    already = pipeline.finished_job_ids(run_dir) if args.resume else set()
    todo = [j for j in jobs if j.job_id not in already]
    n_ga = sum(j.kind == "ga" for j in jobs)
    print(
        f"{len(cfg.targets)} kangaroo(s) x {len(cfg.n_joints)} size(s) x {len(cfg.seeds)} "
        f"seed(s) = {n_ga} GA jobs"
        + (f" + {len(jobs) - n_ga} refine-best jobs" if refine_best else "")
    )
    if already:
        print(
            f"resuming {run_dir}: {len(already)} jobs already done, {len(todo)} to run"
        )
    seconds = pipeline.estimate_seconds(todo, cfg, cfg.n_workers)
    print(f"rough time: ~{_duration(seconds)} with {cfg.n_workers} worker(s)")

    if args.dry_run:
        for job in jobs:
            print(
                ("  done  " if job.job_id in already else "  todo  ") + job.describe()
            )
        print("dry run: nothing was run")
        return 0

    # 3. Record everything needed to reproduce (and resume) this run.
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "refine_best": refine_best,
                    "update_best": update,
                    "command": "python run.py "
                    + " ".join(argv if argv is not None else sys.argv[1:]),
                    "git_commit": _git_commit(),
                    "machine": platform.platform(),
                    "started": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
                },
                indent=2,
            )
            + "\n"
        )
    print(f"saving to {run_dir}")

    # 4. Run, with a progress bar and one line per finished job.
    bar = tqdm(total=len(todo), unit="job", desc="Jobs")

    def on_result(result, done, total):
        j = result.job
        if result.error:
            tqdm.write(
                f"[{done}/{total}] {j.describe()}: FAILED ({result.error.strip().splitlines()[-1]})"
            )
        else:
            tqdm.write(
                f"[{done}/{total}] {j.describe()}: hypervolume {result.hv_ga:.3f} -> "
                f"{result.hv_refined:.3f} after refining ({result.seconds:.0f} s)"
            )
        bar.update(1)

    best = pipeline.load_best(args.best) if refine_best else None
    stopped = False
    try:
        pipeline.run_jobs(
            todo, cfg, run_dir, cfg.n_workers, best=best, on_result=on_result
        )
    except KeyboardInterrupt:
        stopped = True
        tqdm.write("stopped: pooling the finished jobs (Ctrl+C again to quit now)")
    bar.close()

    # 5. Pool: this run alone, then into best.npy (only kept if it improves).
    if not pipeline.finished_job_ids(run_dir):
        print("no finished jobs to pool")
        return 1
    outcome = pipeline.pool_run(run_dir, update=update, best_path=args.best)
    summary = _summary(run_dir, outcome, stopped, update)
    (run_dir / "summary.txt").write_text(summary + "\n")
    print(summary)
    return 1 if stopped else 0


def _summary(run_dir, outcome, stopped, update) -> str:
    s = outcome.run_scores
    lines = [
        "=" * 60,
        f"run {run_dir.name}"
        + (
            " (STOPPED early; finish with --resume " + str(run_dir) + ")"
            if stopped
            else ""
        ),
        "this run alone: overall "
        + f"{s['Overall Score']:.4f}  ("
        + ", ".join(f"{k}: {v:.3f}" for k, v in s["Score Breakdown"].items())
        + ")",
    ]
    if update and outcome.best is not None:
        b = outcome.best
        verdict = "IMPROVED -> saved" if b.improved else "no improvement -> unchanged"
        lines.append(f"best.npy: {b.old_score:.4f} -> {b.new_score:.4f}  {verdict}")
    else:
        lines.append("best.npy: not updated (--no-update-best)")
    lines.append("=" * 60)
    return "\n".join(lines)


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _git_commit() -> str:
    """The code's git commit, plus '-dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
