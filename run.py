"""Run the pipeline: GA -> refinement for every kangaroo x mechanism size x seed, in
parallel; then pool the designs into submissions/best.npy if the score improves.

    python run.py --preset smoke                          # does it work? (~1 min)
    python run.py --preset quick --dry-run                # list jobs + rough time
    python run.py --preset quick --seeds 0-49             # 50 replicates
    python run.py --preset full --n-joints 10 12 --n-gen 200
    python run.py --preset quick --seeds 0-4 --sweep mutation_prob=none,0.3,0.7
    caffeinate -i python run.py --preset full             # overnight (Mac stays awake)
    python run.py --resume runs/20261001-221500           # finish a stopped run

Settings: start from --preset, then override any of them (see the README's
"Choosing settings" table). --targets are kangaroo indices: 0 = Kangaroo 1.
--sweep NAME=V1,V2,... runs every value of a setting (repeatable: every
combination of every swept setting, for every kangaroo, size and seed).

Each run is saved in runs/<date-time>/: config.json (settings, command, git
commit), jobs/ (each job's designs, saved as it finishes), jobs.csv (one row per
job), submission.npy + scores.json (this run alone), summary.txt. Every job and
run is also added to the committed logs experiments_jobs.csv / experiments_log.csv.

Ctrl+C once: stop, keep and pool the finished jobs (resume later with --resume).
Ctrl+C again: quit immediately (finished jobs are still saved in the run folder).
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from linkopt import experiments, pipeline
from linkopt.archive import BEST_PATH, ROOT, RUNS_DIR
from linkopt.config import Config, preset

SETTING_FLAGS = {  # command-line flag -> Config field, for the simple settings
    "n_start": int,
    "pop_size": int,
    "n_gen": int,
    "grad_steps": int,
    "warm_start": float,
}


def parse_mutation_prob(text):
    """'0.3' -> 0.3; 'none' -> None (pymoo's defaults: what the notebook runs)."""
    text = str(text).strip().lower()
    return None if text == "none" else float(text)


SWEEP_TYPES = {**SETTING_FLAGS, "mutation_prob": parse_mutation_prob}
NOT_SWEEPABLE = {  # the settings --sweep refuses, and what to do instead
    "targets": "list them instead (--targets 0 1 2): every one is run",
    "n_joints": "list them instead (--n-joints 6 7 8): every one is run",
    "seeds": "list them instead (--seeds 0-4): every one is run",
    "step_sizes": "each job already tries every step size and logs which one won "
    "(step_size_wins in the logs)",
    "n_workers": "it only changes speed, not results (use --workers)",
    "warm_start": "run it twice instead (--warm-start 0, then --warm-start 0.5): "
    "every run's full command is logged, so the two runs can be compared. Sweeping "
    "it would need a new column in the committed logs, which older code can't read",
}
# Flags a --resume can't change (the run's saved settings are used), by args name.
RESUME_KEEPS = [
    "preset",
    "targets",
    "n_joints",
    "seeds",
    *SETTING_FLAGS,
    "mutation_prob",
    "step_sizes",
    "sweep",
    "refine_best",
    "no_update_best",
]
# Files runs themselves change: they don't make the code's git commit "-dirty".
DATA_FILES = [
    "experiments_*.csv",
    "submissions/best.npy",
    "submissions/best_score.json",
]


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


def parse_sweep(texts) -> tuple:
    """['mutation_prob=none,0.3', 'n_gen=50,100'] ->
    (('mutation_prob', (None, 0.3)), ('n_gen', (50, 100))).

    Values are read with each setting's own type; their ranges are checked by
    pipeline.make_jobs, which builds every combination before anything runs."""
    sweep = []
    for text in texts or []:
        name, has_equals, values = text.partition("=")
        name = name.strip().replace("-", "_")
        if not has_equals:
            raise ValueError(f"--sweep {text!r}: expected NAME=V1,V2,...")
        if name in NOT_SWEEPABLE:
            raise ValueError(f"--sweep {name}: not sweepable; {NOT_SWEEPABLE[name]}")
        if name not in SWEEP_TYPES:
            raise ValueError(
                f"--sweep {name}: unknown setting; sweepable: {', '.join(SWEEP_TYPES)}"
            )
        try:
            parsed = tuple(
                SWEEP_TYPES[name](v.strip()) for v in values.split(",") if v.strip()
            )
        except ValueError:
            kind = "numbers or none" if name == "mutation_prob" else "whole numbers"
            raise ValueError(
                f"--sweep {name}: values must be {kind}, got {values!r}"
            ) from None
        sweep.append((name, parsed))
    return tuple(sweep)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preset", choices=["smoke", "quick", "full"], help="default: quick"
    )
    parser.add_argument("--targets", type=int, nargs="+", help="kangaroos: 0 1 2")
    parser.add_argument("--n-joints", type=int, nargs="+", help="mechanism sizes")
    parser.add_argument("--seeds", nargs="+", help="e.g. 0-49, or 0 1 2")
    for name, kind in SETTING_FLAGS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=kind)
    parser.add_argument("--mutation-prob", help="a number in [0, 1], or none")
    parser.add_argument("--step-sizes", type=float, nargs="+")
    parser.add_argument(
        "--sweep",
        action="append",
        metavar="NAME=V1,V2",
        help="run every value of a setting, e.g. mutation_prob=none,0.3,0.7 "
        f"(repeatable; sweepable: {', '.join(SWEEP_TYPES)})",
    )
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
    parser.add_argument(
        "--log-dir", default=str(experiments.LOG_DIR), help=argparse.SUPPRESS
    )
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
        overrides["mutation_prob"] = parse_mutation_prob(args.mutation_prob)
    if args.step_sizes is not None:
        overrides["step_sizes"] = tuple(args.step_sizes)
    if args.workers is not None:
        overrides["n_workers"] = args.workers
    return preset(args.preset or "quick", **overrides)


def main(argv=None) -> int:
    args = parse_args(argv)
    started, clock = datetime.now(), time.perf_counter()  # noqa: DTZ005 (local time)
    commit, machine = _git_commit(), _machine()
    command = "python run.py " + " ".join(argv if argv is not None else sys.argv[1:])

    # 1. Settings: a new run (from flags), or a stopped run to finish (--resume).
    #    Then the jobs: make_jobs builds every combination of every swept setting
    #    as a Config, so a bad value stops here, before anything runs.
    try:
        if args.resume:
            run_dir = Path(args.resume)
            # "is", not "in": 0 == False in Python, so --grad-steps 0 would slip by.
            unset = [None, False]  # argparse's values for flags not given
            given = [
                n for n in RESUME_KEEPS if not any(getattr(args, n) is u for u in unset)
            ]
            if given:
                flags = ", ".join("--" + n.replace("_", "-") for n in given)
                raise ValueError(
                    f"--resume reuses the run's saved settings, so {flags} would be "
                    "ignored; only --workers can change. Start a new run instead."
                )
            saved = json.loads((run_dir / "config.json").read_text())
            cfg = Config(**saved["config"])
            sweep = tuple((n, tuple(v)) for n, v in saved.get("sweep", {}).items())
            refine_best, update = saved["refine_best"], saved["update_best"]
            if args.workers is not None:  # the only thing a resume may change
                cfg = replace(cfg, n_workers=args.workers)
            if not pipeline.header_matches(run_dir / "jobs.csv", pipeline.JOB_COLUMNS):
                raise ValueError(
                    f"{run_dir} was made by older code (its jobs.csv has other "
                    "columns), so it can't be resumed. Start a new run instead."
                )
        else:
            cfg = config_from_args(args)
            sweep = parse_sweep(args.sweep)
            for name, _ in sweep:
                if getattr(args, name) is not None:
                    flag = "--" + name.replace("_", "-")
                    raise ValueError(
                        f"{name} is set by both {flag} and --sweep; use one"
                    )
            refine_best, update = args.refine_best, not args.no_update_best
            run_dir = Path(args.runs_dir) / started.strftime("%Y%m%d-%H%M%S")
        jobs = pipeline.make_jobs(cfg, refine_best=refine_best, sweep=sweep)
    except ValueError as e:
        print(f"invalid settings: {e}")
        return 2
    try:
        experiments.check(args.log_dir)
    except ValueError as e:
        print(f"can't add to the experiment logs: {e}")
        return 2

    # 2. The jobs to run now (skipping any a resumed run already finished).
    already = pipeline.finished_job_ids(run_dir) if args.resume else set()
    todo = [j for j in jobs if j.job_id not in already]
    n_ga = sum(j.kind == "ga" for j in jobs)
    swept = "".join(f" x {len(values)} {name}" for name, values in sweep)
    print(
        f"{len(cfg.targets)} kangaroo(s) x {len(cfg.n_joints)} size(s) x {len(cfg.seeds)} "
        f"seed(s){swept} = {n_ga} GA jobs"
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
    if not args.resume:
        run_dir = _new_run_dir(run_dir)
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "config": asdict(cfg),  # swept settings: see "sweep"
                    "sweep": {name: list(values) for name, values in sweep},
                    "refine_best": refine_best,
                    "update_best": update,
                    "command": command,
                    "git_commit": commit,
                    "machine": machine,
                    "started": started.isoformat(timespec="seconds"),
                },
                indent=2,
            )
            + "\n"
        )
    print(f"saving to {run_dir}")

    # 4. Run, with a progress bar and one line per finished job. Each finished job
    #    is also added to experiments_jobs.csv straight away.
    # The time left is estimated from the jobs finished so far ("?" until the first).
    bar = tqdm(
        total=len(todo),
        unit="job",
        desc="Jobs",
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed} elapsed, ~{remaining} left, {rate_fmt}]"
        ),
    )
    failed = []

    def on_result(result, done, total):
        j = result.job
        if result.error:
            failed.append(j.job_id)
            tqdm.write(
                f"[{done}/{total}] {j.describe()}: FAILED ({result.error.strip().splitlines()[-1]})"
            )
        else:
            tqdm.write(
                f"[{done}/{total}] {j.describe()}: hypervolume {result.hv_ga:.3f} -> "
                f"{result.hv_refined:.3f} after refining ({result.seconds:.0f} s)"
            )
        bar.update(1)
        row = {"run_id": run_dir.name, "finished": _now(), "git_commit": commit}
        try:
            experiments.log_job({**row, **pipeline.job_row(result, cfg)}, args.log_dir)
        except OSError as e:  # a log problem must not stop an overnight run
            tqdm.write(
                f"warning: couldn't add {j.job_id} to {experiments.JOBS_LOG} ({e}); "
                f"it's still in {run_dir / 'jobs.csv'}"
            )

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
    done = pipeline.finished_job_ids(run_dir)
    outcome = None
    if done:
        # A second bar: pooling a big run takes minutes, and its label says which
        # step it's on (scoring a kangaroo is one long step, so it moves in jumps).
        pool_bar = tqdm(
            total=1,
            unit="step",
            desc="Pooling",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed} elapsed{postfix}]",
        )

        def on_step(label, n, total):
            pool_bar.total, pool_bar.n = total, n
            pool_bar.set_postfix_str(label)  # also redraws the bar

        outcome = pipeline.pool_run(
            run_dir, update=update, best_path=args.best, on_step=on_step
        )
        pool_bar.close()
        summary = _summary(run_dir, outcome, stopped, update)
        (run_dir / "summary.txt").write_text(summary + "\n")
        print(summary)
    else:
        print("no finished jobs to pool")

    # 6. This invocation's row in experiments_log.csv.
    row = {
        "run_id": run_dir.name,
        "started": started.isoformat(timespec="seconds"),
        "finished": _now(),
        "git_commit": commit,
        "machine": machine,
        "command": command,
        "sweep": pipeline.format_sweep(sweep),
        "resumed": bool(args.resume),
        "stopped": stopped,
        "jobs_planned": len(jobs),
        "jobs_done": len(done),
        "jobs_failed": len(failed),
        "seconds": round(time.perf_counter() - clock, 1),
        **_score_columns(outcome),
    }
    try:
        experiments.log_run(row, args.log_dir)
    except OSError as e:
        print(f"warning: couldn't add this run to {experiments.RUNS_LOG} ({e})")
    if outcome is None:
        return 1
    return 1 if stopped else 0


def _score_columns(outcome) -> dict:
    """experiments_log.csv's score columns (left empty if nothing was pooled)."""
    if outcome is None:
        return {}
    s = outcome.run_scores
    columns = {"run_score": round(s["Overall Score"], 6)}
    for t in range(3):
        columns[f"hv_k{t + 1}"] = round(s["Score Breakdown"][f"Problem {t + 1}"], 6)
    if outcome.best is not None:
        b = outcome.best
        columns["best_before"] = round(b.old_score, 6)
        columns["best_after"] = round(b.new_score if b.improved else b.old_score, 6)
        columns["improved"] = b.improved
    return columns


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


def _new_run_dir(path: Path) -> Path:
    """Create and claim a new run folder: runs/<date-time>, or <date-time>-2, -3, ...
    if another run started in the same second (mkdir fails if the folder exists, so
    two runs can never share one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for n in range(1, 1000):
        candidate = path if n == 1 else path.with_name(f"{path.name}-{n}")
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"couldn't create a new run folder next to {path}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")  # noqa: DTZ005 (local time)


def _machine() -> str:
    """The OS, plus the chip on a Mac (e.g. 'Apple M1 Max'): job times depend on it."""
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        chip = ""
    return platform.platform() + (f", {chip}" if chip else "")


def _git_commit() -> str:
    """The code's git commit, plus '-dirty' if there are uncommitted changes to
    anything but the files runs themselves change (DATA_FILES)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", "."]
            + [f":(exclude){path}" for path in DATA_FILES],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
