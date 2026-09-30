"""The experiment (DOE) logs: two CSV files at the repo root, committed to git, so
every run by every teammate ends up in one table.

    experiments_jobs.csv   one row per finished job: its run, every setting it ran
                           with, and its own hypervolume (the DOE's response)
    experiments_log.csv    one row per run.py invocation (a --resume adds another):
                           the command, this run's own score, best.npy before -> after

Rows are only ever added (run.py appends a job's row the moment it finishes, so
a stopped run keeps its rows). .gitattributes merges these files with git's
"union" driver: when two teammates both add rows, git keeps both sides' lines.

For DOE comparisons use the rows with kind == "ga" and an empty error: a
refine_best row scores best.npy's designs before/after refining (a whole kangaroo's
best, not one GA run), and a crashed job's results are left empty.

Tests write to their own folder (run.py's hidden --log-dir), never to these files.
"""

from pathlib import Path

from linkopt.archive import ROOT
from linkopt.pipeline import JOB_COLUMNS, append_row, header_matches

LOG_DIR = ROOT
JOBS_LOG = "experiments_jobs.csv"
RUNS_LOG = "experiments_log.csv"

JOBS_LOG_COLUMNS = ["run_id", "finished", "git_commit", *JOB_COLUMNS]
RUNS_LOG_COLUMNS = [
    "run_id",  # the run folder's name, runs/<run_id>/ (links to experiments_jobs.csv)
    "started",
    "finished",
    "git_commit",
    "machine",
    "command",
    "sweep",  # e.g. "mutation_prob=none,0.3,0.7" (empty: no sweep)
    "resumed",
    "stopped",  # Ctrl+C: finish it with --resume
    "jobs_planned",
    "jobs_done",  # jobs saved in the run folder so far (all invocations)
    "jobs_failed",  # jobs that crashed in this invocation
    "seconds",  # this invocation's wall-clock time
    "run_score",  # this run's own designs, scored by the grader
    "hv_k1",
    "hv_k2",
    "hv_k3",
    "best_before",  # best.npy's overall score (empty with --no-update-best)
    "best_after",
    "improved",
]


def check(log_dir=LOG_DIR) -> None:
    """Raise ValueError if a log's header isn't the columns this code writes (a log
    from older code): appending would put every value under the wrong column."""
    for name, columns in ((JOBS_LOG, JOBS_LOG_COLUMNS), (RUNS_LOG, RUNS_LOG_COLUMNS)):
        path = Path(log_dir) / name
        if not header_matches(path, columns):
            raise ValueError(
                f"{path} doesn't have the columns this code writes. Rename it "
                f"(e.g. to {path.stem}_old.csv) and run again."
            )


def log_job(row: dict, log_dir=LOG_DIR) -> None:
    append_row(Path(log_dir) / JOBS_LOG, JOBS_LOG_COLUMNS, row)


def log_run(row: dict, log_dir=LOG_DIR) -> None:
    append_row(Path(log_dir) / RUNS_LOG, RUNS_LOG_COLUMNS, row)
