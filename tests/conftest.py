import copy
from pathlib import Path

import numpy as np
import pytest

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS is imported)
from linkopt.experiments import JOBS_LOG, LOG_DIR, RUNS_LOG
from linkopt.submission import fill_default_target_joints, load

ROOT = Path(__file__).resolve().parent.parent
BEST_PATH = ROOT / "submissions" / "best.npy"
# The team's real files: tests must only ever use sandbox copies of them.
REAL_FILES = [
    BEST_PATH,
    BEST_PATH.with_name("best_score.json"),
    LOG_DIR / JOBS_LOG,
    LOG_DIR / RUNS_LOG,
]


def _contents(path):
    return path.read_bytes() if path.exists() else None


@pytest.fixture(scope="session", autouse=True)
def _real_files_untouched():
    """Fail the test session if any test wrote the real best.npy or experiment logs."""
    before = {path: _contents(path) for path in REAL_FILES}
    yield
    changed = [path.name for path in REAL_FILES if _contents(path) != before[path]]
    assert not changed, (
        f"the real {changed} changed during the tests. If a real run.py was running "
        "at the same time, that's the cause: keep its changes (don't revert them) and "
        "re-run the tests after it finishes. Otherwise a test wrote them: tests must "
        "use a sandbox."
    )


@pytest.fixture(scope="session")
def _starter_mech():
    mech = np.load(ROOT / "starter_mechanism.npy", allow_pickle=True).item()
    return fill_default_target_joints([mech])[0]


@pytest.fixture
def starter_mech(_starter_mech):
    """The course's starter mechanism (motor stored as a list), with target_joint set."""
    return copy.deepcopy(_starter_mech)


@pytest.fixture(scope="session")
def _best_submission():
    return load(BEST_PATH)


@pytest.fixture
def best_submission(_best_submission):
    """The committed best submission: strict format, nonzero score."""
    return copy.deepcopy(_best_submission)
