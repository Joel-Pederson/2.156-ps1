import copy
from pathlib import Path

import numpy as np
import pytest

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS is imported)
from linkopt.submission import fill_default_target_joints, load

ROOT = Path(__file__).resolve().parent.parent
BEST_PATH = ROOT / "submissions" / "best.npy"


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
