"""Every committed submission file meets the requirements stated in the starter
notebook (2_155_Fall_26_CP1_Starter_Notebook.ipynb, sections "Instructions" and
"Submission Format").

These checks are written directly from the notebook text and deliberately do NOT
reuse linkopt.submission.validate, so a bug in our own tooling can't hide a
violation. They run on every file in submissions/*.npy.
"""

import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from conftest import ROOT
from upstream_manifest import MANIFEST_PATH, sha256

import LINKS.CP
from LINKS.Optimization import Tools

SUBMISSION_FILES = sorted((ROOT / "submissions").glob("*.npy"))

# Numbers as stated in the starter notebook ("Instructions" section).
NOTEBOOK_MAX_JOINTS = 20
NOTEBOOK_MAX_MECHANISMS = 1000
NOTEBOOK_LIMITS = {  # problem key -> (distance limit, material limit)
    "Problem 1": (0.75, 10.0),  # Kangaroo 1 (round body)
    "Problem 2": (1.2, 10.0),  # Kangaroo 2 (no ears, no tail)
    "Problem 3": (1.75, 20.0),  # Kangaroo 3 (full meme)
}
NOTEBOOK_NORMALIZERS = {"Problem 1": 2.0, "Problem 2": 1.5, "Problem 3": 10.0}

per_file = pytest.mark.parametrize(
    "path", SUBMISSION_FILES, ids=[p.name for p in SUBMISSION_FILES]
)


@cache
def _load(path: Path):
    """Load the way the grader does: np.load(..., allow_pickle=True).item()."""
    arr = np.load(path, allow_pickle=True)
    assert arr.shape == (), (
        f"{path.name} holds an array of shape {arr.shape}, not a saved dict "
        "(save the submission dict itself with np.save)"
    )
    return arr.item()


def _mechanisms(path: Path):
    """(problem key, index, mechanism) for every mechanism in the file."""
    return [
        (key, i, m) for key, mechs in _load(path).items() for i, m in enumerate(mechs)
    ]


def test_there_is_a_submission_to_check():
    assert SUBMISSION_FILES, "no submissions/*.npy files; the deliverable is missing"


@per_file
def test_file_is_a_single_python_dict(path):
    """'a single numpy file ... This numpy file will be a standard python dictionary'"""
    assert isinstance(_load(path), dict)


@per_file
def test_keys_are_exactly_the_three_problems(path):
    """'Problem 1, Problem 2 and Problem 3 correspond to Kangaroo 1, 2 and 3'"""
    keys = sorted(_load(path))
    assert keys == ["Problem 1", "Problem 2", "Problem 3"], f"keys are {keys}"


@per_file
def test_every_problem_is_a_nonempty_list(path):
    """'if any of the problems are not in your submission dictionary or if the list
    in the dictionary is empty, a hypervolume of 0 will be assigned to that problem'"""
    for key, mechs in _load(path).items():
        assert isinstance(mechs, list), f"{key} is a {type(mechs).__name__}, not a list"
        assert len(mechs) > 0, f"{key} is empty and would score 0"


@per_file
def test_at_most_1000_mechanisms_per_problem(path):
    """'we limit the number of mechanisms you are permitted to submit as potential
    solutions for any given curve to 1000' (the grader scores only the first 1000)"""
    for key, mechs in _load(path).items():
        assert len(mechs) <= NOTEBOOK_MAX_MECHANISMS, (
            f"{key}: {len(mechs)} mechanisms; everything past the first 1000 is ignored"
        )


@per_file
def test_at_most_20_joints_per_mechanism(path):
    """'we want you to generate mechanisms with no more than 20 joints'"""
    too_big = [
        (k, i, len(m["x0"])) for k, i, m in _mechanisms(path) if len(m["x0"]) > 20
    ]
    assert not too_big, f"(problem, index, joints) over the limit: {too_big[:10]}"


@per_file
def test_mechanism_fields_match_the_format(path):
    """'x0': np.ndarray (N, 2), 'edges': np.ndarray (E, 2), 'fixed_joints':
    np.ndarray (F,), 'motor': np.ndarray (2,), 'target_joint': int index"""
    for key, i, m in _mechanisms(path):
        where = f"{key}[{i}]"
        assert {"x0", "edges", "fixed_joints", "motor", "target_joint"} <= set(m), where
        x0, edges, fixed, motor = m["x0"], m["edges"], m["fixed_joints"], m["motor"]
        for name, arr in [
            ("x0", x0),
            ("edges", edges),
            ("fixed_joints", fixed),
            ("motor", motor),
        ]:
            assert isinstance(arr, np.ndarray), f"{where}: {name} is not an np.ndarray"
        n = x0.shape[0] if x0.ndim == 2 else -1
        assert x0.ndim == 2 and x0.shape[1] == 2 and n > 0, (
            f"{where}: x0 shape {x0.shape}"
        )
        assert np.isfinite(x0).all(), f"{where}: x0 has NaN/inf"
        assert edges.ndim == 2 and edges.shape[1] == 2, (
            f"{where}: edges shape {edges.shape}"
        )
        assert fixed.ndim == 1, f"{where}: fixed_joints shape {fixed.shape}"
        assert motor.shape == (2,), f"{where}: motor shape {motor.shape}"
        for name, arr in [("edges", edges), ("fixed_joints", fixed), ("motor", motor)]:
            assert arr.dtype.kind in "iu", (
                f"{where}: {name} dtype {arr.dtype} is not integer"
            )
            assert ((arr >= 0) & (arr < n)).all(), (
                f"{where}: {name} refers to joints outside 0..{n - 1}"
            )


@per_file
def test_target_joint_is_set(path):
    """'if you do not provide target_joint, during evaluation we will automatically
    pick the most complex joint ... so be careful to include that'"""
    for key, i, m in _mechanisms(path):
        t = m["target_joint"]
        assert isinstance(t, (int, np.integer)) and not isinstance(t, bool), (
            f"{key}[{i}]: target_joint is {t!r}; the grader would silently pick a joint"
        )
        assert 0 <= t < len(m["x0"]), f"{key}[{i}]: target_joint {t} is not a joint"


@per_file
def test_motor_is_one_of_the_links(path):
    """'Motor: ... the pair of node indices associated with the actuated edge'"""
    for key, i, m in _mechanisms(path):
        links = {frozenset(e) for e in m["edges"].tolist()}
        assert frozenset(m["motor"].tolist()) in links, (
            f"{key}[{i}]: motor is not an edge"
        )


@per_file
def test_every_mechanism_is_within_its_distance_and_material_limits(path):
    """'Any mechanisms output with a distance to the target curve larger than the
    target's distance limit ... [or] total linkage lengths more than the target's
    material limit ... will not be used to calculate the score'

    Checked the way the grader checks it (same Tools settings, strict '<'), so no
    submission slot is wasted on a design that won't count.
    """
    tools = Tools(timesteps=200, max_size=20, material=True, scaled=False, device="cpu")
    tools.compile()
    curves = np.load(ROOT / "kangaroo_target_curves.npy")
    for p, (key, mechs) in enumerate(sorted(_load(path).items())):
        dist_limit, mat_limit = NOTEBOOK_LIMITS[key]
        distance, material = tools(
            [m["x0"] for m in mechs],
            [m["edges"] for m in mechs],
            [m["fixed_joints"] for m in mechs],
            [m["motor"] for m in mechs],
            curves[p],
            [m["target_joint"] for m in mechs],
        )
        bad = np.where(~((distance < dist_limit) & (material < mat_limit)))[0]
        assert bad.size == 0, (
            f"{key}: {bad.size}/{len(mechs)} designs outside distance < {dist_limit} and "
            f"material < {mat_limit}; e.g. index {bad[0]}: distance {distance[bad[0]]:.3f}, "
            f"material {material[bad[0]]:.3f}"
        )


def test_grader_enforces_the_numbers_stated_in_the_notebook():
    """The limits above are the ones LINKS.CP.evaluate_submission actually uses."""
    for i, key in enumerate(["Problem 1", "Problem 2", "Problem 3"]):
        assert tuple(LINKS.CP.REFERENCE_POINTS[i]) == NOTEBOOK_LIMITS[key]
        assert LINKS.CP.SCORE_NORMALIZERS[i] == NOTEBOOK_NORMALIZERS[key]
    assert LINKS.CP.MAX_JOINTS == NOTEBOOK_MAX_JOINTS
    source = Path(LINKS.CP.__file__).read_text()
    assert f"counter > {NOTEBOOK_MAX_MECHANISMS}" in source  # the grader's 1000 cutoff


@pytest.mark.parametrize(
    "relpath", ["LINKS/CP/__init__.py", "kangaroo_target_curves.npy"]
)
def test_grader_and_target_curves_are_the_course_versions(relpath):
    """All local scoring (and the limits check above) runs through these two files;
    if either were edited, our scores would not match the leaderboard's."""
    expected = json.loads(MANIFEST_PATH.read_text())["files"][relpath]
    assert sha256(ROOT / relpath) == expected, (
        f"{relpath} differs from the course version; restore it (git checkout -- {relpath})"
    )
