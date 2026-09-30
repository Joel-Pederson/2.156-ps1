"""The submission format from the starter notebook's "Submission Format" section,
and the grader's rules in LINKS/CP/__init__.py:evaluate_submission."""

import copy

import numpy as np
import pytest

import score
from linkopt.submission import (
    MAX_PER_PROBLEM,
    TARGET_CURVES_PATH,
    SubmissionError,
    build_submission,
    fill_default_target_joints,
    load,
    save,
    to_entry,
    validate,
)
from LINKS.CP import evaluate_submission


def test_to_entry_produces_the_exact_types(starter_mech):
    assert isinstance(starter_mech["motor"], list)  # as the course ships it
    entry = to_entry(starter_mech)

    assert set(entry) == {"x0", "edges", "fixed_joints", "motor", "target_joint"}
    n = entry["x0"].shape[0]
    assert entry["x0"].shape == (n, 2) and entry["x0"].dtype.kind == "f"
    assert entry["edges"].ndim == 2 and entry["edges"].shape[1] == 2
    assert entry["edges"].dtype.kind == "i"
    assert entry["fixed_joints"].ndim == 1 and entry["fixed_joints"].dtype.kind == "i"
    assert entry["motor"].shape == (2,) and entry["motor"].dtype.kind == "i"
    assert type(entry["target_joint"]) is int


def test_to_entry_requires_target_joint(starter_mech):
    starter_mech["target_joint"] = None
    with pytest.raises(SubmissionError, match="target_joint"):
        to_entry(starter_mech)


def test_to_entry_rejects_fractional_joint_indices(starter_mech):
    starter_mech["edges"] = np.array([[0, 1.5]])
    with pytest.raises(SubmissionError, match="whole-number"):
        to_entry(starter_mech)


def test_build_submission_maps_target_index_to_problem_key(starter_mech):
    submission = build_submission({1: [starter_mech]})
    assert list(submission) == ["Problem 1", "Problem 2", "Problem 3"]
    assert [len(v) for v in submission.values()] == [0, 1, 0]
    validate(submission, strict=True)


def test_build_submission_refuses_more_than_1000(starter_mech):
    with pytest.raises(SubmissionError, match="1000"):
        build_submission({0: [starter_mech] * (MAX_PER_PROBLEM + 1)})


def _break(submission, how):
    s = copy.deepcopy(submission)
    first = s["Problem 1"][0]
    if how == "missing problem key":
        del s["Problem 2"]
    elif how == "not a list":
        s["Problem 1"] = tuple(s["Problem 1"])
    elif how == "more than 1000":
        s["Problem 1"] = [first] * (MAX_PER_PROBLEM + 1)
    elif how == "21 joints":
        first["x0"] = np.random.rand(21, 2)
    elif how == "x0 wrong shape":
        first["x0"] = first["x0"].reshape(-1)
    elif how == "edge out of range":
        first["edges"] = np.vstack([first["edges"], [0, len(first["x0"])]])
    elif how == "missing motor":
        del first["motor"]
    elif how == "target_joint None":
        first["target_joint"] = None
    elif how == "target_joint out of range":
        first["target_joint"] = len(first["x0"])
    return s


@pytest.mark.parametrize(
    "how",
    [
        "missing problem key",
        "not a list",
        "more than 1000",
        "21 joints",
        "x0 wrong shape",
        "edge out of range",
        "missing motor",
        "target_joint None",
        "target_joint out of range",
    ],
)
def test_validate_rejects(best_submission, how):
    validate(best_submission, strict=True)  # the unbroken version is fine
    with pytest.raises(SubmissionError):
        validate(_break(best_submission, how), strict=True)


def test_starter_style_is_grader_compatible_but_not_strict(best_submission):
    # What the starter notebook writes: motor as a list, target_joint None.
    for mechs in best_submission.values():
        for m in mechs:
            m["motor"] = m["motor"].tolist()
            m["target_joint"] = None
    warnings = validate(best_submission, strict=False)
    assert any("motor is a list" in w for w in warnings)
    assert any("target_joint is missing or None" in w for w in warnings)
    with pytest.raises(SubmissionError):
        validate(best_submission, strict=True)


def test_fill_default_target_joints_keeps_the_grader_score(best_submission, tmp_path):
    starter_style = copy.deepcopy(best_submission)
    for mechs in starter_style.values():
        for m in mechs:
            m["target_joint"] = None
    filled = build_submission(
        {
            i: fill_default_target_joints(starter_style[f"Problem {i + 1}"])
            for i in range(3)
        }
    )
    np.save(tmp_path / "starter_style.npy", starter_style)
    assert save(filled, tmp_path / "filled.npy") == evaluate_submission(
        str(tmp_path / "starter_style.npy"), str(TARGET_CURVES_PATH)
    )


def test_save_round_trip_matches_the_in_memory_score(best_submission, tmp_path):
    from_file = save(best_submission, tmp_path / "sub.npy")
    in_memory = evaluate_submission(best_submission, str(TARGET_CURVES_PATH))
    assert from_file == in_memory
    assert from_file["Overall Score"] > 0
    validate(load(tmp_path / "sub.npy"), strict=True)


def test_save_requires_npy_suffix(best_submission, tmp_path):
    with pytest.raises(SubmissionError, match=".npy"):
        save(best_submission, tmp_path / "sub.npz")


def test_score_cli_exit_codes(best_submission, tmp_path):
    good, bad = tmp_path / "good.npy", tmp_path / "bad.npy"
    np.save(good, best_submission)
    np.save(bad, _break(best_submission, "21 joints"))
    assert score.main(["--strict", str(good)]) == 0
    assert score.main([str(bad)]) == 1
