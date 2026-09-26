"""Build, check and save CP1 submissions.

This is the only module that creates or writes submission files, so the format
rules live in one place. The format is the one in the starter notebook's
"Submission Format" section:

    {'Problem 1': [mech, ...], 'Problem 2': [...], 'Problem 3': [...]}

    mech = {'x0':           (N, 2) float array,
            'edges':        (E, 2) int array,
            'fixed_joints': (F,)   int array,
            'motor':        (2,)   int array,
            'target_joint': int}

'Problem i+1' is scored against target_curves[i] by LINKS.CP.evaluate_submission.

Two levels of checking:
    strict=False  what the grader accepts. The starter notebook's own output passes
                  (it stores motor as a list and target_joint as None).
    strict=True   the exact types above, with target_joint always set. Everything
                  this framework writes is strict.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from LINKS.CP import MAX_JOINTS, N_PROBLEMS, evaluate_submission, make_empty_submission

MAX_PER_PROBLEM = 1000  # evaluate_submission ignores everything past the first 1000
# The grader silently skips entries missing any of these.
REQUIRED_KEYS = ("x0", "edges", "fixed_joints", "motor")
TARGET_CURVES_PATH = (
    Path(__file__).resolve().parent.parent / "kangaroo_target_curves.npy"
)

PathLike = str | Path


class SubmissionError(ValueError):
    """The grader would reject, silently skip, or mis-score part of the submission."""


def problem_key(target_index: int) -> str:
    """target_curves[i] is scored as 'Problem i+1'."""
    if not 0 <= target_index < N_PROBLEMS:
        raise SubmissionError(
            f"target index must be 0..{N_PROBLEMS - 1}, got {target_index}"
        )
    return f"Problem {target_index + 1}"


def to_entry(mech: Mapping) -> dict:
    """Convert one mechanism dict to the exact submission types (e.g. a list motor
    becomes an int array). Raises SubmissionError if it can't be made valid."""
    if mech.get("target_joint") is None:
        raise SubmissionError(
            "target_joint is missing or None; set it to the joint the optimizer scored "
            "(otherwise the grader silently uses the last joint in the solve order)"
        )
    missing = [k for k in REQUIRED_KEYS if k not in mech]
    if missing:
        raise SubmissionError(f"mechanism is missing {missing}")

    entry = {
        "x0": np.asarray(mech["x0"], dtype=np.float64),
        "edges": _as_int_array(mech["edges"], "edges"),
        "fixed_joints": _as_int_array(mech["fixed_joints"], "fixed_joints").reshape(-1),
        "motor": _as_int_array(mech["motor"], "motor"),
        "target_joint": _as_int(mech["target_joint"], "target_joint"),
    }
    errors, _ = _check_entry(entry, strict=True)
    if errors:
        raise SubmissionError("; ".join(errors))
    return entry


def build_submission(designs: Mapping[int, Sequence[Mapping]]) -> dict:
    """{target_index: [mech, ...]} -> submission dict. Problems not given stay empty."""
    submission = make_empty_submission()
    for target_index, mechs in designs.items():
        key = problem_key(target_index)
        if len(mechs) > MAX_PER_PROBLEM:
            # Refuse rather than truncate: which 1000 to keep is a scoring decision.
            raise SubmissionError(
                f"{key}: {len(mechs)} mechanisms, the grader only scores the first "
                f"{MAX_PER_PROBLEM}; select the best {MAX_PER_PROBLEM} before building"
            )
        submission[key] = [to_entry(m) for m in mechs]
    return submission


def validate(submission, strict: bool = True) -> list[str]:
    """Check a submission against the format rules.

    Raises SubmissionError listing every error; returns the list of warnings
    (things the grader tolerates but that are probably not intended).
    """
    if not isinstance(submission, dict):
        raise SubmissionError(
            f"submission must be a dict, got {type(submission).__name__}"
        )

    errors: list[str] = []
    warnings: list[str] = []
    expected = [problem_key(i) for i in range(N_PROBLEMS)]

    for key in expected:
        if key not in submission:
            errors.append(f"missing key '{key}' (the grader scores it as 0)")
    unknown = sorted(str(k) for k in submission if k not in expected)
    if unknown:
        warnings.append(f"unrecognized keys {unknown} are ignored by the grader")

    for key in expected:
        mechs = submission.get(key)
        if mechs is None:
            continue
        if not isinstance(mechs, list):
            errors.append(
                f"{key}: must be a list of mechanisms, got {type(mechs).__name__}"
            )
            continue
        if len(mechs) == 0:
            warnings.append(f"{key}: no mechanisms (scores 0)")
        if len(mechs) > MAX_PER_PROBLEM:
            errors.append(
                f"{key}: {len(mechs)} mechanisms, only the first {MAX_PER_PROBLEM} are scored"
            )

        # Group identical messages so 1000 copies of one problem read as one line.
        entry_errors, entry_warnings = defaultdict(list), defaultdict(list)
        for i, mech in enumerate(mechs):
            errs, warns = _check_entry(mech, strict)
            for msg in errs:
                entry_errors[msg].append(i)
            for msg in warns:
                entry_warnings[msg].append(i)
        errors += [
            _grouped(key, idx, len(mechs), msg) for msg, idx in entry_errors.items()
        ]
        warnings += [
            _grouped(key, idx, len(mechs), msg) for msg, idx in entry_warnings.items()
        ]

    if errors:
        raise SubmissionError("invalid submission:\n  - " + "\n  - ".join(errors))
    return warnings


def load(path: PathLike) -> dict:
    """Load a submission file exactly the way the grader does."""
    try:
        submission = np.load(path, allow_pickle=True).item()
    except ValueError as e:
        raise SubmissionError(f"{path} does not contain a pickled dict ({e})") from e
    if not isinstance(submission, dict):
        raise SubmissionError(
            f"{path} contains a {type(submission).__name__}, not a dict"
        )
    return submission


def save(
    submission: dict, path: PathLike, target_curves: PathLike = TARGET_CURVES_PATH
) -> dict:
    """Validate (strict), write, reload, re-validate, and score with the course grader.

    Returns evaluate_submission's result for the file on disk.
    """
    path = Path(path)
    if path.suffix != ".npy":
        raise SubmissionError(
            f"{path}: must end in .npy (np.save would silently append it)"
        )

    validate(submission, strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, submission, allow_pickle=True)

    validate(load(path), strict=True)
    return evaluate_submission(str(path), str(target_curves))


def _check_entry(mech, strict: bool):
    """Return (errors, warnings) for one mechanism."""
    if not isinstance(mech, Mapping):
        return [f"entry is a {type(mech).__name__}, not a dict"], []
    missing = [k for k in REQUIRED_KEYS if k not in mech]
    if missing:
        return [f"missing {missing} (the grader silently skips such entries)"], []

    errors: list[str] = []
    warnings: list[str] = []
    type_issues = errors if strict else warnings

    try:
        x0 = np.asarray(mech["x0"], dtype=np.float64)
    except (TypeError, ValueError):
        return ["x0 is not numeric"], []
    if x0.ndim != 2 or x0.shape[1] != 2 or x0.shape[0] == 0:
        return [f"x0 must have shape (N, 2), got {x0.shape}"], []
    n = x0.shape[0]
    if n > MAX_JOINTS:
        errors.append(f"{n} joints, the limit is {MAX_JOINTS} (the grader skips it)")
    if not np.all(np.isfinite(x0)):
        errors.append("x0 contains NaN or inf")

    shapes = {"edges": "(E, 2)", "fixed_joints": "(F,)", "motor": "(2,)"}
    for name, shape in shapes.items():
        value = mech[name]
        try:
            arr = _as_int_array(value, name)
        except SubmissionError as e:
            errors.append(str(e))
            continue
        if name == "edges":
            bad_shape = arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0
        elif name == "fixed_joints":
            bad_shape = arr.ndim != 1
        else:
            bad_shape = arr.shape != (2,)
        if bad_shape:
            errors.append(f"{name} must have shape {shape}, got {arr.shape}")
            continue
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            errors.append(f"{name} refers to joints outside 0..{n - 1}")
        if not isinstance(value, np.ndarray):
            type_issues.append(
                f"{name} is a {type(value).__name__}, the format asks for np.ndarray"
            )
        elif value.dtype.kind not in "iu":
            type_issues.append(
                f"{name} has dtype {value.dtype}, expected an integer dtype"
            )

    if isinstance(mech["x0"], np.ndarray) and mech["x0"].dtype.kind != "f":
        type_issues.append(f"x0 has dtype {mech['x0'].dtype}, expected float")
    elif not isinstance(mech["x0"], np.ndarray):
        type_issues.append(
            f"x0 is a {type(mech['x0']).__name__}, the format asks for np.ndarray"
        )

    target = mech.get("target_joint")
    if target is None:
        type_issues.append(
            "target_joint is missing or None, so the grader uses the last joint in the solve order"
        )
    else:
        try:
            t = _as_int(target, "target_joint")
        except SubmissionError as e:
            errors.append(str(e))
        else:
            if not 0 <= t < n:
                errors.append(f"target_joint {t} is outside 0..{n - 1}")
            if strict and not isinstance(target, int):
                errors.append(
                    f"target_joint is a {type(target).__name__}, expected a Python int"
                )

    return errors, warnings


def _as_int_array(value, name: str) -> np.ndarray:
    """Convert to an int array, refusing values that aren't whole numbers."""
    try:
        arr = np.asarray(value)
        as_int = arr.astype(np.int64)
    except (TypeError, ValueError) as e:
        raise SubmissionError(f"{name} is not an integer array") from e
    if arr.dtype.kind == "b" or not np.array_equal(arr, as_int):
        raise SubmissionError(f"{name} must contain whole-number joint indices")
    return as_int


def _as_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise SubmissionError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _grouped(key: str, indices: list[int], total: int, msg: str) -> str:
    if len(indices) == 1:
        return f"{key} entry {indices[0]}: {msg}"
    shown = ", ".join(map(str, indices[:5])) + (", ..." if len(indices) > 5 else "")
    return f"{key} entries [{shown}] ({len(indices)} of {total}): {msg}"
