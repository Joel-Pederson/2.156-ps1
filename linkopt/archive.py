"""Pool designs from many runs into the best possible submission, and keep
submissions/best.npy improving (never getting worse).

For each kangaroo, `select` turns a pile of designs into the ones worth submitting:
    1. Drop broken entries (missing fields, more than 20 joints, NaN positions...).
    2. Re-score every design with the grader's scorer (don't trust old numbers).
    3. Drop designs outside the limits (with the same safety margin as refine).
    4. Drop duplicates (same links, fixed joints, motor, target joint, and the same
       positions to 9 decimal places).
    5. Keep only non-dominated designs: a design beaten on both distance and
       material adds no area, so no score.
    6. If more than 1000 remain, trim to 1000 while losing as little hypervolume as
       possible: repeatedly remove the design that alone covers the least area.

`update_best` pools new designs WITH the current best.npy, runs `select`, scores
the result with the course grader, and only replaces best.npy if the score went
up. Because the current best's designs are always in the pool, nothing already in
it can be lost. Writes are locked (one writer at a time) and atomic (a temporary
file is swapped in), and the previous best.npy is backed up to runs/best_backups/.
"""

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from linkopt.ga import target_curve
from linkopt.problem import LIMIT_MARGIN, evaluate, safe_limits
from linkopt.submission import (
    MAX_PER_PROBLEM,
    TARGET_CURVES_PATH,
    build_submission,
    fill_default_target_joints,
    load,
    problem_key,
    save,
    to_entry,
)
from LINKS.CP import N_PROBLEMS, REFERENCE_POINTS, evaluate_submission

ROOT = Path(__file__).resolve().parent.parent
BEST_PATH = ROOT / "submissions" / "best.npy"
RUNS_DIR = ROOT / "runs"  # git-ignored: backups, the lock, temporary files

# best.npy is only replaced if the score rises by more than this (relative). The
# grader computes in float32, so re-scoring the same designs in a different order
# can shift the score by ~1e-6; that noise must not count as an improvement.
MIN_IMPROVEMENT = 1e-5


# --- Selection: one kangaroo ------------------------------------------------------


@dataclass
class Selection:
    """What `select` kept, and why the rest was dropped (counts per reason)."""

    target: int  # which kangaroo (0 = Kangaroo 1)
    designs: list  # the designs worth submitting, as submission entries
    F: np.ndarray  # [distance, material] of each kept design, same order
    counts: dict = field(default_factory=dict)  # how many were dropped, per reason

    @property
    def hypervolume(self) -> float:
        return hypervolume(self.F, self.target)


def select(designs, target, margin=LIMIT_MARGIN, max_designs=MAX_PER_PROBLEM):
    """The designs worth submitting for kangaroo `target` (steps 1-6 above)."""
    counts = {"in": len(designs)}

    # 1. Broken entries are dropped. Starter-notebook style designs (target_joint
    #    None) are converted first, keeping the joint the grader would use.
    entries = [e for e in map(_entry_or_none, designs) if e is not None]
    counts["broken"] = len(designs) - len(entries)

    # 2. Re-score with the grader's scorer (unsimulatable designs get inf).
    F = np.column_stack(evaluate(entries, target_curve(target))) if entries else None

    # 3-6. Choose which to keep.
    keep, reasons = _choose(F, entries, target, margin, max_designs)
    counts.update(reasons)
    counts["kept"] = len(keep)
    return Selection(
        target=target,
        designs=[entries[i] for i in keep],
        F=F[keep] if len(keep) else np.empty((0, 2)),
        counts=counts,
    )


def _choose(F, entries, target, margin, max_designs):
    """Indices of the designs to keep (steps 3-6), plus drop counts per reason."""
    reasons = {"outside_limits": 0, "duplicate": 0, "dominated": 0, "trimmed": 0}
    if F is None or len(F) == 0:
        return [], reasons

    # 3. Inside the limits, by the safety margin.
    limits = safe_limits(REFERENCE_POINTS[target], margin)
    idx = np.where((F <= limits).all(axis=1))[0]
    reasons["outside_limits"] = len(F) - len(idx)

    # 4. Duplicates: keep the first of each identical mechanism.
    seen, unique = set(), []
    for i in idx:
        key = _design_key(entries[i])
        if key not in seen:
            seen.add(key)
            unique.append(i)
    reasons["duplicate"] = len(idx) - len(unique)

    # 5. Non-dominated only.
    front = [unique[j] for j in non_dominated(F[unique])]
    reasons["dominated"] = len(unique) - len(front)

    # 6. At most max_designs, losing as little hypervolume as possible.
    kept = [front[j] for j in trim(F[front], target, max_designs)]
    reasons["trimmed"] = len(front) - len(kept)
    return kept, reasons


def _entry_or_none(mech):
    """The design as a strict submission entry, or None if it's broken."""
    try:
        if mech.get("target_joint") is None:
            mech = fill_default_target_joints([mech])[0]
        return to_entry(mech)
    except Exception:  # noqa: BLE001 -- any broken entry is simply dropped
        return None


def _design_key(entry) -> tuple:
    """Identical mechanisms get identical keys (link order and direction ignored)."""
    edges = np.sort(np.asarray(entry["edges"]), axis=1)
    edges = edges[np.lexsort(edges.T[::-1])]
    return (
        np.round(np.asarray(entry["x0"], dtype=float), 9).tobytes(),
        edges.tobytes(),
        np.sort(np.asarray(entry["fixed_joints"])).tobytes(),
        np.asarray(entry["motor"]).tobytes(),
        int(entry["target_joint"]),
    )


# --- The hypervolume geometry (two objectives) ------------------------------------


def hypervolume(F, target) -> float:
    """The grader's score for these [distance, material] rows on kangaroo `target`."""
    return float(HV(REFERENCE_POINTS[target])(np.asarray(F))) if len(F) else 0.0


def non_dominated(F) -> np.ndarray:
    """Indices of the rows of F (n x 2, both minimized) that no other row beats.

    A row with exactly the same scores as an earlier one counts as dominated: it
    would add no area.
    """
    F = np.asarray(F)
    order = np.lexsort((F[:, 1], F[:, 0]))  # by distance, then material
    keep, best_material = [], np.inf
    for i in order:
        if F[i, 1] < best_material:  # less material than every closer-fitting design
            keep.append(i)
            best_material = F[i, 1]
    return np.array(sorted(keep), dtype=int)


def contributions(F, target) -> np.ndarray:
    """For non-dominated rows of F: the area each one ALONE covers.

    Sorted by material (left to right on the plot), the front is a staircase.
    Design i covers a rectangle only it covers: from its own material to its right
    neighbor's, and from its own distance up to its left neighbor's (the limits
    stand in for the missing neighbors at the ends). Removing design i loses
    exactly that rectangle.
    """
    F = np.asarray(F, dtype=float)
    d_ref, m_ref = REFERENCE_POINTS[target]
    order = np.argsort(F[:, 1])
    d, m = F[order, 0], F[order, 1]
    right_material = np.append(m[1:], m_ref)
    left_distance = np.insert(d[:-1], 0, d_ref)
    area = np.empty(len(F))
    area[order] = (right_material - m) * (left_distance - d)
    return area


def trim(F, target, max_designs) -> np.ndarray:
    """Indices of at most `max_designs` rows of the non-dominated F to keep.

    Repeatedly removes the design whose own area (see `contributions`) is
    smallest, then recomputes its neighbors' areas, until max_designs remain. The
    same idea as the SMS-EMOA optimizer's selection.
    """
    F = np.asarray(F, dtype=float)
    keep = np.arange(len(F))
    while len(keep) > max_designs:
        worst = np.argmin(contributions(F[keep], target))
        keep = np.delete(keep, worst)
    return keep


# --- Updating best.npy safely ------------------------------------------------------


@dataclass
class UpdateResult:
    improved: bool  # was best.npy replaced?
    old_score: float  # overall score before
    new_score: float  # overall score of the pooled candidate
    scores: dict  # evaluate_submission's full result for the candidate
    selections: dict  # target -> Selection (what was kept and why)
    backup: Path | None  # where the previous best.npy was copied, if replaced


def update_best(
    new_designs,
    source,
    best_path=BEST_PATH,
    fresh=False,
    write=True,
):
    """Pool `new_designs` ({target: [designs]}) with the current best, select, score
    with the grader, and replace best.npy only if the overall score went up.

    fresh=True ignores the current best's designs (a deliberate reset; merge.py
    asks you to type RESET first). write=False computes everything but writes
    nothing (merge.py --dry-run).
    """
    best_path = Path(best_path)
    record_path = best_path.with_name("best_score.json")
    runs_dir = best_path.parent.parent / "runs"  # RUNS_DIR for the real best.npy
    runs_dir.mkdir(exist_ok=True)

    with _locked(runs_dir / ".best.lock"):
        current = load(best_path) if best_path.exists() else None
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        # The current best scored now, on this computer, not the recorded score: that
        # may come from another computer (CI's Linux scores the same file ~3e-5
        # differently than a Mac), and the difference would look like an improvement.
        old_score = (
            evaluate_submission(str(best_path), str(TARGET_CURVES_PATH))[
                "Overall Score"
            ]
            if current is not None
            else 0.0
        )

        # Pool: the current best's designs (unless fresh) plus the new ones.
        selections = {}
        for target in range(N_PROBLEMS):
            pool = (
                [] if (fresh or current is None) else list(current[problem_key(target)])
            )
            pool += list(new_designs.get(target, []))
            selections[target] = select(pool, target)

        # Write the candidate to a temporary file and score it with the grader.
        candidate = runs_dir / ".best_candidate.npy"
        submission = build_submission({t: s.designs for t, s in selections.items()})
        scores = save(submission, candidate)
        new_score = scores["Overall Score"]

        improved = write and (fresh or new_score > old_score * (1 + MIN_IMPROVEMENT))
        backup = None
        if improved:
            if best_path.exists():
                backups = runs_dir / "best_backups"
                backups.mkdir(exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # noqa: DTZ005 (local time)
                backup = backups / f"best-{stamp}.npy"
                shutil.copy2(best_path, backup)
            os.replace(candidate, best_path)  # atomic: never a half-written best.npy
            _write_record(record_path, record, scores, source, fresh)
        else:
            candidate.unlink(missing_ok=True)

    return UpdateResult(
        improved=improved,
        old_score=old_score,
        new_score=new_score,
        scores=scores,
        selections=selections,
        backup=backup,
    )


def _write_record(record_path, old_record, scores, source, fresh):
    """best_score.json: the new scores, plus a history of what contributed."""
    today = datetime.now().strftime("%Y-%m-%d %H:%M")  # noqa: DTZ005 (local time)
    history = [] if fresh else list(old_record.get("history", []))
    # A record from before histories existed: its source becomes the first entry
    # (unless this is a reset, which discards the old designs).
    if not fresh and not history and old_record.get("source"):
        history.append(
            {
                "date": old_record.get("date"),
                "source": old_record["source"],
                "overall_score": old_record.get("overall_score"),
            }
        )
    history.append(
        {"date": today, "source": source, "overall_score": scores["Overall Score"]}
    )
    record = {
        "overall_score": scores["Overall Score"],
        "score_breakdown": scores["Score Breakdown"],
        "normalized_score_breakdown": scores["Normalized Score Breakdown"],
        "source": source,
        "date": today,
        "history": history,
    }
    tmp = record_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, record_path)


@contextmanager
def _locked(lock_path):
    """Only one process at a time may update best.npy (waits for the others)."""
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
