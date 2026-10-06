"""linkopt.refine runs the advanced notebook's gradient loop, keeps every refined
design inside the limits, and its designs make it into a submission."""

import json

import numpy as np
import pytest
from conftest import ROOT
from pymoo.indicators.hv import HV

from linkopt.config import preset
from linkopt.ga import target_curve
from linkopt.problem import LIMIT_MARGIN, batch_size_for, evaluate, safe_limits
from linkopt.refine import _descend, gradient_tools, refine
from linkopt.submission import build_submission, load, save
from LINKS.CP import REFERENCE_POINTS

ADVANCED_NB = ROOT / "Fall_26_CP1_Advanced_Starter_Notebook.ipynb"
TARGET = 2  # Kangaroo 3
STEPS = 20


@pytest.fixture(scope="session")
def designs(_best_submission):
    """26 feasible 12-joint designs for Kangaroo 3 (the committed baseline)."""
    return _best_submission["Problem 3"]


@pytest.fixture(scope="session")
def refined(designs):
    return refine(designs, TARGET, preset("quick", grad_steps=STEPS))


def test_descend_is_the_notebooks_gradient_loop(designs):
    """Run the advanced notebook's own loop cell (from the .ipynb) next to ours on
    the same batch: identical positions, bit for bit.

    The notebook's cell is run for STEPS + 1 steps; its `x_last` (positions after its
    final check, before its final unchecked move) is what ours returns after STEPS
    moves plus a final check. margin=0 gives the notebook's `<= limit` rule.
    """
    batch = list(designs) + [designs[0]] * (batch_size_for(len(designs)) - len(designs))
    cells = ["".join(c["source"]) for c in json.loads(ADVANCED_NB.read_text())["cells"]]
    loop_cell = next(c for c in cells if "done_optimizing = np.zeros" in c)
    namespace = {
        "np": np,
        "trange": lambda n: range(STEPS + 1),  # the cell's `for step in trange(1000)`
        "x0s": [np.array(m["x0"], dtype=float) for m in batch],
        "edges": [m["edges"] for m in batch],
        "fixed_joints": [m["fixed_joints"] for m in batch],
        "motors": [m["motor"] for m in batch],
        "target_idxs": [m["target_joint"] for m in batch],
        "target_curve": target_curve(TARGET),
        "reference_point": REFERENCE_POINTS[TARGET],
        "differentiable_optimization_tools": gradient_tools(),
    }
    exec(loop_cell, namespace)  # noqa: S102 -- running the notebook's own code is the point

    ours, _, _, _ = _descend(
        batch,
        target_curve(TARGET),
        safe_limits(REFERENCE_POINTS[TARGET], 0),
        STEPS,
        4e-4,
    )
    for a, b in zip(ours, namespace["x_last"]):
        np.testing.assert_array_equal(a, b)


def test_refined_designs_stay_inside_the_limits(refined):
    limits = safe_limits(REFERENCE_POINTS[TARGET], LIMIT_MARGIN)
    assert (refined.F_after <= limits).all()
    # and strictly inside the real limits, re-scored independently
    d, m = evaluate(refined.designs, target_curve(TARGET))
    assert (d < REFERENCE_POINTS[TARGET][0]).all() and (
        m < REFERENCE_POINTS[TARGET][1]
    ).all()


def test_refinement_only_moves_joints(designs, refined):
    for before, after, n in zip(designs, refined.designs, refined.steps):
        for key in ("edges", "fixed_joints", "motor", "target_joint"):
            np.testing.assert_array_equal(before[key], after[key])
        assert (n > 0) == (not np.array_equal(before["x0"], after["x0"]))


def test_refinement_never_makes_distance_worse(refined):
    moved = refined.steps > 0
    assert moved.any()
    # Each design comes back at its best position: never a higher distance (up to
    # float32 noise between the gradient scorer and the grader's scorer) ...
    assert (refined.F_after[:, 0] <= refined.F_before[:, 0] * (1 + 1e-5)).all()
    # ... and most of the ones that moved are better by a real margin (some designs
    # are already near-optimal and only move by float32 noise, ~1e-7).
    better = refined.F_after[moved, 0] < refined.F_before[moved, 0] * (1 - 1e-4)
    assert better.mean() >= 0.5
    hv = HV(REFERENCE_POINTS[TARGET])
    both = np.vstack([refined.F_before, refined.F_after[moved]])
    assert hv(both) > hv(refined.F_before)


def test_refined_designs_end_up_in_the_submission(tmp_path, designs, refined):
    both = list(designs) + refined.moved()
    scores = save(build_submission({TARGET: both}), tmp_path / "sub.npy")
    saved = load(tmp_path / "sub.npy")["Problem 3"]
    assert len(saved) == len(both)
    for d in refined.moved():
        assert any(np.array_equal(d["x0"], s["x0"]) for s in saved)
    original = save(build_submission({TARGET: designs}), tmp_path / "orig.npy")
    assert (
        scores["Score Breakdown"]["Problem 3"]
        > original["Score Breakdown"]["Problem 3"]
    )


def test_zero_steps_changes_nothing(designs):
    r = refine(designs[:3], TARGET, preset("quick", grad_steps=0))
    assert (r.steps == 0).all() and r.moved() == []
    for a, b in zip(designs[:3], r.designs):
        np.testing.assert_array_equal(a["x0"], b["x0"])


def test_safe_limits_are_strictly_inside():
    for ref in REFERENCE_POINTS:
        assert (safe_limits(ref) < ref).all()
        np.testing.assert_array_equal(safe_limits(ref, 0), ref)


def test_several_step_sizes_help_sensitive_designs(_best_submission):
    """The baseline's Kangaroo 1 designs overshoot or jam at the notebook's step size
    (4e-4) and only improve with smaller ones. Trying several sizes keeps, for each
    design, the best of them: never worse than any single size, and more designs
    improve."""
    k1 = _best_submission["Problem 1"]
    one = refine(k1, 0, preset("quick", grad_steps=30, step_sizes=(4e-4,)))
    several = refine(k1, 0, preset("quick", grad_steps=30, step_sizes=(4e-4, 3e-5)))
    assert (several.F_after[:, 0] <= one.F_after[:, 0] * (1 + 1e-5)).all()
    assert (several.steps > 0).sum() > (one.steps > 0).sum()
    assert set(several.step_size[several.steps > 0]) <= {4e-4, 3e-5}


def test_never_worse_even_when_the_step_overshoots(designs):
    """With only the notebook's step size (4e-4), most of these designs overshoot:
    their final positions are worse than where they started (72% in our check).
    Each design must still come back at its best position, never worse."""
    r = refine(designs, TARGET, preset("quick", grad_steps=STEPS, step_sizes=(4e-4,)))
    assert (r.F_after[:, 0] <= r.F_before[:, 0] * (1 + 1e-5)).all()


# --- Smarter descent rules (cfg.refine_method) -----------------------------------


@pytest.fixture(scope="session")
def k2_designs(_best_submission):
    """A few Kangaroo 2 designs with material above 2.4 (the accurate, heavy ones)."""
    mechs = list(_best_submission["Problem 2"])
    _, material = evaluate(mechs, target_curve(1))
    heavy = [m for m, mat in zip(mechs, material) if mat > 2.4]
    return heavy[:8]


def _refined(designs, target, method, steps):
    return refine(
        designs, target, preset("quick", refine_method=method, grad_steps=steps)
    )


def test_plain_is_still_the_default_and_unchanged(designs):
    """rule: new behavior is opt-in. Asking for "plain" must be bit-identical to the
    default, which is the behavior every earlier run had."""
    assert preset("quick").refine_method == "plain"
    default = refine(designs[:6], TARGET, preset("quick", grad_steps=STEPS))
    explicit = _refined(designs[:6], TARGET, "plain", STEPS)
    np.testing.assert_array_equal(default.F_after, explicit.F_after)
    np.testing.assert_array_equal(default.steps, explicit.steps)
    for a, b in zip(default.designs, explicit.designs):
        np.testing.assert_array_equal(a["x0"], b["x0"])


@pytest.mark.parametrize("method", ["adam", "basin"])
def test_every_method_keeps_refines_inside_the_limits_and_never_worse(
    k2_designs, method
):
    """The promise refine() makes, for the new methods too: each design comes back
    inside the limits and never with a higher distance than it went in with."""
    result = _refined(k2_designs, 1, method, 10)
    limits = safe_limits(REFERENCE_POINTS[1], LIMIT_MARGIN)
    assert (result.F_after <= limits).all()
    assert (result.F_after[:, 0] <= result.F_before[:, 0]).all()
    # Material is never optimized, only distance, so it may drift either way but the
    # design must stay legal; and a design that didn't move is reported as unmoved.
    for i, moved in enumerate(result.steps > 0):
        same = np.array_equal(
            np.asarray(result.designs[i]["x0"]), np.asarray(k2_designs[i]["x0"])
        )
        assert moved != same, f"design {i}: steps and positions disagree"
    assert len(result.moved()) == int((result.steps > 0).sum())


def test_basin_is_reproducible(k2_designs):
    """Fixed RNG seed: same designs in, same positions out."""
    a = _refined(k2_designs, 1, "basin", 6)
    b = _refined(k2_designs, 1, "basin", 6)
    np.testing.assert_array_equal(a.F_after, b.F_after)
    for m, m2 in zip(a.designs, b.designs):
        np.testing.assert_array_equal(m["x0"], m2["x0"])


def test_adam_moves_differently_from_plain(k2_designs):
    """Sanity check that the Adam branch is actually a different walk (otherwise the
    experiment would be comparing a method against itself)."""
    plain = _refined(k2_designs, 1, "plain", 10)
    adam = _refined(k2_designs, 1, "adam", 10)
    assert not np.array_equal(plain.F_after, adam.F_after)


def test_gradient_clipping_keeps_direction_and_caps_length():
    from linkopt.refine import GRAD_CLIP, _clipped

    small = np.array([[0.1, 0.0], [0.0, 0.2]])
    np.testing.assert_array_equal(_clipped(small), small)  # untouched

    big = np.array([[30.0, 40.0]])  # length 50
    out = _clipped(big)
    assert np.isclose(np.sqrt((out**2).sum()), GRAD_CLIP)
    assert np.allclose(out / np.linalg.norm(out), big / np.linalg.norm(big))
