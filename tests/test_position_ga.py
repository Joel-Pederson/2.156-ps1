"""linkopt.position_ga searches only one design's joint positions: it never changes the
shape it was given, keeps every coordinate in [0, 1], and its result refines like a
normal GA's."""

import numpy as np
import pytest

from linkopt.config import preset
from linkopt.ga import target_curve
from linkopt.position_ga import (
    rank_shapes,
    run_position_ga,
    shape_key,
    start_population,
    topology,
)
from linkopt.problem import evaluate
from linkopt.refine import refine

TARGET = 1  # Kangaroo 2, the one this experiment is about
POP, GENS = 8, 2  # tiny: these tests check behavior, not optimization quality


@pytest.fixture(scope="session")
def k2(_best_submission):
    """The committed baseline's Kangaroo 2 front (79 designs, 9 distinct shapes)."""
    return _best_submission["Problem 2"]


@pytest.fixture(scope="session")
def base(k2):
    """Its lowest-distance design, the one a position_ga run starts with."""
    return k2[rank_shapes(k2, TARGET, 1)[0]]


def _frozen(design):
    """Everything a positions-only GA must leave exactly as it found it."""
    return (
        np.asarray(design["edges"]),
        np.asarray(design["fixed_joints"]),
        np.asarray(design["motor"]),
        int(design["target_joint"]),
    )


def _assert_same_shape(design, base):
    got, want = _frozen(design), _frozen(base)
    for a, b in zip(got[:3], want[:3]):
        np.testing.assert_array_equal(a, b)
    assert got[3] == want[3]


# --- Picking which shapes to fine-tune ---------------------------------------------


def test_rank_shapes_orders_by_lowest_distance(k2):
    chosen = rank_shapes(k2, TARGET, 10)
    assert len(chosen) == 10
    distance, _ = evaluate([k2[i] for i in chosen], target_curve(TARGET))
    assert list(distance) == sorted(distance)


def test_dedup_returns_one_design_per_distinct_shape(k2):
    """The reason dedup exists: Kangaroo 2's 15 lowest-distance designs are only 4
    distinct shapes, so without it most of the budget re-searches the same one."""
    assert len({topology(k2[i]) for i in rank_shapes(k2, TARGET, 15)}) == 4

    deduped = rank_shapes(k2, TARGET, 15, dedup=True)
    shapes = [topology(k2[i]) for i in deduped]
    assert len(set(shapes)) == len(shapes)  # every one different
    assert len(deduped) == len({topology(d) for d in k2})  # and all of them: 9
    # Each kept design is the lowest-distance one of its shape.
    distance, _ = evaluate(list(k2), target_curve(TARGET))
    for i in deduped:
        same = [j for j, d in enumerate(k2) if topology(d) == topology(k2[i])]
        assert distance[i] == min(distance[j] for j in same)


def test_rank_shapes_handles_an_empty_front():
    assert rank_shapes([], TARGET, 5) == []


# --- The starting population ------------------------------------------------------


def test_start_population_begins_with_the_design_itself(base):
    start = start_population(base["x0"], POP, seed=0)
    assert start.shape == (POP, 2 * len(base["x0"]))
    # Row 0 unchanged, so the GA starts from a design already known to be good.
    np.testing.assert_array_equal(
        start[0], np.asarray(base["x0"], dtype=float).flatten()
    )
    assert (start >= 0).all() and (start <= 1).all()
    assert not np.array_equal(start[1], start[0])  # the rest really are nudged


def test_start_population_is_reproducible_and_seed_dependent(base):
    same = start_population(base["x0"], POP, seed=0)
    np.testing.assert_array_equal(same, start_population(base["x0"], POP, seed=0))
    assert not np.array_equal(same, start_population(base["x0"], POP, seed=1))


# --- The GA itself ----------------------------------------------------------------


def test_the_shape_is_held_fixed_and_positions_stay_in_bounds(base):
    """The whole point of this GA: only x0 may change."""
    cfg = preset("quick", pop_size=POP, n_gen=GENS)
    result = run_position_ga(TARGET, base, 0, cfg, expect_shape=shape_key(base))
    assert result.designs, "the starting population contains the base design itself"
    assert result.n_joints == len(base["x0"])
    for design in result.designs:
        _assert_same_shape(design, base)
        x = np.asarray(design["x0"])
        assert x.shape == np.asarray(base["x0"]).shape
        assert (x >= 0).all() and (x <= 1).all()


def test_refining_the_front_also_keeps_the_shape(base):
    """grad_steps > 0 on purpose: with 0 steps refine moves nothing and this handoff
    would go untested. position_ga is the first caller whose designs must come back
    with the same shape for a before/after comparison to mean anything."""
    cfg = preset("quick", pop_size=POP, n_gen=GENS, grad_steps=2)
    result = run_position_ga(TARGET, base, 0, cfg, expect_shape=shape_key(base))
    refined = refine(result.designs, TARGET, cfg)
    for design in refined.designs:
        _assert_same_shape(design, base)
    assert (refined.F_after[:, 0] <= refined.F_before[:, 0]).all()


def test_the_base_design_is_not_modified(base):
    """The GA copies what it freezes; a worker reuses best.npy's list for other jobs."""
    before = {k: np.array(v, copy=True) for k, v in base.items()}
    run_position_ga(
        TARGET, base, 0, preset("quick", pop_size=POP, n_gen=GENS), shape_key(base)
    )
    for key, value in before.items():
        np.testing.assert_array_equal(np.asarray(base[key]), value)


def test_a_shape_mismatch_is_an_error_not_a_wrong_answer(k2, base):
    """A job carries an index into best.npy. If that list ever changed order the job
    would quietly fine-tune a different shape and still look plausible."""
    cfg = preset("quick", pop_size=POP, n_gen=GENS)
    with pytest.raises(ValueError, match="shape mismatch"):
        run_position_ga(TARGET, base, 0, cfg, expect_shape="not-the-same-shape")

    # The check has to survive a swap for a design of the SAME SIZE: six of Kangaroo
    # 2's nine shapes have 6 joints, so comparing joint counts would miss it.
    others = [
        d
        for d in k2
        if len(d["x0"]) == len(base["x0"]) and topology(d) != topology(base)
    ]
    assert others, "expected another 6-joint shape in the Kangaroo 2 front"
    with pytest.raises(ValueError, match="shape mismatch"):
        run_position_ga(TARGET, others[0], 0, cfg, expect_shape=shape_key(base))


def test_shape_key_tracks_the_shape_and_not_the_positions(base):
    moved = {**base, "x0": np.clip(np.asarray(base["x0"]) + 0.05, 0, 1)}
    assert shape_key(moved) == shape_key(base)  # same shape, different positions
    relinked = {**base, "target_joint": (int(base["target_joint"]) + 1) % 2}
    assert shape_key(relinked) != shape_key(base)
