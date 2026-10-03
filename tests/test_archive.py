"""linkopt.archive keeps the designs worth submitting, and best.npy can only improve."""

import copy
import itertools
import json
import shutil

import numpy as np
import pytest
from conftest import BEST_PATH

import merge
from linkopt import archive
from linkopt.archive import (
    _choose,
    contributions,
    hypervolume,
    non_dominated,
    select,
    trim,
    update_best,
)
from linkopt.config import preset
from linkopt.ga import target_curve
from linkopt.problem import evaluate
from linkopt.refine import refine
from linkopt.submission import load, validate
from LINKS.CP import REFERENCE_POINTS

K2 = 1  # Kangaroo 2: limits (distance 1.2, material 10)


def random_front(seed, n):
    """n non-dominated [distance, material] rows inside Kangaroo 2's box."""
    rng = np.random.default_rng(seed)
    material = np.sort(rng.uniform(5, 10, n))
    distance = np.sort(rng.uniform(0.4, 1.2, n))[::-1]
    return np.column_stack([distance, material])


# --- The hypervolume geometry ---------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_contribution_is_the_area_lost_when_removed(seed):
    F = random_front(seed, 15)
    total = hypervolume(F, K2)
    for i, area in enumerate(contributions(F, K2)):
        without = hypervolume(np.delete(F, i, axis=0), K2)
        assert area == pytest.approx(total - without, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_non_dominated_matches_brute_force(seed):
    F = np.random.default_rng(seed).uniform([0.4, 5], [1.2, 10], (60, 2))
    beaten = [
        any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)))
        for i in range(len(F))
    ]
    assert set(non_dominated(F)) == {i for i, b in enumerate(beaten) if not b}


def test_trim_matches_the_best_possible_subset():
    """12 designs trimmed to 8: compare with EVERY possible 8-design subset (495)."""
    ratios = []
    for seed in range(20):
        F = random_front(seed, 12)
        kept = hypervolume(F[trim(F, K2, 8)], K2)
        best = max(
            hypervolume(F[list(c)], K2) for c in itertools.combinations(range(12), 8)
        )
        ratios.append(kept / best)
    ratios = np.array(ratios)
    assert ratios.min() > 0.999  # never more than 0.1% below the best possible
    assert (ratios > 1 - 1e-12).mean() >= 0.9  # and exactly the best in most cases


def test_trim_loses_far_less_than_random_trimming():
    F = random_front(0, 300)
    total = hypervolume(F, K2)
    loss_ours = total - hypervolume(F[trim(F, K2, 100)], K2)
    rng = np.random.default_rng(0)
    loss_random = np.mean(
        [
            total - hypervolume(F[rng.choice(300, 100, replace=False)], K2)
            for _ in range(20)
        ]
    )
    assert loss_ours < loss_random / 5


def test_trim_keeps_everything_under_the_cap():
    F = random_front(1, 20)
    assert list(trim(F, K2, 1000)) == list(range(20))


# --- Selecting designs for one kangaroo -------------------------------------------


@pytest.fixture(scope="module")
def k3_designs(_best_submission):
    return list(_best_submission["Problem 3"])  # 26 valid, non-dominated designs


def test_under_the_cap_selection_keeps_all_the_hypervolume(k3_designs):
    sel = select(k3_designs, 2)
    d, m = evaluate(k3_designs, target_curve(2))
    assert sel.hypervolume == pytest.approx(
        hypervolume(np.column_stack([d, m]), 2), rel=1e-6
    )


def test_duplicates_are_dropped_even_with_links_reordered(k3_designs):
    shuffled = copy.deepcopy(k3_designs[0])
    shuffled["edges"] = shuffled["edges"][::-1, ::-1].copy()  # same links, other order
    sel = select(k3_designs + [copy.deepcopy(k3_designs[0]), shuffled], 2)
    assert sel.counts["duplicate"] == 2
    assert len(sel.designs) == len(select(k3_designs, 2).designs)


def test_broken_entries_are_dropped_and_the_rest_kept(k3_designs):
    good = copy.deepcopy(k3_designs[0])
    no_motor = {k: v for k, v in good.items() if k != "motor"}
    too_big = dict(good, x0=np.random.rand(21, 2))
    nan_x0 = dict(good, x0=np.full_like(good["x0"], np.nan))
    sel = select(k3_designs + [no_motor, too_big, nan_x0, "not a design", None], 2)
    assert sel.counts["broken"] == 5
    assert sel.hypervolume == pytest.approx(select(k3_designs, 2).hypervolume, rel=1e-6)


def test_starter_style_designs_keep_their_score(k3_designs):
    starter_style = [
        dict(d, target_joint=None, motor=list(d["motor"])) for d in k3_designs
    ]
    assert select(starter_style, 2).hypervolume == pytest.approx(
        select(k3_designs, 2).hypervolume, rel=1e-6
    )


def test_safety_margin_drops_designs_just_inside_the_limits():
    limit_d, limit_m = REFERENCE_POINTS[K2]
    F = np.array(
        [
            [limit_d * (1 - 1e-5), 5.0],  # inside the real limit, but within the margin
            [limit_d * (1 - 1e-3), 5.0],  # comfortably inside
            [limit_d * 0.5, limit_m * 1.01],  # over the material limit
            [np.inf, np.inf],  # couldn't be simulated
        ]
    )
    entries = [
        {
            "x0": np.full((5, 2), i),
            "edges": np.array([[0, 1]]),
            "fixed_joints": np.array([0]),
            "motor": np.array([0, 1]),
            "target_joint": 4,
        }
        for i in range(4)
    ]
    kept, reasons = _choose(F, entries, K2, 1e-4, 1000)
    assert kept == [1] and reasons["outside_limits"] == 3


# --- Updating best.npy ------------------------------------------------------------


@pytest.fixture
def best(tmp_path):
    """A private copy of best.npy + best_score.json, so tests can't touch the real one."""
    subs = tmp_path / "submissions"
    subs.mkdir()
    shutil.copy2(BEST_PATH, subs / "best.npy")
    shutil.copy2(BEST_PATH.with_name("best_score.json"), subs / "best_score.json")
    return subs / "best.npy"


def _bytes(path):
    return path.read_bytes(), path.with_name("best_score.json").read_bytes()


def test_nothing_new_changes_nothing(best):
    before = _bytes(best)
    result = update_best({}, "test", best_path=best)
    assert not result.improved and _bytes(best) == before


def test_garbage_cannot_make_the_best_worse(best, k3_designs):
    before = _bytes(best)
    garbage = {
        0: [None, {"x0": np.zeros((3, 2))}],
        1: [
            dict(k3_designs[0], x0=k3_designs[0]["x0"] * 5)
        ],  # valid format, far outside limits
        2: [dict(k3_designs[0], x0=np.random.rand(21, 2))],  # 21 joints
    }
    result = update_best(garbage, "garbage", best_path=best)
    assert not result.improved and _bytes(best) == before
    assert result.new_score >= result.old_score * (1 - 1e-5)


@pytest.fixture(scope="module")
def improving_designs(_best_submission):
    """Refined versions of the baseline's Kangaroo 2 designs: a genuine improvement."""
    r = refine(
        _best_submission["Problem 2"],
        K2,
        preset("quick", grad_steps=10, step_sizes=(4e-4,)),
    )
    assert r.moved()
    return {K2: r.moved()}


def test_a_real_improvement_is_saved_safely(best, improving_designs):
    before = json.loads(best.with_name("best_score.json").read_text())
    entries_before = (
        len(before.get("history", [])) or 1
    )  # old records: 1 implicit entry
    result = update_best(improving_designs, "test improvement", best_path=best)
    assert result.improved and result.new_score > result.old_score

    # the previous best was backed up, byte for byte
    assert (
        result.backup is not None
        and result.backup.read_bytes() == BEST_PATH.read_bytes()
    )

    # the new best meets every submission requirement, re-checked independently
    new = load(best)
    validate(new, strict=True)
    for t in range(3):
        mechs = new[f"Problem {t + 1}"]
        assert 0 < len(mechs) <= 1000
        d, m = evaluate(mechs, target_curve(t))
        assert (d < REFERENCE_POINTS[t][0]).all() and (m < REFERENCE_POINTS[t][1]).all()

    # the record matches, and remembers where the designs came from
    record = json.loads(best.with_name("best_score.json").read_text())
    assert record["overall_score"] == pytest.approx(result.new_score)
    assert [h["source"] for h in record["history"]][-1] == "test improvement"
    # exactly one new entry, whatever the real best's history was (it grows with runs)
    assert len(record["history"]) == entries_before + 1


def test_dry_run_writes_nothing(best, improving_designs):
    before = _bytes(best)
    result = update_best(improving_designs, "dry", best_path=best, write=False)
    assert result.new_score > result.old_score  # it would have improved...
    assert not result.improved and _bytes(best) == before  # ...but nothing changed


def test_fresh_is_a_deliberate_reset(best, improving_designs):
    result = update_best(improving_designs, "reset", best_path=best, fresh=True)
    assert result.improved  # written even though Kangaroos 1 and 3 are now empty
    record = json.loads(best.with_name("best_score.json").read_text())
    assert [h["source"] for h in record["history"]] == ["reset"]


def test_merge_cli_dry_run_and_fresh_confirmation(best, monkeypatch, capsys):
    monkeypatch.setattr(archive, "BEST_PATH", best)
    monkeypatch.setattr(merge, "BEST_PATH", best)
    before = _bytes(best)
    monkeypatch.setattr(
        merge, "update_best", lambda *a, **k: update_best(*a, best_path=best, **k)
    )
    assert merge.main(["--dry-run", str(best)]) == 0
    assert "dry run: nothing changed" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert merge.main(["--fresh", str(best)]) == 1  # not confirmed: cancelled
    assert _bytes(best) == before
