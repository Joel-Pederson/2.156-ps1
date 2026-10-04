"""linkopt.problem gives the same answers as the advanced notebook, only faster."""

import json

import numpy as np
import pytest
from conftest import ROOT

from linkopt import problem
from linkopt.problem import (
    MechanismProblem,
    batch_size_for,
    evaluate,
    mixed_variables,
    tools,
)
from linkopt.submission import to_entry
from LINKS.CP import REFERENCE_POINTS
from LINKS.Optimization import MechanismRandomizer

ADVANCED_NB = ROOT / "Fall_26_CP1_Advanced_Starter_Notebook.ipynb"
N = 7
TARGET = 1  # Kangaroo 2, as in the notebook
# LINKS computes in float32 and XLA rounds slightly differently for different batch
# sizes: the same design can differ by ~1e-6 (relative) between a batch of 1 and a
# batch of 64. The grader batches too, so this noise exists there as well.
RTOL = 1e-5


@pytest.fixture(scope="session")
def target_curve():
    return np.load(ROOT / "kangaroo_target_curves.npy")[TARGET]


@pytest.fixture(scope="session")
def random_mechs():
    """40 random valid 7-joint mechanisms, like the notebook's starting population,
    each with a random target joint."""
    np.random.seed(0)
    randomizer = MechanismRandomizer(min_size=6, max_size=14, device="cpu")
    rng = np.random.default_rng(0)
    return [
        dict(randomizer(n=N), target_joint=int(rng.integers(1, N))) for _ in range(40)
    ]


@pytest.fixture(scope="session")
def notebook_problem_class():
    """The advanced notebook's own mechanism_synthesis_optimization, executed from
    the notebook file (its import cell + its class cell)."""
    cells = ["".join(c["source"]) for c in json.loads(ADVANCED_NB.read_text())["cells"]]
    imports = next(c for c in cells if "from pymoo.core.mixed import" in c)
    class_cell = next(c for c in cells if "class mechanism_synthesis_optimization" in c)
    namespace = {"np": np, "REFERENCE_POINTS": REFERENCE_POINTS, "target_index": TARGET}
    exec(imports, namespace)  # noqa: S102 -- running the notebook's own code is the point
    exec(class_cell, namespace)  # noqa: S102
    return namespace["mechanism_synthesis_optimization"]


def test_batched_evaluate_matches_one_at_a_time(random_mechs, target_curve):
    distance, material = evaluate(random_mechs, target_curve)
    for i, m in enumerate(random_mechs):
        d, mat = tools()(
            m["x0"],
            m["edges"],
            m["fixed_joints"],
            m["motor"],
            target_curve,
            m["target_joint"],
        )
        np.testing.assert_allclose([distance[i], material[i]], [d, mat], rtol=RTOL)


def test_padding_does_not_change_results(random_mechs, target_curve):
    # 5 designs are padded to a batch of 16, 40 designs to a batch of 64.
    few = evaluate(random_mechs[:5], target_curve)
    many = evaluate(random_mechs, target_curve)
    np.testing.assert_allclose(few[0], many[0][:5], rtol=RTOL)
    np.testing.assert_allclose(few[1], many[1][:5], rtol=RTOL)


def test_big_pools_are_scored_in_chunks(random_mechs, target_curve, monkeypatch):
    """Chunked scoring (memory stays bounded) gives the same scores, in order."""
    whole = evaluate(random_mechs, target_curve)
    monkeypatch.setattr(problem, "EVAL_CHUNK", 16)  # 40 designs -> 16 + 16 + 8
    chunked = evaluate(random_mechs, target_curve)
    np.testing.assert_allclose(chunked[0], whole[0], rtol=RTOL)
    np.testing.assert_allclose(chunked[1], whole[1], rtol=RTOL)


def test_batch_sizes():
    sizes = [batch_size_for(n) for n in (1, 8, 9, 50, 100, 200, 1001)]
    assert sizes == [8, 8, 16, 56, 104, 200, 1008]


def test_same_variables_as_the_notebook(notebook_problem_class, target_curve):
    ours = MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], N)
    theirs = notebook_problem_class(
        target_curve, N=N, reference_point=REFERENCE_POINTS[TARGET]
    )
    assert list(ours.vars) == list(theirs.vars)
    for name, var in ours.vars.items():
        assert type(var) is type(theirs.vars[name]), name
        assert var.bounds == theirs.vars[name].bounds, name


def test_conversions_match_the_notebook(
    notebook_problem_class, random_mechs, target_curve
):
    ours = MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], N)
    theirs = notebook_problem_class(
        target_curve, N=N, reference_point=REFERENCE_POINTS[TARGET]
    )
    for m in random_mechs:
        x_ours = ours.from_mech(m)
        x_theirs = theirs.convert_mech_to_1D(target_idx=m["target_joint"], **m)
        assert x_ours == x_theirs

        mech = ours.to_mech(dict(x_ours))
        x0, edges, fixed, motor, target = theirs.convert_1D_to_mech(dict(x_theirs))
        np.testing.assert_array_equal(mech["x0"], x0)
        np.testing.assert_array_equal(mech["edges"], edges)
        np.testing.assert_array_equal(mech["fixed_joints"], fixed)
        np.testing.assert_array_equal(mech["motor"], motor)
        assert mech["target_joint"] == target


def test_objectives_and_constraints_match_the_notebook(
    notebook_problem_class, random_mechs, target_curve
):
    ours = MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], N)
    theirs = notebook_problem_class(
        target_curve, N=N, reference_point=REFERENCE_POINTS[TARGET]
    )
    X = np.array([ours.from_mech(m) for m in random_mechs])
    F_ours, G_ours = ours.evaluate(X, return_values_of=["F", "G"])
    F_theirs, G_theirs = theirs.evaluate(X, return_values_of=["F", "G"])
    np.testing.assert_allclose(F_ours, F_theirs, rtol=RTOL)
    np.testing.assert_allclose(G_ours, G_theirs, rtol=RTOL, atol=1e-5)


def test_ga_designs_convert_to_valid_submission_entries(random_mechs, target_curve):
    problem = MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], N)
    for m in random_mechs:
        entry = to_entry(problem.to_mech(problem.from_mech(m)))
        assert entry["x0"].shape == (N, 2)
        assert entry["target_joint"] == m["target_joint"]


def test_smaller_mechanisms_are_padded_with_fixed_unconnected_joints(
    random_mechs, target_curve
):
    problem = MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], N + 2)
    x = problem.from_mech(random_mechs[0])
    assert x[f"fixed_nodes{N}"] and x[f"fixed_nodes{N + 1}"]
    assert not any(x[f"C{j}_{N}"] for j in range(N))  # padded joints have no links


@pytest.mark.parametrize("n_joints", [4, 21])
def test_problem_rejects_sizes_outside_the_limits(target_curve, n_joints):
    with pytest.raises(ValueError):
        MechanismProblem(target_curve, REFERENCE_POINTS[TARGET], n_joints)


def test_variable_count():
    # N(N-1)/2 - 1 links + 2N positions + N fixed flags + 1 target
    assert len(mixed_variables(N)) == N * (N - 1) // 2 - 1 + 2 * N + N + 1
