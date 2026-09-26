"""linkopt.ga runs the advanced notebook's GA, and its output is submittable."""

import json

import numpy as np
import pytest
from conftest import ROOT

from linkopt import ga
from linkopt.config import Config, preset
from linkopt.ga import _mating, make_start_population, run_ga, target_curve
from linkopt.problem import MechanismProblem, evaluate, tools
from linkopt.submission import build_submission, save, to_entry
from LINKS.CP import REFERENCE_POINTS

ADVANCED_NB = ROOT / "Fall_26_CP1_Advanced_Starter_Notebook.ipynb"


class _OneAtATime(MechanismProblem):
    """Our problem scoring one design per LINKS call, like the notebook, so both GAs
    see bit-identical numbers (batching shifts float32 results by ~1e-6, and a GA
    amplifies any difference into a completely different run)."""

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            m = self.to_mech(x)
            F.append(
                tools()(
                    m["x0"],
                    m["edges"],
                    m["fixed_joints"],
                    m["motor"],
                    self.target_curve,
                    m["target_joint"],
                )
            )
        out["F"] = np.array(F, dtype=float)
        out["G"] = out["F"] - self.reference_point


def test_start_population_is_reproducible_and_simulates():
    a = make_start_population(7, 6, seed=3)
    b = make_start_population(7, 6, seed=3)
    for m, m2 in zip(a, b):
        np.testing.assert_array_equal(m["x0"], m2["x0"])
        np.testing.assert_array_equal(m["edges"], m2["edges"])
        assert m["x0"].shape == (7, 2) and list(m["motor"]) == [0, 1]
    distance, material = evaluate(a, target_curve(0))
    assert np.isfinite(distance).all() and np.isfinite(material).all()


def test_mutation_prob_sets_every_variable_type():
    defaults = _mating(None)  # pymoo's defaults, as the notebook runs
    custom = _mating(0.3)
    assert all(m.prob.value == 0.3 for m in custom.mutation.values())
    assert any(m.prob.value != 0.3 for m in defaults.mutation.values())


def test_ga_is_the_notebooks_ga(monkeypatch):
    """Same start, same seed, same scores -> identical populations every generation."""
    target, n, seed, gens, pop = 1, 7, 123, 8, 30
    cells = ["".join(c["source"]) for c in json.loads(ADVANCED_NB.read_text())["cells"]]
    ns = {"np": np, "REFERENCE_POINTS": REFERENCE_POINTS, "target_index": target}
    exec(next(c for c in cells if "from pymoo.core.mixed import" in c), ns)  # noqa: S102
    exec(next(c for c in cells if "class mechanism_synthesis_optimization" in c), ns)  # noqa: S102
    nb_problem = ns["mechanism_synthesis_optimization"](
        target_curve(target), N=n, reference_point=REFERENCE_POINTS[target]
    )
    start = [
        nb_problem.convert_mech_to_1D(**m) for m in make_start_population(n, pop, seed)
    ]

    class sample_from_random(ns["Sampling"]):  # the notebook's sampling class
        def _do(self, problem, n_samples, **kwargs):
            return np.array([start[i % len(start)] for i in range(n_samples)])

    notebook = ns["minimize"](  # the notebook's GA cell
        nb_problem,
        ns["NSGA2"](
            pop_size=pop,
            sampling=sample_from_random(),
            mating=ns["MixedVariableMating"](
                eliminate_duplicates=ns["MixedVariableDuplicateElimination"]()
            ),
            mutation=ns["PolynomialMutation"](prob=0.5),
            eliminate_duplicates=ns["MixedVariableDuplicateElimination"](),
        ),
        ("n_gen", gens),
        seed=seed,
        save_history=True,
    )

    captured = {}
    real_minimize = ga.minimize

    def minimize_with_history(*args, **kwargs):
        captured["res"] = real_minimize(*args, save_history=True, **kwargs)
        return captured["res"]

    monkeypatch.setattr(ga, "MechanismProblem", _OneAtATime)
    monkeypatch.setattr(ga, "minimize", minimize_with_history)
    run_ga(
        target,
        n,
        seed,
        Config(targets=(target,), n_start=pop, pop_size=pop, n_gen=gens),
    )
    ours = captured["res"]

    assert len(ours.history) == len(notebook.history) == gens
    for g, (a, b) in enumerate(zip(notebook.history, ours.history), 1):
        fa, fb = a.pop.get("F"), b.pop.get("F")
        assert fa.shape == fb.shape, f"generation {g}"
        np.testing.assert_array_equal(
            np.sort(fa, axis=0), np.sort(fb, axis=0), f"generation {g}"
        )


@pytest.mark.slow
def test_smoke_ga_produces_a_valid_scoring_submission(tmp_path):
    """End to end on Kangaroo 3 (the loosest limits): GA -> submission -> grader."""
    target = 2
    result = run_ga(target, 7, seed=0, cfg=preset("smoke"))
    assert result.designs, "the smoke GA found no feasible design for Kangaroo 3"
    limits = REFERENCE_POINTS[target]
    assert (result.F <= limits).all()
    for m in result.designs:
        entry = to_entry(m)
        assert entry["x0"].shape == (7, 2) and 1 <= entry["target_joint"] <= 6

    scores = save(build_submission({target: result.designs}), tmp_path / "ga.npy")
    assert scores["Score Breakdown"]["Problem 3"] > 0
