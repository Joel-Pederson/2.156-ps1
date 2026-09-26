"""Steps 1-2 of the pipeline: random valid starting mechanisms, then NSGA-II.

This is the advanced notebook's GA ("Now let's generate 100 mechanisms of size 7
and initialize a population for optimization" onward) on our batched
MechanismProblem:

    make_start_population   MechanismRandomizer: random mechanisms that simulate.
                            Starting from these is what lets the GA find anything;
                            the notebook shows that fully random variables fail.
    run_ga                  NSGA-II over links, positions, fixed joints and target
                            joint, started from those mechanisms.

Differences from the notebook: `mutation_prob` really sets the mutation rate (the
notebook's setting is ignored, see Config.mutation_prob), and history isn't saved.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableDuplicateElimination, MixedVariableMating
from pymoo.core.sampling import Sampling
from pymoo.core.variable import Binary, Integer, Real
from pymoo.operators.mutation.bitflip import BFM
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from linkopt.config import Config
from linkopt.problem import MechanismProblem
from linkopt.submission import TARGET_CURVES_PATH
from LINKS.CP import MAX_JOINTS, REFERENCE_POINTS
from LINKS.Optimization import MechanismRandomizer

_RANDOMIZER = None


def randomizer() -> MechanismRandomizer:
    """This process's MechanismRandomizer (built once; it compiles a solver)."""
    global _RANDOMIZER
    if _RANDOMIZER is None:
        _RANDOMIZER = MechanismRandomizer(max_size=MAX_JOINTS, device="cpu")
    return _RANDOMIZER


def target_curve(target: int) -> np.ndarray:
    """Kangaroo `target + 1`'s outline (200 x 2)."""
    return np.load(TARGET_CURVES_PATH)[target]


def make_start_population(n_joints: int, count: int, seed: int) -> list[dict]:
    """`count` random mechanisms with `n_joints` joints that simulate (the notebook's
    `[randomizer(n=7) for _ in trange(100)]`), reproducible for a given seed."""
    np.random.seed(seed)  # MechanismRandomizer draws from numpy's global generator
    r = randomizer()
    return [r(n=n_joints) for _ in range(count)]


class _FromDesigns(Sampling):
    """Start the GA from given designs, cycling through them if the population is
    larger (the notebook's `sample_from_random`)."""

    def __init__(self, designs):
        super().__init__()
        self.designs = designs

    def _do(self, problem, n_samples, **kwargs):
        return np.array([self.designs[i % len(self.designs)] for i in range(n_samples)])


def _mating(mutation_prob: float | None) -> MixedVariableMating:
    dedupe = MixedVariableDuplicateElimination()
    if mutation_prob is None:  # pymoo's defaults: what the notebook actually runs
        return MixedVariableMating(eliminate_duplicates=dedupe)
    mutation = {
        Binary: BFM(prob=mutation_prob),
        Real: PM(prob=mutation_prob),
        Integer: PM(prob=mutation_prob, vtype=float, repair=RoundingRepair()),
    }
    return MixedVariableMating(mutation=mutation, eliminate_duplicates=dedupe)


@dataclass
class GAResult:
    target: int  # kangaroo index (0 = Kangaroo 1)
    n_joints: int
    seed: int
    designs: list[dict]  # the final feasible non-dominated designs, as mechanism dicts
    F: np.ndarray  # their (distance, material), one row per design
    n_evals: int  # designs evaluated
    seconds: float
    start_F: np.ndarray = field(
        repr=False
    )  # (distance, material) of the start population


def run_ga(
    target: int, n_joints: int, seed: int, cfg: Config, verbose: bool = False
) -> GAResult:
    """One GA job: NSGA-II on kangaroo `target` with `n_joints`-joint mechanisms.

    Returns the feasible non-dominated designs (possibly none, if the GA never got
    within the kangaroo's distance and material limits).
    """
    t0 = time.perf_counter()
    problem = MechanismProblem(target_curve(target), REFERENCE_POINTS[target], n_joints)
    start = [
        problem.from_mech(m) for m in make_start_population(n_joints, cfg.n_start, seed)
    ]
    start_F = problem.evaluate(np.array(start), return_values_of=["F"])

    algorithm = NSGA2(
        pop_size=cfg.pop_size,
        sampling=_FromDesigns(start),
        mating=_mating(cfg.mutation_prob),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )
    res = minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=seed, verbose=verbose)

    if res.opt is None:  # nothing feasible found
        designs, F = [], np.empty((0, 2))
    else:
        designs = [problem.to_mech(x) for x in res.opt.get("X")]
        F = res.opt.get("F")
    return GAResult(
        target=target,
        n_joints=n_joints,
        seed=seed,
        designs=designs,
        F=F,
        n_evals=res.algorithm.evaluator.n_eval,
        seconds=time.perf_counter() - t0,
        start_F=start_F,
    )
