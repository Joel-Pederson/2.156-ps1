"""Steps 1-2 of the pipeline: random valid starting mechanisms, then NSGA-II.

    random valid mechanisms --from_mech--> GA population --NSGA-II--> best designs
    (make_start_population)                 (run_ga)                   (GAResult)

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
    `[randomizer(n=7) for _ in trange(100)]`), reproducible for a given seed.

    Each is a mechanism dict (x0, edges, fixed_joints, motor = [0, 1]) without a
    target_joint; they simulate, but usually trace nothing like a kangaroo.
    """
    # Same seed -> same mechanisms. MechanismRandomizer draws from numpy's global
    # random generator, so seeding it here makes the whole run repeatable.
    np.random.seed(seed)
    r = randomizer()
    return [r(n=n_joints) for _ in range(count)]


class _FromDesigns(Sampling):
    """Start the GA from given designs, cycling through them if the population is
    larger (the notebook's `sample_from_random`).

    In pymoo, a "sampling" is what creates the first generation; the default one
    picks fully random variables, which the notebook shows doesn't work here.
    """

    def __init__(self, designs):
        super().__init__()
        self.designs = designs  # GA-variable dicts (from MechanismProblem.from_mech)

    def _do(self, problem, n_samples, **kwargs):
        # pymoo asks for n_samples designs (= pop_size); reuse ours in a cycle.
        return np.array([self.designs[i % len(self.designs)] for i in range(n_samples)])


def _mating(mutation_prob: float | None) -> MixedVariableMating:
    """How children are made each generation: pick parents, cross them over, then
    mutate, with an operator per variable type (yes/no, real, integer), and never
    produce a child identical to an existing design.

    mutation_prob None keeps pymoo's defaults (what the notebook actually runs);
    a number sets the chance that each child is mutated, for every variable type.
    """
    dedupe = MixedVariableDuplicateElimination()
    if mutation_prob is None:  # pymoo's defaults: what the notebook actually runs
        return MixedVariableMating(eliminate_duplicates=dedupe)
    mutation = {
        Binary: BFM(prob=mutation_prob),  # flips yes/no switches (links, fixed joints)
        Real: PM(prob=mutation_prob),  # nudges positions (polynomial mutation)
        # the target joint: nudged as a number, then rounded back to a whole joint
        Integer: PM(prob=mutation_prob, vtype=float, repair=RoundingRepair()),
    }
    return MixedVariableMating(mutation=mutation, eliminate_duplicates=dedupe)


@dataclass
class GAResult:
    """What one GA job returns."""

    target: int  # kangaroo index (0 = Kangaroo 1)
    n_joints: int  # mechanism size used
    seed: int  # random seed used
    designs: list[dict]  # the final feasible non-dominated designs, as mechanism dicts
    F: np.ndarray  # their [distance, material], one row per design (same order)
    n_evals: int  # how many designs the GA scored in total
    seconds: float  # wall-clock time of the job
    # [distance, material] of the starting mechanisms, for before/after comparisons
    start_F: np.ndarray = field(repr=False)


def run_ga(
    target: int, n_joints: int, seed: int, cfg: Config, verbose: bool = False
) -> GAResult:
    """One GA job: NSGA-II on kangaroo `target` with `n_joints`-joint mechanisms.

    Returns the feasible non-dominated designs (possibly none, if the GA never got
    within the kangaroo's distance and material limits).
    """
    t0 = time.perf_counter()

    # 1. The problem: this kangaroo's outline and limits, N-joint mechanisms.
    problem = MechanismProblem(target_curve(target), REFERENCE_POINTS[target], n_joints)

    # 2. The starting designs: random valid mechanisms, flattened into GA variables.
    start = [
        problem.from_mech(m) for m in make_start_population(n_joints, cfg.n_start, seed)
    ]
    start_F = problem.evaluate(np.array(start), return_values_of=["F"])  # for reference

    # 3. The GA: NSGA-II, starting from our designs, with our mating (crossover +
    #    mutation) settings, and no duplicate designs in the population.
    algorithm = NSGA2(
        pop_size=cfg.pop_size,
        sampling=_FromDesigns(start),
        mating=_mating(cfg.mutation_prob),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    # 4. Run it for n_gen generations. pymoo calls problem._evaluate once per
    #    generation with the whole population. seed makes the run repeatable.
    res = minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=seed, verbose=verbose)

    # 5. The result: res.opt is the feasible non-dominated set (inside both limits,
    #    and not beaten on both objectives), or None if nothing got inside the limits.
    #    Un-flatten each back into a mechanism dict, ready for submission.
    if res.opt is None:
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
