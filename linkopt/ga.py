"""Run the genetic algorithm (NSGA-II) on one kangaroo.

This is the advanced notebook's GA, from "Now let's generate 100 mechanisms of
size 7" onward. The pipeline:

    1. make_start_population   random mechanisms that move (don't jam), and, with
       / warm_start_mechs      cfg.warm_start, some of best.npy's designs
    2. run_ga                  NSGA-II evolves them toward the kangaroo
    3. GAResult                the best designs it found, ready to submit

At the detail level, run_ga does the following:
    Start: n_start random mechanisms that move become generation 1 (pop_size designs).
    Then, every generation (n_gen times):
    1. Pick parents at random
    2. Crossover: mix two parents' variables to make each child
    3. Mutate: nudge the children a bit (or flip a yes/no switch)
    4. Score the children: how well they match the kangaroo and how much material they use
    5. Elitist survival: pool parents + children (2 x pop_size designs) and fill
       pop_size seats.
       a. Rank: feasible designs always come first (infeasible ones after them, least
          outside the limits first). Non-dominated sorting peels the feasible designs
          into fronts like an onion: front 1 = nothing beats them on both distance and
          material; set it aside, and front 2 = nothing left beats them; and so on.
       b. Fill the seats one whole front at a time: front 1, then front 2, ...
          Dominated designs survive when front 1 doesn't fill the seats; they give
          crossover varied material to mix.
       c. The first front that doesn't fit gets only the seats left, for its most
          spread-out designs (largest crowding distance; both ends of the front always
          stay). If front 1 alone has more than pop_size designs, it is the one trimmed.
       Parents compete with their children, so a design on the front only loses its
       seat to better ones: the GA's score (practically) never goes down.
    Outcome: after the last generation, the feasible designs on front 1 are the job's
    designs; the rest of the population is discarded.

Two differences from the notebook: mutation_prob actually sets the mutation rate
(the notebook's setting is silently ignored), and the GA history isn't saved.
"""

import time
from dataclasses import dataclass
from pathlib import Path

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
from linkopt.problem import MechanismProblem, evaluate
from linkopt.submission import TARGET_CURVES_PATH, load, problem_key
from LINKS.CP import MAX_JOINTS, REFERENCE_POINTS
from LINKS.Optimization import MechanismRandomizer

# One randomizer for the whole file. MechanismRandomizer is the course staff's code
# (from LINKS); this is our one copy of it. Making it is instant; its first use takes
# about a second (JAX compiles the simulator), and every use after that is fast.
RANDOMIZER = MechanismRandomizer(max_size=MAX_JOINTS, device="cpu")

# best.npy, for warm starts (cfg.warm_start). This is the same file as
# archive.BEST_PATH, spelled out again because archive imports this module.
BEST_PATH = Path(__file__).resolve().parent.parent / "submissions" / "best.npy"


def target_curve(target):
    """Outline of kangaroo `target` (0, 1 or 2) as a 200 x 2 array."""
    return np.load(TARGET_CURVES_PATH)[target]


def make_start_population(n_joints, count, seed):
    """Make `count` random mechanisms with `n_joints` joints each.

    Every one moves without jamming, but traces a random blob, likely not a kangaroo.
    Each is a dict with x0, edges, fixed_joints and motor (always [0, 1]).
    The same seed always gives the same mechanisms.
    """
    np.random.seed(seed)  # the randomizer uses numpy's global random numbers
    return [RANDOMIZER(n=n_joints) for _ in range(count)]


def warm_start_mechs(target, n_joints, count, best_path=None):
    """`count` of best.npy's designs for kangaroo `target`, to start the GA from.

    A random starting mechanism traces a blob, nothing like a kangaroo (distance
    around 2 on most of them), so every run spends its first generations getting back
    to where the last run already was. Once best.npy holds a good front, starting part
    of the population there begins the search from it instead: the advanced notebook's
    "cycle through multiple optimization runs" hint.

    Only mechanisms with at most `n_joints` joints can be used, because
    MechanismProblem.from_mech pads a smaller one up to n_joints (the padding joints
    are unconnected and marked fixed) but cannot shrink a bigger one.

    The ones picked are spread evenly along the front by material, not taken from its
    most accurate end: the hypervolume rewards a spread of designs, and a population
    cloned from one corner of the front leaves crossover nothing to vary.

    Returns [] when best.npy is missing or holds nothing usable, so the caller simply
    falls back to random mechanisms.
    """
    path = Path(best_path) if best_path else BEST_PATH
    if count <= 0 or not path.exists():
        return []
    usable = [
        e
        for e in load(path)[problem_key(target)]
        if np.asarray(e["x0"]).shape[0] <= n_joints
    ]
    if not usable:
        return []
    _, material = evaluate(usable, target_curve(target))
    order = np.argsort(material)  # along the front: cheapest first
    # Evenly spaced picks along the front, both ends included. np.unique also sorts,
    # and drops the repeats when count is larger than the front.
    spread = np.linspace(0, len(order) - 1, min(count, len(order)))
    return [usable[order[i]] for i in np.unique(spread.round().astype(int))]


class _FromDesigns(Sampling):
    """Tells pymoo to start the GA from our designs instead of random numbers.

    pymoo asks for `n_samples` designs (the population size); if we have fewer,
    we cycle through ours. This is the notebook's `sample_from_random`.
    """

    def __init__(self, designs):
        super().__init__()
        self.designs = designs

    def _do(self, problem, n_samples, **kwargs):
        return np.array([self.designs[i % len(self.designs)] for i in range(n_samples)])


def _mating(mutation_prob):
    """How the GA makes children: crossover, then mutation, no duplicates.

    mutation_prob = None  -> pymoo's default mutation (what the notebook runs)
    mutation_prob = 0.3   -> each child has a 30% chance of mutating

    The professor emphasized that tuning this gives diminishing returns, so don't
    over-index on it.
    """
    no_duplicates = MixedVariableDuplicateElimination()

    if mutation_prob is None:
        return MixedVariableMating(eliminate_duplicates=no_duplicates)

    # One mutation rule per kind of variable.
    mutation = {
        Binary: BFM(prob=mutation_prob),  # yes/no (links, fixed joints): flip it
        Real: PM(prob=mutation_prob),  # positions: nudge the number
        # target joint: nudge as a number, then round back to a whole joint
        Integer: PM(prob=mutation_prob, vtype=float, repair=RoundingRepair()),
    }
    return MixedVariableMating(mutation=mutation, eliminate_duplicates=no_duplicates)


@dataclass
class GAResult:
    """Everything one GA run returns (like a MATLAB struct with fixed fields)."""

    target: int  # which kangaroo (0 = Kangaroo 1)
    n_joints: int  # mechanism size used
    seed: int  # random seed used
    designs: list  # best designs found, as mechanism dicts
    F: np.ndarray  # [distance, material] of each design, same order
    n_evals: int  # total designs scored during the run
    seconds: float  # how long the run took
    start_F: np.ndarray  # [distance, material] of the starting mechanisms


def run_ga(target, n_joints, seed, cfg: Config, verbose=False, callback=None):
    """Run NSGA-II on kangaroo `target` using `n_joints`-joint mechanisms.

    Returns a GAResult. Its designs list is empty if the GA never got inside
    the kangaroo's distance and material limits.

    callback (optional): pymoo calls callback(algorithm) after every generation;
    convergence.py uses it to record the score as the GA goes. It only watches:
    the GA runs exactly the same with or without it.
    """
    start_time = time.perf_counter()

    # 1. The problem: what the GA is optimizing. MechanismProblem takes
    #    - target_curve(target):      the kangaroo outline to match, a 200 x 2
    #                                 array of (x, y) points
    #                                 (0 = Kangaroo 1 ... 2 = Kangaroo 3)
    #    - REFERENCE_POINTS[target]:  that kangaroo's limits,
    #                                 [max distance, max material], e.g. [1.2, 10.0]
    #                                 for Kangaroo 2. Designs over either limit are
    #                                 infeasible; it's also the hypervolume corner.
    #    - n_joints:                  how many joints every mechanism has (5 to 20);
    #                                 this fixes how many variables the GA changes
    problem = MechanismProblem(target_curve(target), REFERENCE_POINTS[target], n_joints)

    # 2. Starting designs: random mechanisms, converted to the GA's variables.
    #    make_start_population takes
    #    - n_joints:     joints per mechanism (must match the problem above)
    #    - cfg.n_start:  how many random mechanisms to make (50 in the "quick" preset)
    #    - seed:         same seed -> same mechanisms, so runs are repeatable
    #    problem.from_mech(m) flattens one mechanism dict (x0, edges, fixed_joints,
    #    motor) into the named yes/no switches and numbers the GA works with. With no
    #    target_joint given, the last joint is used as the traced joint.
    #    problem.evaluate scores the starting designs; start_F is one row of
    #    [distance, material] per design, kept only to compare before vs after.
    #    With cfg.warm_start > 0, that fraction of the starting designs comes from
    #    best.npy instead (warm_start_mechs); the rest are random, as before. Fewer
    #    usable designs than asked for just means more random ones.
    warm = warm_start_mechs(target, n_joints, round(cfg.n_start * cfg.warm_start))
    start_mechs = warm + make_start_population(
        n_joints, cfg.n_start - len(warm), seed
    )
    start = [problem.from_mech(m) for m in start_mechs]
    start_F = problem.evaluate(np.array(start), return_values_of=["F"])

    # 3. Set up the GA. NSGA2 takes
    #    - pop_size:              designs per generation (cfg.pop_size, 50 in "quick")
    #    - sampling:              where generation 1 comes from: our starting designs,
    #                             reused in a cycle if pop_size is bigger than n_start
    #    - mating:                how children are made: crossover, then mutation at
    #                             rate cfg.mutation_prob (None = pymoo's defaults)
    #    - eliminate_duplicates:  drop any child identical to an existing design
    algorithm = NSGA2(
        pop_size=cfg.pop_size,
        sampling=_FromDesigns(start),
        mating=_mating(cfg.mutation_prob),
        eliminate_duplicates=MixedVariableDuplicateElimination(),
    )

    # 4. Run it. minimize takes
    #    - problem, algorithm:     from steps 1 and 3
    #    - ("n_gen", cfg.n_gen):   when to stop: after n_gen generations (30 in "quick")
    #    - seed:                   makes the GA's own random choices repeatable
    #    - verbose:                True prints a progress table each generation
    #    - callback:               only passed if given (pymoo can't take None here)
    #    It returns res, pymoo's result object (res.opt is used below).
    watch = {"callback": callback} if callback else {}
    res = minimize(
        problem, algorithm, ("n_gen", cfg.n_gen), seed=seed, verbose=verbose, **watch
    )

    # 5. Collect the best designs. res.opt holds the designs that are inside both
    #    limits and not beaten on both distance and material (None if there are none).
    #    - res.opt.get("X"):  those designs as GA variables; problem.to_mech un-flattens
    #                         each back into a mechanism dict, ready for submission
    #    - res.opt.get("F"):  their [distance, material], one row per design, same order
    if res.opt is None:
        designs = []
        F = np.empty((0, 2))
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
        seconds=time.perf_counter() - start_time,
        start_F=start_F,
    )
