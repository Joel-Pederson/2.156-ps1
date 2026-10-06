"""A second, smaller GA that moves the joints of ONE mechanism and nothing else.

Why mix two kinds of GA
-----------------------
The mixed GA in ga.py searches everything at once: which joints are linked (yes/no),
which are bolted to the ground (yes/no), which joint is traced (a whole number), and
where all N joints sit (2N numbers). That is what finds good mechanism *shapes*, but it
is a clumsy way to *polish* one: most mutations flip a link or a ground pivot, which
usually turns a working mechanism into one that jams, so the positions never get much
attention. Kangaroo 2's front shows the symptom -- 79 designs but only 9 distinct
shapes, with the best distance stuck at 0.4632.

This module takes a shape the mixed GA already found and freezes it: edges,
fixed_joints, motor and target_joint never change. The only variables left are the 2N
joint coordinates, every one a plain number in [0, 1]. Three things follow:

    1. The search is much smaller -- 12 variables for a 6-joint design instead of ~36.
    2. Every child is the same mechanism with its joints nudged, so it still moves.
       Nothing is thrown away for jamming.
    3. All the variables are continuous, so SBX and PM can interpolate: a child lands
       *between* its parents rather than at a random corner. The population behaves
       like a crowd of hill-climbers that share information.

The pipeline mirrors ga.py's:

    1. start_population   the design's own positions, plus noisy copies of them
    2. NSGA-II            evolves them, still minimizing (distance, material)
    3. GAResult           the same dataclass ga.run_ga returns, so run_job can refine
                          the result with exactly the same code

rank_shapes picks which designs to fine-tune. Ranking Kangaroo 2's front by lowest
distance gives mostly the same shape over and over (11 of its top 15 are one 6-joint,
7-edge topology at slightly different positions), so dedup=True keeps only the
lowest-distance design of each distinct shape.
"""

import hashlib
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from linkopt.config import Config
from linkopt.ga import GAResult, target_curve
from linkopt.problem import evaluate
from LINKS.CP import REFERENCE_POINTS

# The operators this GA uses. pymoo's NSGA2 already defaults to exactly these for a
# real-valued problem (SBX(eta=15, prob=0.9), PM(eta=20), binary tournament selection);
# they are passed explicitly below so the choice is recorded here rather than inherited
# from a library default that could change.
SBX_ETA = 15  # crossover: higher keeps children closer to their parents
SBX_PROB = 0.9  # chance a pair of parents is crossed at all
PM_ETA = 20  # mutation: higher makes the nudge smaller

# How far the starting copies are nudged from the design's own positions. Joint
# coordinates live in [0, 1], so 0.01-0.03 is a 1-3% jiggle: big enough to give
# crossover something to work with, small enough that the mechanism still moves.
START_SIGMAS = (0.01, 0.02, 0.03)


def topology(design) -> tuple:
    """A design's shape, as a hashable key: everything this module holds fixed.

    Two designs with the same key differ only in where their joints sit, which is
    exactly what a positions-only GA searches over -- so fine-tuning both is doing the
    same job twice. Edges are normalized (each link sorted, then the whole list
    sorted) because [1, 3] and [3, 1] are the same bar.
    """
    edges = np.asarray(design["edges"])
    return (
        len(design["x0"]),
        tuple(sorted(tuple(sorted(link)) for link in edges.tolist())),
        tuple(sorted(np.asarray(design["fixed_joints"]).tolist())),
        int(design["target_joint"]),
    )


def shape_key(design) -> str:
    """A short fingerprint of a design's shape, for checking it hasn't been swapped.

    A position_ga job carries an index into best.npy's list for its kangaroo. If that
    list is ever rebuilt -- `archive.select` re-filters and reorders it every time a
    run is pooled into best.npy -- the index points at a different design. Comparing
    joint counts isn't enough to notice: Kangaroo 2's nine shapes include six 6-joint
    ones. So the job records this instead, and the worker checks it.
    """
    return hashlib.sha256(repr(topology(design)).encode()).hexdigest()[:12]


def rank_shapes(designs, target, k: int, dedup: bool = False) -> list[int]:
    """Which of `designs` to fine-tune: the k lowest-distance ones, best first.

    Returns positions in `designs`, so the caller can hand a worker just an index.
    Designs that can't be simulated (infinite distance) are skipped. dedup=True keeps
    only the first design of each `topology`, i.e. the lowest-distance example of every
    distinct shape, which is usually far fewer than k.
    """
    designs = list(designs)
    if not designs:
        return []
    distance, _ = evaluate(designs, target_curve(target))
    order = [int(i) for i in np.argsort(distance) if np.isfinite(distance[i])]
    if dedup:
        seen, keep = set(), []
        for i in order:
            shape = topology(designs[i])
            if shape not in seen:
                seen.add(shape)
                keep.append(i)
        order = keep
    return order[:k]


def start_population(x0, pop_size: int, seed: int) -> np.ndarray:
    """The GA's generation 1: the design itself, then noisy copies of it.

    Row 0 is the design's own positions, untouched, so the GA starts from a design
    already known to be good and (with elitist survival) can never end up worse than
    it. The other rows add normal noise, cycling the width through START_SIGMAS, and
    clip back into [0, 1] so every row is a legal set of coordinates.
    """
    flat = np.asarray(x0, dtype=float).flatten()
    rng = np.random.default_rng(seed)  # same seed -> same starting population
    rows = [flat.copy()]
    for i in range(pop_size - 1):
        sigma = START_SIGMAS[i % len(START_SIGMAS)]
        rows.append(np.clip(flat + rng.normal(0, sigma, flat.size), 0.0, 1.0))
    return np.array(rows)


class PositionProblem(Problem):
    """Minimize (distance, material) over one mechanism's joint coordinates.

    Deliberately a plain real-valued pymoo Problem, not the mixed-variable kind
    MechanismProblem uses: that is what lets the standard SBX/PM operators apply.
    n_var is 2N (x and y of every joint, flattened the same way as problem.py's X0k
    variables), all bounded to [0, 1].
    """

    def __init__(self, base_design, target_curve, reference_point):
        self.x0 = np.asarray(base_design["x0"], dtype=float)
        self.N = self.x0.shape[0]
        # Everything that is NOT a variable, copied once so the base design can't be
        # mutated from under us and every rebuilt mechanism is identical here.
        self.edges = np.asarray(base_design["edges"]).copy()
        self.fixed_joints = np.asarray(base_design["fixed_joints"]).copy()
        self.motor = np.asarray(base_design["motor"]).copy()
        self.target_joint = int(base_design["target_joint"])

        self.target_curve = np.asarray(target_curve)
        # As in MechanismProblem, the reference point doubles as the constraint limits.
        self.reference_point = np.asarray(reference_point, dtype=float)
        super().__init__(
            n_var=2 * self.N,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.zeros(2 * self.N),
            xu=np.ones(2 * self.N),
        )

    def to_mech(self, row) -> dict:
        """One row of 2N numbers -> a mechanism dict: the frozen shape, new positions."""
        return {
            "x0": np.asarray(row, dtype=float).reshape(self.N, 2),
            "edges": self.edges.copy(),
            "fixed_joints": self.fixed_joints.copy(),
            "motor": self.motor.copy(),
            "target_joint": self.target_joint,
        }

    def _evaluate(self, X, out, *args, **kwargs):
        """pymoo calls this once per generation with the whole population (an array of
        2N-number rows). Scored by problem.evaluate, the same batched call the mixed GA
        uses, so both kinds of GA are measured by identical code."""
        distance, material = evaluate([self.to_mech(row) for row in X], self.target_curve)
        out["F"] = np.column_stack([distance, material])
        # Feasible when every G <= 0: distance and material inside the kangaroo's limits.
        out["G"] = out["F"] - self.reference_point


def run_position_ga(
    target, base_design, seed, cfg: Config, expect_shape=None, verbose=False
) -> GAResult:
    """Fine-tune `base_design`'s joint positions with NSGA-II on kangaroo `target`.

    Returns the same GAResult that ga.run_ga returns, so the caller can refine and
    score it with the same code. Its designs list is empty if nothing stayed inside
    the kangaroo's limits (which would be surprising: row 0 of the starting population
    is the base design itself).

    expect_shape, if given, is the `shape_key` of the design the caller meant to pass.
    A worker indexes best.npy by position, so if that list was rebuilt in between it
    would quietly fine-tune a different shape and still produce plausible-looking
    before/after numbers. Checked rather than trusted: better a failed job than a
    believable wrong one.
    """
    start_time = time.perf_counter()

    n_joints = len(np.asarray(base_design["x0"]))
    if expect_shape is not None and shape_key(base_design) != expect_shape:
        raise ValueError(
            f"shape mismatch: this design is {shape_key(base_design)} "
            f"({n_joints} joints), but the job was built for {expect_shape}; "
            "best.npy's designs for this kangaroo must have changed since it started"
        )

    problem = PositionProblem(
        base_design, target_curve(target), REFERENCE_POINTS[target]
    )

    # Generation 1: the design's own positions plus noisy copies (pop_size rows of 2N
    # numbers). pymoo takes a raw array as `sampling`.
    start = start_population(problem.x0, cfg.pop_size, seed)
    start_F = problem.evaluate(start, return_values_of=["F"])

    algorithm = NSGA2(
        pop_size=cfg.pop_size,
        sampling=start,
        crossover=SBX(eta=SBX_ETA, prob=SBX_PROB),
        mutation=PM(eta=PM_ETA),
        # selection: NSGA2's default binary tournament (constraint-aware).
        eliminate_duplicates=True,
    )
    res = minimize(
        problem, algorithm, ("n_gen", cfg.n_gen), seed=seed, verbose=verbose
    )

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
        seconds=time.perf_counter() - start_time,
        start_F=start_F,
    )
