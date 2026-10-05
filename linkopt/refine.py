"""Fine-tune each GA design's joint positions with gradients (the "gradient refinement").

The GA is good at big jumps (trying different structures) but bad at fine-tuning,
because its changes are random. The refinement walks each design straight downhill:
the course's DifferentiableTools gives, for every joint, the direction to move it
that reduces distance fastest (the gradient), and we take many small steps that way.

Only the joint positions (x0) change. Links, fixed joints, motor and target joint
stay exactly as the GA left them.

This is the advanced notebook's gradient loop ("Now let's take the GA solutions from
above and just optimize for the lowest distance"), with four differences:
    1. Each design returns its BEST position (lowest distance while inside the
       limits), not wherever it ended up. With a fixed step size, designs can
       overshoot and get worse: on the baseline's Kangaroo 3 designs, the notebook's
       step (4e-4) left 72% of the designs that moved with a higher distance.
       Keeping the best means a refined design is never worse than the original.
       And because no single step size suits every design, we run the loop once
       per size in cfg.step_sizes and keep each design's best over all of them.
    2. The refined designs are returned for the submission (the notebook computes
       them but never submits them).
    3. The stop rule keeps a small safety margin inside the limits (the notebook
       stops at <= limit, but the grader requires < limit; see problem.LIMIT_MARGIN).
    4. The last step is checked too (the notebook never checks its final step, so a
       design can end just outside the limits).

Typical use, after the GA:
    result = run_ga(target, n_joints, seed, cfg)
    refined = refine(result.designs, target, cfg)
    both = result.designs + refined.moved()      # keep both versions
"""

import time
from dataclasses import dataclass

import numpy as np

from linkopt.config import Config
from linkopt.ga import target_curve
from linkopt.problem import LIMIT_MARGIN, batch_size_for, evaluate, safe_limits
from LINKS.CP import REFERENCE_POINTS
from LINKS.Optimization import DifferentiableTools  # the course's gradient scorer

_GRADIENT_TOOLS = None


def gradient_tools() -> DifferentiableTools:
    """Our one compiled copy of the course's DifferentiableTools.

    Like problem.tools(), but it also returns gradients: for each design, an N x 2
    array saying how distance (and material) change as each joint moves in x and y.
    Same settings as the grader. Built on first use, then reused.
    """
    global _GRADIENT_TOOLS
    if _GRADIENT_TOOLS is None:
        _GRADIENT_TOOLS = DifferentiableTools(
            timesteps=200, max_size=20, material=True, scaled=False, device="cpu"
        )
        _GRADIENT_TOOLS.compile()
    return _GRADIENT_TOOLS


@dataclass
class RefineResult:
    """Everything one refinement returns (like a MATLAB struct with fixed fields).

    designs, F_before, F_after and steps are in the same order as the designs that
    went in: designs[i] is the refined version of input design i.
    """

    target: int  # which kangaroo (0 = Kangaroo 1)
    designs: list  # refined designs, as mechanism dicts (only x0 differs)
    F_before: np.ndarray  # [distance, material] of each input design
    F_after: np.ndarray  # [distance, material] of each refined design
    steps: np.ndarray  # step at which each design's best was reached (0 = unchanged)
    step_size: np.ndarray  # step size that gave each design its best (0 = unchanged)
    seconds: float  # how long the refinement took

    def moved(self) -> list:
        """The refined designs that actually changed (steps > 0).

        Unchanged ones are copies of the inputs, so adding them again would only
        waste submission slots.
        """
        return [d for d, n in zip(self.designs, self.steps) if n > 0]


def refine(designs, target, cfg: Config, margin=LIMIT_MARGIN) -> RefineResult:
    """Gradient-refine `designs` (mechanism dicts) for kangaroo `target`.

    Refines once per step size in cfg.step_sizes (up to cfg.grad_steps steps each),
    and returns each design at the best position any of them reached: never a higher
    distance than it started with, and always inside the kangaroo's limits by
    `margin`.
    """
    start_time = time.perf_counter()
    curve = target_curve(target)
    limits = safe_limits(REFERENCE_POINTS[target], margin)
    n = len(designs)

    # 1. Score the designs as they are, with the grader's scorer.
    F_before = _scores(designs, curve)
    if n == 0 or cfg.grad_steps == 0:
        return RefineResult(
            target=target,
            designs=[dict(d) for d in designs],
            F_before=F_before,
            F_after=F_before.copy(),
            steps=np.zeros(n, dtype=int),
            step_size=np.zeros(n),
            seconds=time.perf_counter() - start_time,
        )

    # 2. Walk downhill, once per step size. The batch is padded to a fixed size (like
    #    problem.evaluate) so JAX doesn't recompile; the padded copies are ignored.
    #    Each design keeps its best position over all step sizes (lowest distance
    #    inside the limits).
    batch = list(designs) + [designs[0]] * (batch_size_for(n) - n)
    best_x = [np.array(d["x0"], dtype=float) for d in designs]
    best_distance = np.full(n, np.inf)
    steps = np.zeros(n, dtype=int)
    step_size = np.zeros(n)
    for size in cfg.step_sizes:
        _, x, at_step, distance = _descend(batch, curve, limits, cfg.grad_steps, size)
        for i in np.where((distance[:n] < best_distance) & (at_step[:n] > 0))[0]:
            best_x[i], best_distance[i] = x[i], distance[i]
            steps[i], step_size[i] = at_step[i], size
    refined = [dict(d, x0=x0) for d, x0 in zip(designs, best_x)]

    # 3. Re-score with the grader's scorer. Safety net: any design outside the
    #    limits (not expected, thanks to the margin), or with a higher distance than
    #    it started with, goes back to its input version. The descent picks "best"
    #    by DifferentiableTools' distance, which can differ slightly from the
    #    grader's (one design: 0.198989 -> 0.199053), so check with the grader's.
    F_after = _scores(refined, curve)
    revert = ~(F_after <= limits).all(axis=1) | (F_after[:, 0] > F_before[:, 0])
    for i in np.where(revert)[0]:
        refined[i] = dict(designs[i])
        steps[i], step_size[i] = 0, 0.0
    F_after[revert] = F_before[revert]

    return RefineResult(
        target=target,
        designs=refined,
        F_before=F_before,
        F_after=F_after,
        steps=steps,
        step_size=step_size,
        seconds=time.perf_counter() - start_time,
    )


def _descend(mechs, curve, limits, n_steps, step_size):
    """The advanced notebook's gradient loop, on one batch of mechanisms, plus a
    record of each design's best position.

    Repeats: score every design; any design now outside `limits` goes back to where
    it was one step ago and stops; every other design moves each joint one small
    step downhill in distance (x0 minus step_size times the distance gradient).
    After the last step, the designs are checked once more (the notebook skips this).

    Returns (x, best_x, best_step, best_distance):
        x              each design's final positions (the notebook's result)
        best_x         each design's best positions: lowest distance seen while
                       inside the limits (the start counts, so it's never worse)
        best_step      the step at which that best was reached (0 = the start)
        best_distance  the distance at that best (inf if never inside the limits)
    """
    edges = [m["edges"] for m in mechs]
    fixed = [m["fixed_joints"] for m in mechs]
    motors = [m["motor"] for m in mechs]
    targets = [m.get("target_joint") for m in mechs]

    x = [np.array(m["x0"], dtype=float) for m in mechs]  # current positions
    x_last = list(x)  # positions one step ago
    done = np.zeros(len(x), dtype=bool)  # stopped designs
    best_x = list(x)
    best_distance = np.full(len(x), np.inf)
    best_step = np.zeros(len(x), dtype=int)

    for step in range(n_steps + 1):  # n_steps moves, n_steps + 1 checks
        # Score every design and get its gradients (one batched call). The material
        # gradient isn't used yet: this refinement only reduces distance.
        distance, material, distance_grad, _ = gradient_tools()(
            x, edges, fixed, motors, curve, list(targets)
        )
        inside = (distance <= limits[0]) & (material <= limits[1])

        # Remember each design's best position so far (inside, lowest distance).
        for i in np.where(inside & (distance < best_distance))[0]:
            best_x[i], best_distance[i], best_step[i] = x[i], distance[i], step

        # Designs outside the limits: go back one step, and stop (the notebook's rule).
        for i in np.where(~inside)[0]:
            done[i] = True
            x[i] = x_last[i]

        if step == n_steps or done.all():
            break

        # Every design still going moves one step downhill.
        x_last = list(x)
        for i in np.where(~done)[0]:
            x[i] = x[i] - step_size * distance_grad[i]

    return x, best_x, best_step, best_distance


def _scores(mechs, curve) -> np.ndarray:
    """[distance, material] per mechanism, from the grader's scorer (n x 2)."""
    if len(mechs) == 0:
        return np.empty((0, 2))
    return np.column_stack(evaluate(mechs, curve))
