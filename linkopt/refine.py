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

# --- "Smarter" descent rules (cfg.refine_method) ---------------------------------
# Adam: the usual defaults. beta1 averages the gradient (momentum), beta2 averages its
# square, so each coordinate ends up with its own step size.
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
# Longest gradient a step is allowed to use, per design (L2 over all its joints).
# DifferentiableTools can return a very large gradient near a jamming configuration;
# unclipped, one such step throws the design far outside the limits and retires it.
GRAD_CLIP = 1.0
# Basin hopping: how many noisy restarts per step size, and how far each jumps.
BASIN_RESTARTS = 5
BASIN_SIGMA = 0.01

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

    cfg.refine_method chooses how each step is taken: "plain" (the notebook's fixed
    step), "adam" (per-joint step sizes from the gradient's history; step_sizes are
    learning rates) or "basin" (plain descent restarted from nearby random positions).
    The guarantees above hold for all three -- they only change how the walk moves, and
    the re-score below reverts anything that didn't actually improve.
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
    # One fixed RNG for the whole refinement, so "basin" is reproducible.
    rng = np.random.default_rng(0)
    for size in cfg.step_sizes:
        if cfg.refine_method == "adam":  # step_sizes are learning rates here
            _, x, at_step, distance = _descend_adam(
                batch, curve, limits, cfg.grad_steps, size
            )
        elif cfg.refine_method == "basin":
            _, x, at_step, distance = _descend_basin(
                batch, curve, limits, cfg.grad_steps, size, rng
            )
        else:
            _, x, at_step, distance = _descend(
                batch, curve, limits, cfg.grad_steps, size
            )
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


def _clipped(grad, limit=GRAD_CLIP):
    """`grad` scaled down if it is longer than `limit` (its direction is kept)."""
    norm = float(np.sqrt(np.sum(np.asarray(grad, dtype=float) ** 2)))
    return grad if norm <= limit or norm == 0 else np.asarray(grad) * (limit / norm)


def _descend_adam(mechs, curve, limits, n_steps, lr):
    """_descend's loop with the Adam update instead of a fixed step.

    Plain descent moves every joint the same multiple of its gradient, so one step size
    has to suit a joint whose gradient is tiny and one whose gradient is huge. Adam
    keeps, per coordinate, a running mean of the gradient (`m`, momentum) and of its
    square (`v`), and steps `lr * m / sqrt(v)`: a coordinate with a small but consistent
    gradient still moves about `lr`, and a coordinate with a huge gradient doesn't
    overshoot. Early steps are bias-corrected, because `m` and `v` start at zero.

    Everything else matches _descend exactly: one batched gradient call per step, the
    same inside-the-limits test, the same "remember each design's best position while
    inside" rule, and the same permanent retirement of a design that steps outside.

    Returns (x, best_x, best_step, best_distance), like _descend.
    """
    edges = [m["edges"] for m in mechs]
    fixed = [m["fixed_joints"] for m in mechs]
    motors = [m["motor"] for m in mechs]
    targets = [m.get("target_joint") for m in mechs]
    beta1, beta2 = ADAM_BETAS

    x = [np.array(m["x0"], dtype=float) for m in mechs]
    x_last = list(x)
    done = np.zeros(len(x), dtype=bool)
    best_x = list(x)
    best_distance = np.full(len(x), np.inf)
    best_step = np.zeros(len(x), dtype=int)
    moment = [np.zeros_like(xi) for xi in x]  # m: the averaged gradient
    velocity = [np.zeros_like(xi) for xi in x]  # v: the averaged squared gradient
    taken = np.zeros(len(x), dtype=int)  # Adam steps each design has taken, for the bias

    for step in range(n_steps + 1):
        distance, material, distance_grad, _ = gradient_tools()(
            x, edges, fixed, motors, curve, list(targets)
        )
        inside = (distance <= limits[0]) & (material <= limits[1])

        for i in np.where(inside & (distance < best_distance))[0]:
            best_x[i], best_distance[i], best_step[i] = x[i], distance[i], step

        for i in np.where(~inside)[0]:
            done[i] = True
            x[i] = x_last[i]

        if step == n_steps or done.all():
            break

        x_last = list(x)
        for i in np.where(~done)[0]:
            grad = _clipped(distance_grad[i])
            moment[i] = beta1 * moment[i] + (1 - beta1) * grad
            velocity[i] = beta2 * velocity[i] + (1 - beta2) * grad**2
            taken[i] += 1
            m_hat = moment[i] / (1 - beta1 ** taken[i])
            v_hat = velocity[i] / (1 - beta2 ** taken[i])
            x[i] = x[i] - lr * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    return x, best_x, best_step, best_distance


def _descend_basin(mechs, curve, limits, n_steps, step_size, rng):
    """Basin hopping: the plain descent, restarted from nearby random positions.

    One downhill walk only reaches the bottom of the valley it starts in. Basin hopping
    jumps to a nearby random point (every joint nudged by BASIN_SIGMA), walks downhill
    from there with the unchanged _descend, and keeps the new result for a design only
    if it beat that design's best so far. BASIN_RESTARTS jumps per step size.

    Positions are clipped to [0, 1] after each jump: that is the box the GA's position
    variables live in, so a design outside it isn't one the GA could have produced.

    Returns (x, best_x, best_step, best_distance), like _descend. `best_step` is the
    step within the winning restart, so 0 means no restart improved the design.
    """
    best_x = [np.array(m["x0"], dtype=float) for m in mechs]
    best_distance = np.full(len(mechs), np.inf)
    best_step = np.zeros(len(mechs), dtype=int)

    for _ in range(BASIN_RESTARTS):
        jumped = [
            dict(
                m,
                x0=np.clip(
                    np.asarray(m["x0"], dtype=float)
                    + rng.normal(0.0, BASIN_SIGMA, size=np.shape(m["x0"])),
                    0.0,
                    1.0,
                ),
            )
            for m in mechs
        ]
        _, x, at_step, distance = _descend(jumped, curve, limits, n_steps, step_size)
        # Keep a restart only where it improved on every restart so far.
        better = (distance < best_distance) & (at_step > 0)
        for i in np.where(better)[0]:
            best_x[i], best_distance[i], best_step[i] = x[i], distance[i], at_step[i]

    return best_x, best_x, best_step, best_distance


def _scores(mechs, curve) -> np.ndarray:
    """[distance, material] per mechanism, from the grader's scorer (n x 2)."""
    if len(mechs) == 0:
        return np.empty((0, 2))
    return np.column_stack(evaluate(mechs, curve))
