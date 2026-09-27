"""The GA's view of a mechanism, and fast batched scoring.

Two ways to describe the same mechanism
---------------------------------------
1. A mechanism dict: what LINKS simulates, and what a submission entry contains.

    x0            (N, 2) floats  starting (x, y) position of each of the N joints
    edges         (E, 2) ints    the links: each row is two joints joined by a bar
    fixed_joints  (F,)   ints    joints bolted to the ground (they never move)
    motor         (2,)   ints    the link the motor turns; always [0, 1] here
    target_joint  int            the joint whose path is compared to the kangaroo

   Example: a four-bar linkage plus one traced point (5 joints, 5 links):

    x0           = [[0.3, 0.2], [0.3, 0.3], [0.6, 0.2], [0.6, 0.4], [0.4, 0.5]]
    edges        = [[0, 1], [1, 3], [2, 3], [1, 4], [3, 4]]
    fixed_joints = [0, 2]    # joints 0 and 2 are the ground pivots
    motor        = [0, 1]    # the motor spins link 0-1 around joint 0
    target_joint = 4         # joint 4's path is the one that gets scored

2. GA variables: the same mechanism flattened into named yes/no switches and
   numbers, which is all the GA (pymoo) can change. For N joints (the advanced
   notebook's representation):

    C{j}_{i}        yes/no   joints j and i are linked (every pair j < i, except
                             C0_1: link 0-1 is the motor and always exists)
    X0{k}           0 to 1   positions, flattened: X00 = x of joint 0, X01 = y of
                             joint 0, X02 = x of joint 1, ... (2N numbers)
    fixed_nodes{i}  yes/no   joint i is fixed to the ground
    target          1..N-1   the target joint (not 0: that's the fixed motor pivot)

   The example above becomes 25 variables: C1_3, C2_3, C1_4, C3_4 = True (its
   links besides the motor) and the other C's False; X00 = 0.3, X01 = 0.2,
   X02 = 0.3, ...; fixed_nodes0 and fixed_nodes2 = True; target = 4.

from_mech flattens (mechanism dict -> GA variables); to_mech un-flattens.

MechanismProblem is the advanced notebook's mechanism_synthesis_optimization (same
variables, conversions, objectives and constraints), except that pymoo hands it the
whole population at once and it scores every design in one batched LINKS call.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.variable import Binary, Integer, Real

from linkopt.config import MIN_JOINTS
from LINKS.CP import MAX_JOINTS
from LINKS.Optimization import Tools  # the course's scorer (simulate + measure)

MOTOR = np.array([0, 1])  # every mechanism's motor link, in this representation

_TOOLS = None


def tools() -> Tools:
    """Our one compiled copy of the course's scorer, `Tools` (from LINKS).

    `Tools` is the course's code: given mechanisms and a target curve, it simulates
    each mechanism through a full motor turn and returns its distance and material.
    This helper is ours. It creates one `Tools` with exactly the settings the
    course's grader (`evaluate_submission`) uses, so we optimize the same numbers
    the grader computes. It builds that copy on the first call and hands back the
    same copy on every later call. JAX compiles the simulator the first time it sees
    each batch size (about a second each), which is why `evaluate` pads batches to a
    few fixed sizes. The notebooks do the same with their `PROBLEM_TOOLS`.

    Kept at module level rather than on the problem because pymoo deep-copies the
    problem object, and the notebooks note the compiled scorer can't be deep-copied.
    """
    global _TOOLS
    if _TOOLS is None:  # build + compile once, then reuse
        _TOOLS = Tools(
            timesteps=200, max_size=20, material=True, scaled=False, device="cpu"
        )
        _TOOLS.compile()
    return _TOOLS


def batch_size_for(n: int) -> int:
    """Round batch sizes up to a multiple of 8.

    JAX recompiles for every new batch size (~1 s each on an M1 Max), so batches are
    padded to a limited set of sizes. Multiples of 8 waste at most 7 simulations per
    call (powers of two would waste up to half of every GA generation's work).
    """
    return max(8, -(-n // 8) * 8)  # -(-n // 8) is n / 8 rounded up


def evaluate(mechs: Sequence[Mapping], target_curve) -> tuple[np.ndarray, np.ndarray]:
    """Score every mechanism against one target curve, in one batched LINKS call.

    Returns two arrays with one number per mechanism:
        distance:  how far the target joint's path is from the target curve (after
                   the grader's best shift and rotation, no resizing); lower is better
        material:  total length of all links; lower is better
    A target_joint of None means the grader's default joint. Designs that can't be
    simulated get inf, as in LINKS.

    LINKS computes in float32, and results shift by ~1e-6 (relative) with batch size;
    the grader batches differently, so filter against the limits with a small margin.
    """
    n = len(mechs)
    if n == 0:
        return np.empty(0), np.empty(0)
    # Pad with copies of the first mechanism up to a fixed batch size (avoids JAX
    # recompiles); the copies' results are thrown away below.
    padded = list(mechs) + [mechs[0]] * (batch_size_for(n) - n)
    # Score the whole batch in one call to the course's provided scorer (tools() returns
    # our compiled copy of it). It takes one list per field, one entry per mechanism.
    distance, material = tools()(
        [np.asarray(m["x0"]) for m in padded],
        [np.asarray(m["edges"]) for m in padded],
        [np.asarray(m["fixed_joints"]) for m in padded],
        [np.asarray(m["motor"]) for m in padded],
        np.asarray(target_curve),
        [m.get("target_joint") for m in padded],
    )
    return np.asarray(distance)[:n], np.asarray(material)[:n]  # drop the padding


def mixed_variables(n_joints: int) -> dict:
    """Declare the GA variables (see the module docstring), in the advanced
    notebook's order; pymoo draws random numbers in this order."""
    variables = {}
    for i in range(n_joints):  # one link switch per pair of joints j < i
        for j in range(i):
            variables[f"C{j}_{i}"] = Binary()
    del variables["C0_1"]  # always linked: it's the motor
    for k in range(2 * n_joints):  # x, y of every joint, flattened
        variables[f"X0{k}"] = Real(bounds=(0.0, 1.0))
    for i in range(n_joints):  # one fixed-to-ground switch per joint
        variables[f"fixed_nodes{i}"] = Binary()
    variables["target"] = Integer(bounds=(1, n_joints - 1))  # the scored joint
    return variables


class MechanismProblem(Problem):
    """The problem pymoo solves: minimize (distance, material), subject to
    distance <= limit and material <= limit, over N-joint mechanisms.
    Adapted from mechanism_synthesis_optimization in the advanced notebook, but it
    scores a whole population at once rather than one design at a time."""

    def __init__(self, target_curve, reference_point, n_joints: int):
        if not MIN_JOINTS <= n_joints <= MAX_JOINTS:
            raise ValueError(
                f"n_joints must be in {MIN_JOINTS}..{MAX_JOINTS}, got {n_joints}"
            )
        self.N = n_joints
        self.target_curve = np.asarray(target_curve)  # the kangaroo outline, 200 x 2
        # The reference point doubles as the constraint limits (distance, material).
        self.reference_point = np.asarray(reference_point, dtype=float)
        super().__init__(vars=mixed_variables(n_joints), n_obj=2, n_constr=2)

    def _evaluate(self, X, out, *args, **kwargs):
        """pymoo calls this once per generation. X is the whole population: an array
        of GA-variable dicts, one per design."""
        # Un-flatten every design into a mechanism dict, then score them all at once.
        distance, material = evaluate([self.to_mech(x) for x in X], self.target_curve)
        # Objectives to minimize: one row [distance, material] per design.
        out["F"] = np.column_stack([distance, material])
        # Constraints: pymoo treats a design as feasible when every G value is <= 0,
        # i.e. distance <= its limit and material <= its limit.
        out["G"] = out["F"] - self.reference_point

    def to_mech(self, x: Mapping) -> dict:
        """GA variables -> mechanism dict: un-flattens (the notebook's
        convert_1D_to_mech).

        Unlike the notebook, this doesn't write C0_1 back into the GA's own dict.
        """
        n = self.N

        # Links: fill an N x N matrix from the yes/no switches (upper triangle only),
        # with link 0-1 (the motor) always on; every 1 becomes a row [j, i] of edges.
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                C[j, i] = 1 if (j, i) == (0, 1) else x[f"C{j}_{i}"]
        edges = np.array(np.where(C == 1)).T

        # Positions: un-flatten the 2N numbers X00, X01, ... back into N rows of (x, y).
        x0 = np.array([x[f"X0{k}"] for k in range(2 * n)], dtype=float).reshape(n, 2)

        # Fixed joints: the numbers of the joints whose fixed_nodes switch is on.
        fixed_joints = np.where([x[f"fixed_nodes{i}"] for i in range(n)])[0].astype(int)

        return {
            "x0": x0,
            "edges": edges,
            "fixed_joints": fixed_joints,
            "motor": MOTOR.copy(),
            "target_joint": int(x["target"]),
        }

    def from_mech(self, mech: Mapping) -> dict:
        """Mechanism dict -> GA variables: flattens (the notebook's
        convert_mech_to_1D). Used to start the GA from random valid mechanisms.

        Assumes the motor is [0, 1], which MechanismRandomizer guarantees. A missing
        target_joint defaults to the last joint. A mechanism with fewer than N joints
        is padded with unconnected joints, which are marked fixed; the notebook
        intended this but its check (`i >= N` inside `range(N)`) never fires.
        """
        n = self.N
        x0 = np.asarray(mech["x0"], dtype=float)
        n_real = x0.shape[0]  # joints in this mechanism (may be fewer than N)
        if n_real > n:
            raise ValueError(f"mechanism has {n_real} joints, the problem has {n}")
        edges = np.asarray(mech["edges"])
        fixed = set(np.asarray(mech["fixed_joints"]).tolist())
        target = mech.get("target_joint")

        # Links: mark each edge in an N x N yes/no matrix (both directions).
        C = np.zeros((n, n), dtype=bool)
        C[edges[:, 0], edges[:, 1]] = True
        C[edges[:, 1], edges[:, 0]] = True

        # Positions: pad with (0, 0) rows up to N joints, then flatten the N x 2 array
        # into 2N numbers: x of joint 0, y of joint 0, x of joint 1, ...
        flat = np.pad(x0, ((0, n - n_real), (0, 0))).flatten()

        # Build the GA's dict of named variables.
        x = {"target": int(n_real - 1 if target is None else target)}
        for i in range(n):  # a yes/no switch per pair of joints
            for j in range(i):
                x[f"C{j}_{i}"] = C[i, j]
        del x["C0_1"]  # the motor link isn't a variable
        for k in range(2 * n):  # one variable per position number
            x[f"X0{k}"] = flat[k]
        for i in range(n):  # fixed if listed, or if it's a padding joint
            x[f"fixed_nodes{i}"] = (i in fixed) or (i >= n_real)
        return x
