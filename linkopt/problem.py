"""The GA's view of a mechanism, and fast batched scoring.

MechanismProblem is the advanced notebook's mechanism_synthesis_optimization
(same variables, same conversions, same objectives and constraints) with one
change: pymoo hands it the whole population at once and it scores every design in
a single batched LINKS call, instead of one design per call.

Representation (from the advanced notebook), for an N-joint mechanism:
    C{j}_{i}        Binary   joint j and joint i are linked (upper triangle, j < i;
                             C0_1 is always 1 because link 0-1 is the motor)
    X0{k}           Real     the N x 2 joint positions, flattened, each in [0, 1]
    fixed_nodes{i}  Binary   joint i is fixed to the ground
    target          Integer  the joint whose path is scored, 1..N-1
The motor is always the link [0, 1].
"""

from collections.abc import Mapping, Sequence

import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.variable import Binary, Integer, Real

from linkopt.config import MIN_JOINTS
from LINKS.CP import MAX_JOINTS
from LINKS.Optimization import Tools

MOTOR = np.array([0, 1])

_TOOLS = None


def tools() -> Tools:
    """This process's compiled LINKS scorer, with the grader's settings.

    Kept at module level rather than on the problem (the notebooks do the same,
    because pymoo deep-copies problems).
    """
    global _TOOLS
    if _TOOLS is None:
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
    return max(8, -(-n // 8) * 8)


def evaluate(mechs: Sequence[Mapping], target_curve) -> tuple[np.ndarray, np.ndarray]:
    """Distance and material of every mechanism against one target curve, in one
    batched call. A target_joint of None means the grader's default joint.
    Unsimulatable designs get inf, as in LINKS.

    LINKS computes in float32, and results shift by ~1e-6 (relative) with batch size;
    the grader batches differently, so filter against the limits with a small margin.
    """
    n = len(mechs)
    if n == 0:
        return np.empty(0), np.empty(0)
    padded = list(mechs) + [mechs[0]] * (batch_size_for(n) - n)
    distance, material = tools()(
        [np.asarray(m["x0"]) for m in padded],
        [np.asarray(m["edges"]) for m in padded],
        [np.asarray(m["fixed_joints"]) for m in padded],
        [np.asarray(m["motor"]) for m in padded],
        np.asarray(target_curve),
        [m.get("target_joint") for m in padded],
    )
    return np.asarray(distance)[:n], np.asarray(material)[:n]


def mixed_variables(n_joints: int) -> dict:
    """The advanced notebook's variables, in the same order."""
    variables = {}
    for i in range(n_joints):
        for j in range(i):
            variables[f"C{j}_{i}"] = Binary()
    del variables["C0_1"]  # always linked: it's the motor
    for k in range(2 * n_joints):
        variables[f"X0{k}"] = Real(bounds=(0.0, 1.0))
    for i in range(n_joints):
        variables[f"fixed_nodes{i}"] = Binary()
    variables["target"] = Integer(bounds=(1, n_joints - 1))
    return variables


class MechanismProblem(Problem):
    """Minimize (distance, material) subject to distance <= limit, material <= limit,
    over N-joint mechanisms; evaluated one whole population per call."""

    def __init__(self, target_curve, reference_point, n_joints: int):
        if not MIN_JOINTS <= n_joints <= MAX_JOINTS:
            raise ValueError(
                f"n_joints must be in {MIN_JOINTS}..{MAX_JOINTS}, got {n_joints}"
            )
        self.N = n_joints
        self.target_curve = np.asarray(target_curve)
        # The reference point doubles as the constraint limits (distance, material).
        self.reference_point = np.asarray(reference_point, dtype=float)
        super().__init__(vars=mixed_variables(n_joints), n_obj=2, n_constr=2)

    def _evaluate(self, X, out, *args, **kwargs):
        distance, material = evaluate([self.to_mech(x) for x in X], self.target_curve)
        out["F"] = np.column_stack([distance, material])
        out["G"] = out["F"] - self.reference_point  # <= 0 means within the limits

    def to_mech(self, x: Mapping) -> dict:
        """GA variables -> mechanism dict (the notebook's convert_1D_to_mech).

        Unlike the notebook, this doesn't write C0_1 back into the GA's own dict.
        """
        n = self.N
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i):
                C[j, i] = 1 if (j, i) == (0, 1) else x[f"C{j}_{i}"]
        return {
            "x0": np.array([x[f"X0{k}"] for k in range(2 * n)], dtype=float).reshape(
                n, 2
            ),
            "edges": np.array(np.where(C == 1)).T,
            "fixed_joints": np.where([x[f"fixed_nodes{i}"] for i in range(n)])[
                0
            ].astype(int),
            "motor": MOTOR.copy(),
            "target_joint": int(x["target"]),
        }

    def from_mech(self, mech: Mapping) -> dict:
        """Mechanism dict -> GA variables (the notebook's convert_mech_to_1D).

        Assumes the motor is [0, 1], which MechanismRandomizer guarantees. A missing
        target_joint defaults to the last joint. A mechanism with fewer than N joints
        is padded with unconnected joints, which are marked fixed; the notebook
        intended this but its check (`i >= N` inside `range(N)`) never fires.
        """
        n = self.N
        x0 = np.asarray(mech["x0"], dtype=float)
        n_real = x0.shape[0]
        if n_real > n:
            raise ValueError(f"mechanism has {n_real} joints, the problem has {n}")
        edges = np.asarray(mech["edges"])
        fixed = set(np.asarray(mech["fixed_joints"]).tolist())
        target = mech.get("target_joint")

        C = np.zeros((n, n), dtype=bool)
        C[edges[:, 0], edges[:, 1]] = True
        C[edges[:, 1], edges[:, 0]] = True
        flat = np.pad(x0, ((0, n - n_real), (0, 0))).flatten()

        x = {"target": int(n_real - 1 if target is None else target)}
        for i in range(n):
            for j in range(i):
                x[f"C{j}_{i}"] = C[i, j]
        del x["C0_1"]
        for k in range(2 * n):
            x[f"X0{k}"] = flat[k]
        for i in range(n):
            x[f"fixed_nodes{i}"] = (i in fixed) or (i >= n_real)
        return x
