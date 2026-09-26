from ..Optimization import Tools
import numpy as np
from typing import List, Union
from pymoo.indicators.hv import HV

# Reference point (and therefore the constraint upper bounds) for each target curve.
# These are per-problem: the full kangaroo meme (Problem 3) has thin ears/tail that a
# linkage cannot trace well, so it is scored against a looser reference point.
REFERENCE_POINTS = [
    np.array([0.75, 10.0]),  # Problem 1: Kangaroo 1 - round body
    np.array([1.2, 10.0]),  # Problem 2: Kangaroo 2 - no ears, no tail
    np.array([1.75, 20.0]),  # Problem 3: Kangaroo 3 - full meme
]

N_PROBLEMS = len(REFERENCE_POINTS)

# Mechanism complexity constraint stated in the notebook instructions.
MAX_JOINTS = 20

# Per-problem hypervolume normalizers. The reference boxes above have very different
# areas (0.75x10, 1.2x10, 1.75x20), so the attainable hypervolume differs by roughly
# an order of magnitude between problems. Averaging the raw values would silently give
# the easier-to-score problems more weight, so each score is divided by the normalizer
# below before averaging. Think of these as "a good score for this problem".
# These values are empirical: they are picked so each target contributes roughly
# equally to the overall score, not as an upper bound. A single problem's
# normalized score can exceed 1.0 if you beat the "good score" it was set from.
SCORE_NORMALIZERS = [
    2.0,   # Problem 1: Kangaroo 1 - round body
    1.5,   # Problem 2: Kangaroo 2 - no ears, no tail
    10.0,  # Problem 3: Kangaroo 3 - full meme
]

def make_empty_submission():
    return {f'Problem {i + 1}': [] for i in range(N_PROBLEMS)}

def evaluate_submission(
    submission: Union[dict, str],
    target_curves: Union[np.ndarray, str] = 'kangaroo_target_curves.npy') -> float:

    optimization_tools = Tools(
        timesteps=200,
        max_size=20,
        material=True,
        scaled=False,
        device='cpu'
    )
    optimization_tools.compile()

    if isinstance(submission, str):
        submission = np.load(submission, allow_pickle=True).item()
    if isinstance(target_curves, str):
        target_curves = np.load(target_curves)

    if len(target_curves) < N_PROBLEMS:
        raise ValueError(
            f"Expected at least {N_PROBLEMS} target curves to score "
            f"{N_PROBLEMS} problems, but got {len(target_curves)}. "
            "Did you pass the right target curve file?"
        )

    unknown_keys = [k for k in submission
                    if k not in {f'Problem {i + 1}' for i in range(N_PROBLEMS)}]
    if unknown_keys:
        print(f"Warning: ignoring unrecognized submission keys: {sorted(unknown_keys)}. "
              f"Only 'Problem 1'..'Problem {N_PROBLEMS}' are scored.")

    scores = []
    for problem in range(N_PROBLEMS):
        problem_key = f'Problem {problem + 1}'

        # each target is scored against its own reference point
        ref_point = REFERENCE_POINTS[problem]
        ind = HV(ref_point)

        if problem_key in submission:
            if len(submission[problem_key]) == 0:
                scores.append(0.0)
                continue

            x0s = []
            edges = []
            fixed_joints = []
            motors = []
            target_idx = []
            
            counter = 0
            n_malformed = 0
            n_too_large = 0
            for item in submission[problem_key]:
                counter += 1
                if counter > 1000:
                    print(f"Warning: More than 1000 designs submitted for {problem_key}. Only the first 1000 will be evaluated.")
                    break
                if 'x0' not in item or 'edges' not in item or 'fixed_joints' not in item or 'motor' not in item:
                    # Invalid entry, skip (and say so, otherwise this is silent)
                    n_malformed += 1
                    continue

                # Mechanism complexity constraint: no more than 20 joints
                if np.array(item['x0']).shape[0] > MAX_JOINTS:
                    n_too_large += 1
                    continue

                x0s.append(np.array(item['x0']))
                edges.append(np.array(item['edges']))
                fixed_joints.append(np.array(item['fixed_joints']))
                motors.append(np.array(item['motor']))
                target_idx.append(item.get('target_joint', None))

            if n_malformed:
                print(f"Warning: {problem_key}: skipped {n_malformed} entries missing "
                      "one of 'x0'/'edges'/'fixed_joints'/'motor'.")
            if n_too_large:
                print(f"Warning: {problem_key}: skipped {n_too_large} mechanisms with "
                      f"more than {MAX_JOINTS} joints.")

            if len(x0s) > 0:
                distances, material = optimization_tools(
                    x0s=x0s,
                    edges=edges,
                    fixed_joints=fixed_joints,
                    motors=motors,
                    target_curve=target_curves[problem],
                    target_idx=target_idx
                )
            else:
                scores.append(0.0)
                continue
                
            F = np.vstack([distances, material]).T
            # a design is valid if it beats this problem's own reference point
            valids = np.logical_and(F[:,0] < ref_point[0], F[:,1] < ref_point[1])
            if np.sum(valids) == 0:
                scores.append(0.0)
                continue

            score = ind(F[valids])

            scores.append(score)
        else:
            # problem missing from the submission dictionary -> no credit
            scores.append(0.0)

    # Normalize each problem's hypervolume before averaging so that every target
    # contributes equally, regardless of how large its reference box is.
    normalized = [scores[i] / SCORE_NORMALIZERS[i] for i in range(N_PROBLEMS)]

    return {
        'Overall Score': float(np.mean(normalized)),
        'Score Breakdown': {
            f'Problem {i + 1}': float(scores[i]) for i in range(N_PROBLEMS)
        },
        'Normalized Score Breakdown': {
            f'Problem {i + 1}': float(normalized[i]) for i in range(N_PROBLEMS)
        },
    }