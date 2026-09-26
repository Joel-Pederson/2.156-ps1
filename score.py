"""Check submission files against the starter notebook's format, then score them
with the course grader (LINKS.CP.evaluate_submission).

    python score.py my_full_submission.npy           # what the grader accepts
    python score.py --strict submissions/best.npy    # also require our exact types

Exit code is 1 if any file is invalid, so this can gate CI.
"""

import argparse
import json
import sys

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from LINKS.CP import evaluate_submission
from linkopt.submission import TARGET_CURVES_PATH, SubmissionError, load, validate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="submission .npy files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require exact types (np.ndarray fields, int target_joint), as this framework writes them",
    )
    args = parser.parse_args(argv)

    status = 0
    for path in args.files:
        print(f"== {path}")
        try:
            submission = load(path)
            warnings = validate(submission, strict=args.strict)
        except (SubmissionError, OSError) as e:
            print(f"INVALID: {e}\n")
            status = 1
            continue

        for w in warnings:
            print(f"warning: {w}")
        counts = ", ".join(f"{k}: {len(v)}" for k, v in submission.items())
        print(f"format OK ({'strict' if args.strict else 'grader-compatible'}); {counts}")
        print(json.dumps(evaluate_submission(str(path), str(TARGET_CURVES_PATH)), indent=2))
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
