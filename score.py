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
from linkopt.submission import TARGET_CURVES_PATH, SubmissionError, load, validate
from LINKS.CP import evaluate_submission


def main(argv=None) -> int:
    """Check + score each file; return 0 if all are valid, 1 otherwise."""
    # Read the command line: one or more file names, and the optional --strict flag.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="submission .npy files")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require exact types (np.ndarray fields, int target_joint), as this framework writes them",
    )
    args = parser.parse_args(argv)

    status = 0  # becomes 1 if any file is invalid
    for path in args.files:
        print(f"== {path}")
        # 1. Load it the way the grader does, and check the format rules.
        try:
            submission = load(path)
            warnings = validate(submission, strict=args.strict)
        except (SubmissionError, OSError) as e:  # invalid, or the file can't be read
            print(f"INVALID: {e}\n")
            status = 1
            continue

        # 2. Report what passed (and anything suspicious but tolerated).
        for w in warnings:
            print(f"warning: {w}")
        counts = ", ".join(f"{k}: {len(v)}" for k, v in submission.items())
        print(
            f"format OK ({'strict' if args.strict else 'grader-compatible'}); {counts}"
        )
        # 3. Score it with the course's own grader, so the number is the grader's.
        print(
            json.dumps(
                evaluate_submission(str(path), str(TARGET_CURVES_PATH)), indent=2
            )
        )
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
