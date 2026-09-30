"""Pool submission files into submissions/best.npy, keeping it only if the score
goes up. Use it to merge teammates' results, or any saved run.

    python merge.py fatak.npy leif.npy           # pool with best.npy; save if better
    python merge.py --dry-run fatak.npy          # show what would happen, change nothing
    python merge.py --fresh a.npy b.npy          # REPLACE best.npy with only these files
                                                 #   (asks you to type RESET first)

If git reports a conflict on best.npy (two people both improved it), don't pick one
side: save the other version to a file and merge it with this tool.
"""

import argparse
import sys

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from linkopt.archive import BEST_PATH, update_best
from linkopt.submission import SubmissionError, load, problem_key
from LINKS.CP import N_PROBLEMS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="submission .npy files to pool")
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore the current best (asks to confirm)"
    )
    args = parser.parse_args(argv)

    # 1. Read the files. A file that isn't a submission at all is skipped; broken
    #    entries inside a good file are dropped later, one by one.
    designs = {t: [] for t in range(N_PROBLEMS)}
    for path in args.files:
        try:
            submission = load(path)
        except (SubmissionError, OSError) as e:
            print(f"skipping {path}: {e}")
            continue
        for t in range(N_PROBLEMS):
            designs[t] += list(submission.get(problem_key(t), []))
        print(
            f"read {path}: "
            + ", ".join(
                f"{problem_key(t)}: {len(submission.get(problem_key(t), []))}"
                for t in range(N_PROBLEMS)
            )
        )

    # 2. A reset throws away the current best's designs: make it deliberate.
    if args.fresh and not args.dry_run:
        answer = input(
            f"--fresh REPLACES {BEST_PATH.name} with only these files. Type RESET to continue: "
        )
        if answer.strip() != "RESET":
            print("cancelled; nothing changed")
            return 1

    # 3. Pool with the current best, select, score, and save if better.
    result = update_best(
        designs,
        source="merge.py " + " ".join(args.files),
        fresh=args.fresh,
        write=not args.dry_run,
    )

    # 4. Report.
    for t, sel in result.selections.items():
        c = sel.counts
        print(
            f"{problem_key(t)}: {c['in']} designs in -> {c['kept']} kept "
            f"(dropped: {c['broken']} broken, {c['outside_limits']} outside limits, "
            f"{c['duplicate']} duplicates, {c['dominated']} dominated, {c['trimmed']} trimmed)"
        )
    print(
        f"score: {result.old_score:.4f} (current best) -> {result.new_score:.4f} (pooled)"
    )
    if args.dry_run:
        print("dry run: nothing changed")
    elif result.improved:
        print(f"best.npy UPDATED (previous copy saved to {result.backup})")
    else:
        print("no improvement: best.npy unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
