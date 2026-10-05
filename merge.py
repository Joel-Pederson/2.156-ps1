"""Pool submission files into submissions/best.npy, keeping it only if the score
goes up. Use it to merge teammates' results, or any saved run.

    python merge.py fatak.npy leif.npy           # pool with best.npy; save if better
    python merge.py --dry-run fatak.npy          # show what would happen, change nothing
    python merge.py --fresh a.npy b.npy          # REPLACE best.npy with only these files
                                                 #   (asks you to type RESET first)
    python merge.py --recheck                    # re-select best.npy's own designs (drops
                                                 #   fragile ones; the score can fall a little)

If git reports a conflict on best.npy (two people both improved it), don't pick one
side: save the other version to a file and merge it with this tool.
"""

import argparse
import sys

import linkopt  # noqa: F401  (pins JAX to the CPU before LINKS imports it)
from linkopt.archive import BEST_PATH, recheck_best, update_best
from linkopt.submission import SubmissionError, load, problem_key
from LINKS.CP import N_PROBLEMS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", help="submission .npy files to pool")
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    parser.add_argument(
        "--fresh", action="store_true", help="ignore the current best (asks to confirm)"
    )
    parser.add_argument(
        "--recheck",
        action="store_true",
        help="re-select best.npy's own designs (no files): drops ones select now rejects",
    )
    args = parser.parse_args(argv)
    if args.recheck:
        if args.files or args.fresh:
            parser.error("--recheck takes no files and no --fresh")
        result = recheck_best(write=not args.dry_run)
        _report(result)
        if args.dry_run:
            print("dry run: nothing changed")
        elif result.improved:
            print(f"best.npy RE-SELECTED (previous copy saved to {result.backup})")
        else:
            print("nothing to drop: best.npy unchanged")
        return 0
    if not args.files:
        parser.error("give at least one submission file (or --recheck)")

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
    _report(result)
    if args.dry_run:
        print("dry run: nothing changed")
    elif result.improved:
        print(f"best.npy UPDATED (previous copy saved to {result.backup})")
    else:
        print("no improvement: best.npy unchanged")
    return 0


def _report(result):
    """Per kangaroo: designs in, kept, and why the rest were dropped; then the score."""
    for t, sel in result.selections.items():
        c = sel.counts
        print(
            f"{problem_key(t)}: {c['in']} designs in -> {c['kept']} kept "
            f"(dropped: {c['broken']} broken, {c['outside_limits']} outside limits, "
            f"{c['duplicate']} duplicates, {c.get('fragile', 0)} fragile, "
            f"{c['dominated']} dominated, {c['trimmed']} trimmed)"
        )
    print(
        f"score: {result.old_score:.4f} (current best) -> {result.new_score:.4f} (pooled)"
    )


if __name__ == "__main__":
    sys.exit(main())
