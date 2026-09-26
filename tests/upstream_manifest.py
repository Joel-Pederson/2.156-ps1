"""Fingerprints (sha256) of the files copied from the course repo (decode-mit/2.156-CP1-2026).

    python tests/upstream_manifest.py build <course-repo-clone>   # (re)write tests/upstream_manifest.json
    python tests/upstream_manifest.py check <course-repo-clone>   # exit 1 if the course repo changed

Read-only with respect to the course repo: it only reads files from a local clone.
`check` runs on every push in CI (.github/workflows/upstream-links.yml) to catch staff
updates; tests/test_requirements.py uses the manifest to confirm the grader and the
target curves are the course's versions.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parent / "upstream_manifest.json"
UPSTREAM_URL = "https://github.com/decode-mit/2.156-CP1-2026"


def course_files(root: Path) -> list[str]:
    """Course files we vendor: LINKS/ code plus the top-level notebooks and data."""
    files = (
        list(root.glob("LINKS/**/*.py"))
        + list(root.glob("*.ipynb"))
        + list(root.glob("*.npy"))
    )
    return sorted(str(p.relative_to(root)) for p in files)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(upstream_root: Path) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,  # not a git checkout -> no commit recorded
    ).stdout.strip()
    return {
        "upstream": UPSTREAM_URL,
        "upstream_commit": commit or None,
        "files": {f: sha256(upstream_root / f) for f in course_files(upstream_root)},
    }


def diff(old: dict, new: dict) -> list[str]:
    old_files, new_files = old["files"], new["files"]
    changes = [
        f"added upstream:   {f}" for f in sorted(new_files.keys() - old_files.keys())
    ]
    changes += [
        f"removed upstream: {f}" for f in sorted(old_files.keys() - new_files.keys())
    ]
    changes += [
        f"changed upstream: {f}"
        for f in sorted(old_files.keys() & new_files.keys())
        if old_files[f] != new_files[f]
    ]
    return changes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument(
        "upstream_root", type=Path, help="path to a clone of the course repo"
    )
    args = parser.parse_args(argv)

    new = build(args.upstream_root)
    if args.command == "build":
        MANIFEST_PATH.write_text(json.dumps(new, indent=2) + "\n")
        print(
            f"wrote {MANIFEST_PATH.name}: {len(new['files'])} files, upstream {new['upstream_commit']}"
        )
        return 0

    old = json.loads(MANIFEST_PATH.read_text())
    changes = diff(old, new)
    if not changes:
        print(f"course repo unchanged since {old['upstream_commit']}")
        return 0
    print(
        f"The course repo changed since {old['upstream_commit']} (now {new['upstream_commit']}):"
    )
    print("\n".join(f"  {c}" for c in changes))
    print(
        "\nReview the changes (the grader lives in LINKS/CP), copy them into this repo, then run\n"
        "  python tests/upstream_manifest.py build <course-repo-clone>\n"
        "and commit the updated files together with tests/upstream_manifest.json."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
