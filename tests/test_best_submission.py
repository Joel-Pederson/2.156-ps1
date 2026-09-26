"""Guard for submissions/best.npy: it never scores worse than its recorded score.
(Its format and limits are checked in test_requirements.py, like every submission.)

To replace it, write the new file with linkopt.submission.save and update
submissions/best_score.json in the same commit.
"""

import json

from conftest import BEST_PATH

from linkopt.submission import TARGET_CURVES_PATH
from LINKS.CP import evaluate_submission

RECORD_PATH = BEST_PATH.with_name("best_score.json")
# Scores come from float32 JAX math, so allow for tiny platform differences (macOS vs CI Linux).
REL_TOL = 1e-4


def test_best_submission_is_not_worse_than_recorded():
    recorded = json.loads(RECORD_PATH.read_text())["overall_score"]
    actual = evaluate_submission(str(BEST_PATH), str(TARGET_CURVES_PATH))[
        "Overall Score"
    ]
    assert actual >= recorded * (1 - REL_TOL), (
        f"submissions/best.npy scores {actual:.6f}, below the recorded {recorded:.6f}. "
        "Only replace best.npy with a better submission, and update best_score.json with it."
    )
