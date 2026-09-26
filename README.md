# 2.156 — ps1
September 2026. MIT. 2.156

Fatak Borhani, Joel Pederson, & Leif Akerley

**PS1 objective:** Design a linkage mechanism that can trace a target curve whilst minimizing material usage and complexity.

## Prerequisites

- **A conda installer.** [Miniforge](https://github.com/conda-forge/miniforge) is recommended (it ships
  `mamba` and defaults to conda-forge). Already have Anaconda or Miniconda? That works too — use `conda`
  wherever `mamba` appears below.
  - macOS: `brew install miniforge`, or download the installer from the Miniforge page and run
    `bash Miniforge3-MacOSX-$(uname -m).sh`
  - Windows: run the Miniforge `.exe` installer, then use the "Miniforge Prompt" for the commands below
  - Linux: `bash Miniforge3-Linux-$(uname -m).sh`
  - Then `conda init` (restart your terminal) so `conda activate` works.
- **Git**, and access to this repo (ask Joel to add you as a collaborator).
- **VS Code** with the Python and Jupyter extensions (or use JupyterLab, which the env includes).

No conda at all? Use Colab instead — see [Running on Colab instead](#running-on-colab-instead).

## Setup

```bash
git clone git@github.com:Joel-Pederson/2.156-ps1.git   # no SSH key? use https://github.com/Joel-Pederson/2.156-ps1.git
cd 2.156-ps1
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate ps1
python -m ipykernel install --user --name ps1 --display-name "Python (ps1)"
```

Then open the notebook and select the `Python (ps1)` kernel.

The course library `LINKS/` (plus `kangaroo_target_curves.npy` and `starter_mechanism.npy`) is
copied from [decode-mit/2.156-CP1-2026](https://github.com/decode-mit/2.156-CP1-2026) and committed
here, so no clone step is needed. Staff may still update it — check their repo for new commits
before final submission. `LINKS` runs on JAX, pinned to CPU (`JAX_PLATFORMS=cpu`) in the notebooks.

## Workflow

**Optimization runs in Python files; notebooks are for looking at results and exploring.**

| Where | What it's for | How |
|---|---|---|
| `run.py` *(coming)* | The real runs: all 3 kangaroos in parallel, minutes to overnight | `python run.py --preset quick` in a terminal. Saves `runs/<timestamp>/`, logs the score, updates `submissions/best.npy` when it improves |
| `ps1_results.ipynb` *(coming)* | Visualize: scores, hypervolume plots, best mechanism per kangaroo vs. its target curve | Open, pick a run, **Run All**. It only loads saved results, so closing it never interrupts a run |
| Any notebook | Quick interactive experiments | `from linkopt.config import preset` / `from linkopt.problem import MechanismProblem, evaluate` |
| Starter / advanced notebooks | The course's explanations and examples | Read only. To experiment, work in a copy: `tests/test_problem.py` compares our code against the advanced notebook's original class cell |

Keeping runs out of notebooks means a run doesn't depend on VS Code or a kernel staying alive,
CI tests exactly the code that produces submissions, and a finished run can be re-plotted any
number of times without re-optimizing.

**The loop:**

1. Change an idea in `linkopt/`, or a setting (presets live in `linkopt/config.py`).
2. Run it: `python run.py --preset smoke` to check it works (~1 min), then `--preset quick` or
   `full` for a real score.
3. Look at the result in `ps1_results.ipynb`.
4. If the score beat `submissions/best_score.json`, `best.npy` and the JSON are updated;
   commit both together.
5. Push. CI checks the submission against every starter-notebook requirement.

### Choosing settings

`linkopt/config.py` isn't run directly: start from a preset and override any settings.

In Python (a script or a notebook cell):

```python
from linkopt.config import Config, preset

cfg = preset("quick")                                  # the quick preset as-is
cfg = preset("quick", seeds=(0, 1, 2), n_gen=50)       # quick, but 3 seeds and 50 generations
cfg = preset("full", targets=(0,), n_joints=(6, 8))    # full run on Kangaroo 1 only, 6- and 8-joint mechanisms
cfg = Config(pop_size=100, n_gen=40)                   # no preset: defaults for everything else
```

From the terminal *(coming with `run.py`)*: the same settings as flags.

```bash
python run.py --preset smoke                               # does it run? (~1 min)
python run.py --preset quick --seeds 0 1 2 --n-gen 50      # overrides, as above
python run.py --preset full --targets 0 --n-joints 6 8
```

Invalid values stop immediately with a clear message, for example `n_joints=(21,)` (the
notebook allows at most 20 joints), `targets=(3,)` (only kangaroos 0-2), or a misspelled
setting name.

| Setting | Meaning | `smoke` | `quick` | `full` |
|---|---|---|---|---|
| `targets` | Kangaroos to run: `0` = Kangaroo 1, `1` = Kangaroo 2, `2` = Kangaroo 3 | `(0, 1, 2)` | `(0, 1, 2)` | `(0, 1, 2)` |
| `n_joints` | Mechanism sizes to try (at most 20) | `(7,)` | `(7,)` | `(6, 7, 8)` |
| `seeds` | Random seeds; each is an independent GA run, so more seeds give more designs | `(0,)` | `(0,)` | `(0, 1, 2, 3, 4)` |
| `n_start` | Random valid mechanisms the GA starts from | 16 | 50 | 200 |
| `pop_size` | Designs per GA generation | 16 | 50 | 200 |
| `n_gen` | GA generations | 3 | 30 | 150 |
| `mutation_prob` | GA mutation rate: higher explores more, lower refines more | 0.5 | 0.5 | 0.5 |
| `grad_steps` | Maximum gradient-polish steps per design | 10 | 200 | 1000 |
| `step_size` | Size of each gradient-polish step | 4e-4 | 4e-4 | 4e-4 |
| `n_workers` | Parallel worker processes | 1 | 3 | 3 |

One GA run happens per combination of `targets` × `n_joints` × `seeds`: `full` is
3 × 3 × 5 = 45 runs. Each setting has a one-line explanation in `linkopt/config.py`.

**Submitting:** upload `submissions/best.npy` to the leaderboard. Check it first with
`python score.py --strict submissions/best.npy`.

**Layout:**

```
linkopt/          our framework
  submission.py     builds, checks and saves submissions (the only code that writes them)
  config.py         every run setting + the smoke / quick / full presets
  problem.py        the GA's view of a mechanism + fast batched scoring
score.py          check and score any submission file
submissions/      best.npy (current best) + best_score.json; all *.npy here are checked by CI
tests/            pytest suite (see "Tests and CI")
LINKS/            course library, including the grader (LINKS/CP) - don't edit
runs/             raw output of each run (git-ignored)
```

## Running on Colab instead

Open a notebook straight from GitHub:
`https://colab.research.google.com/github/Joel-Pederson/2.156-ps1/blob/main/<notebook>.ipynb`

The first cell detects Colab, clones this repo into `/content/2.156-ps1`, `cd`s into it and
pip-installs `pymoo` and `svgpath2mpl`. Locally that cell does nothing.

Colab does **not** sync back to this repo. To save work: File → Save a copy in GitHub (pick this
repo and `main`), and download any `.npy` results before the runtime dies — they're lost otherwise.
Pull locally afterwards so your copy stays current.

## Working together

Notebooks merge badly. To avoid three-way conflicts on cell IDs and outputs:

- Say in chat which section you're taking before you start editing.
- Pull before you edit, push as soon as you're done — don't sit on a dirty notebook.
- nbdime is in the env; enable it once (inside your clone) so notebook diffs are readable:
  ```bash
  conda activate ps1 && nbdime config-git --enable
  ```
- Notebooks are committed **with outputs** — the submission needs them.

## Tests and CI

`pytest` is in `environment.yml`. If your env predates that, update it once:
`mamba env update -f environment.yml`.

**Before you push:**

```bash
conda activate ps1
ruff check linkopt tests score.py
pytest -m "not slow"    # seconds: notebook requirements, format, best-submission guard
pytest                  # everything, including end-to-end runs (minutes)
```

**On GitHub**:

- `.github/workflows/ci.yml` runs lint + `pytest` on every push to every branch. Results show as
  ✅/❌ next to the commit and in the repo's **Actions** tab.
- `.github/workflows/upstream-links.yml` ("Course repo updates") runs on every push, and fails if the course repo has changed since we copied it. It only downloads the course
  repo; it never writes to it. If it fails, see "If the course repo changes" below.

**If the course repo changes** (staff updated `LINKS/`, the notebooks or the data):

1. Read the failing check's log: it lists every added/removed/changed file.
2. Clone the course repo somewhere outside this repo and review what changed, especially
   `LINKS/CP/__init__.py` (the grader) and the notebooks' Instructions / Submission Format.
3. Copy the changed files in. For the two notebooks, merge by hand so our Colab setup cell and
   the "Modified by team" notice survive.
4. `python tests/upstream_manifest.py build <course-repo-clone>` to record the new fingerprints.
5. If the rules changed (limits, joint cap, 1000 cap, normalizers), update the numbers at the
   top of `tests/test_requirements.py` to match the notebook.
6. Re-score: `python score.py --strict submissions/best.npy`, and update
   `submissions/best_score.json` if the score changed.
7. `pytest`, then commit everything together.

**What the tests guarantee:**

- Every `submissions/*.npy` file meets the starter notebook's requirements: one dict with
  keys `Problem 1..3`, none empty, ≤ 1000 mechanisms per problem, ≤ 20 joints per mechanism,
  the exact field types/shapes, `target_joint` set, motor is a link, and every design within
  its kangaroo's distance and material limits (`tests/test_requirements.py`). Keep submission
  files in `submissions/` so they get checked.
- Our submission tooling writes that format exactly, and a file
  scores the same when saved and reloaded (`tests/test_submission_format.py`).
- `submissions/best.npy` never scores below `submissions/best_score.json`
  (`tests/test_best_submission.py`). Only replace it with a better submission, and update the
  JSON in the same commit.
- The grader (`LINKS/CP/__init__.py`) and `kangaroo_target_curves.npy` are byte-identical to the
  course's versions (fingerprints in `tests/upstream_manifest.json`), so local scores match
  the leaderboard's.

Check any submission file by hand with `python score.py <file.npy>` (add `--strict` for our
exact format).
