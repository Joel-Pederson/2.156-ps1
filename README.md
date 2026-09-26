# 2.156 — ps1
September 2026. MIT 2.156.
Fatak Borhani, Joel Pederson, & Leif Akerley

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate ps1
python -m ipykernel install --user --name ps1 --display-name "Python (ps1)"
```

Then open the notebook and select the `Python (ps1)` kernel.

The course library `LINKS/` (plus `kangaroo_target_curves.npy` and `starter_mechanism.npy`) is
copied from [decode-mit/2.156-CP1-2026](https://github.com/decode-mit/2.156-CP1-2026) and committed
here, so no clone step is needed. Staff may still update it — check their repo for new commits
before final submission. `LINKS` runs on JAX, pinned to CPU (`JAX_PLATFORMS=cpu`) in the notebooks.

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
- Install nbdime once so diffs are readable:
  ```bash
  mamba install -n ps1 nbdime && nbdime config-git --enable
  ```
- Notebooks are committed **with outputs** — the submission needs them.
