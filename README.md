# 2.156 — ps1

Group problem set. Three collaborators.

## Setup

```bash
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate ps1
python -m ipykernel install --user --name ps1 --display-name "Python (ps1)"
```

Then open the notebook and select the `Python (ps1)` kernel.

Torch device is picked at runtime — Apple Silicon gets MPS, everyone else CPU/CUDA:

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Working together

Notebooks merge badly. To avoid three-way conflicts on cell IDs and outputs:

- Say in chat which section you're taking before you start editing.
- Pull before you edit, push as soon as you're done — don't sit on a dirty notebook.
- Install nbdime once so diffs are readable:
  ```bash
  mamba install -n ps1 nbdime && nbdime config-git --enable
  ```
- Notebooks are committed **with outputs** — the submission needs them.
