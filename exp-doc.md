# Experiment: smarter gradient methods for Kangaroo 2

**Date:** 2026-10-06 · **Branch:** `fatak/K2_material_exp` · **Base commit:** `24b55d3`
**Hint addressed:** *"Are there smarter gradient-based optimization methods?"*
**Status:** measurement only — nothing merged into `submissions/best.npy`.

---

## 1. What the experiment tested

### The question

Our gradient refinement (`linkopt/refine.py`) is textbook-vanilla: a **fixed** step size,
`x ← x − step_size · distance_grad`, repeated up to `grad_steps` times, once per entry in
`cfg.step_sizes`, with each design keeping its own best position. The question was whether
a smarter update rule, at **identical compute**, can lower Kangaroo 2's distances further
than that fixed step can.

### Why refinement is the right lever for K2

There is an asymmetry in `refine.py` worth stating plainly: it asks `DifferentiableTools`
for both gradients and then **discards the material gradient**.

```python
distance, material, distance_grad, _ = gradient_tools()(...)
# The material gradient isn't used yet: this refinement only reduces distance.
```

So refinement only ever reduces **distance**, and its revert rule only checks that the
design stayed inside the limits and that distance didn't worsen. Material is free to drift.
That makes refinement the one stage in the pipeline that moves designs in the pure
hypervolume-adding direction — leftward on the distance axis — without paying for it in
material.

### The three arms

| arm | update rule |
|---|---|
| `plain` | today's behavior. `x ← x − step_size · grad`, fixed step for every joint. |
| `adam` | per-coordinate adaptive step from the gradient's history (β₁=0.9, β₂=0.999, ε=1e-8), gradient L2-norm clipped to 1.0. |
| `basin` | basin hopping. 5 restarts per step size; each adds N(0, 0.01) noise to every joint, runs the **unmodified** plain descent, and keeps the result only if that design's distance improved. Fixed RNG seed. |

All three share the same limit-handling rules, unchanged: a design that steps outside the
limits is reverted one step and **permanently retired** for that step size; every design
keeps its own best position seen while inside the limits; and at the end every design is
re-scored with the grader's own scorer and reverted if it didn't genuinely improve.

### What was held constant (fair-comparison discipline)

- **Same designs:** all 79 Kangaroo 2 designs in `submissions/best.npy`.
- **Same gradient budget, ~2,000 calls per design.** Gradient calls are batched across all
  designs, so calls-per-batch = calls-per-design:

  | arm | structure | `grad_steps` | gradient calls |
  |---|---|---|---|
  | `plain` | 3 step sizes × (666+1) | 666 | 2,001 |
  | `adam` | 3 learning rates × (666+1) | 666 | 2,001 |
  | `basin` | 3 step sizes × 5 restarts × (133+1) | 133 | 2,010 |

- **Same three scales:** `step_sizes = (4e-4, 1e-4, 3e-5)` in every arm; for `adam` those
  values are learning rates. This was a deliberate choice — it means all three arms probe
  the same three scales and the **only** difference between them is the update rule.
- **Same everything else:** no GA involved. Each arm is a direct `refine()` call, which is
  exactly what `pipeline.run_job` does for a `refine_best` job.

Measured wall-clock landed within **3%** across arms (169–175 s; 84.6 / 86.0 / 87.0 ms per
gradient call), which confirms the budgets really were matched rather than just nominally equal.

---

## 2. Results

### All 79 Kangaroo 2 designs

| method | improved | median drop | best drop | new best distance | ΔK2 raw HV | `merge --dry-run` overall | sec |
|---|---|---|---|---|---|---|---|
| `plain` | 26/79 | 0.01030 | 0.25860 | 0.4632 *(unchanged)* | +0.0016 | 2.8224 → **2.8228** | 169 |
| **`adam`** | **65/79** | **0.01343** | **0.28078** | **0.4540** | **+0.0789** | 2.8224 → **2.8400** | 172 |
| `basin` | 28/79 | 0.01340 | 0.08882 | 0.4632 *(unchanged)* | +0.0037 | 2.8224 → **2.8233** | 175 |

"Median drop" is taken over the designs that actually improved — a median over all 79 would
be dominated by the designs that never move.

### The 29 designs with material > 2.4 (the accurate, heavy ones)

| method | improved | median drop | best drop |
|---|---|---|---|
| `plain` | 24/29 | 0.01030 | 0.02080 |
| **`adam`** | **25/29** | **0.01394** | **0.02459** |
| `basin` | 23/29 | 0.01183 | 0.02052 |

All three arms reach most of this subset — it is the *size* of the improvement that separates
them, not the hit rate. Adam's median drop here is **35% larger** than plain's.

### What survived pooling

| method | designs submitted | kept | dropped as dominated |
|---|---|---|---|
| `plain` | 105 | 79 | 25 (+1 duplicate) |
| `adam` | 144 | **67** | **77** |
| `basin` | 107 | 79 | 28 |

Adam's front is **smaller and better**: its refined designs dominate 77 of the originals they
came from. Fewer points, more area.

---

## 3. How Kangaroo 2's hypervolume was impacted

### The scoring chain

Kangaroo 2's reference point is `[1.2, 10.0]`, so its maximum possible raw hypervolume is
`1.2 × 10.0 = 12.0`. The grade divides by K2's normalizer of **1.5** and the overall score is
the **mean** of the three kangaroos' normalized values:

```
overall = ( HV_k1/2.0 + HV_k2/1.5 + HV_k3/10.0 ) / 3
```

Which gives the leverage of K2 work exactly:

- 1.0 of **raw** K2 hypervolume  →  `1/1.5/3` = **0.2222** overall score
- 1.0 of **normalized** K2 hypervolume  →  **0.3333** overall score

### Per arm

| method | K2 raw HV | K2 normalized (of 8.0 max) | Δ normalized | Δ overall |
|---|---|---|---|---|
| *baseline `best.npy`* | 6.4223 | 4.2815 | — | — |
| `plain` | 6.4239 | 4.2826 | +0.0011 | +0.00036 |
| **`adam`** | **6.5012** | **4.3341** | **+0.0526** | **+0.01752** |
| `basin` | 6.4260 | 4.2840 | +0.0025 | +0.00081 |

Check: adam's `0.0789 raw / 1.5 / 3 = 0.01753`, matching the measured `+0.017524`. The
arithmetic closes, which is a useful guard that the HV numbers and the grader's score agree.

### Where adam's hypervolume came from

Adam is the only arm that moved K2's **best distance**: `0.4632 → 0.4540`, a drop of 0.0092.
That design sits at **material 4.477**, at the heavy end of the front. Because hypervolume
gained by a leftward move is `(distance drop) × (10 − material)`, that single design accounts
for

```
(0.4632 − 0.4540) × (10 − 4.477) = +0.051 raw HV
```

— about **65%** of adam's entire +0.0789 gain, from one design. The rest comes from many
smaller leftward moves across the front.

Plain and basin never improved the 0.4632 design at all. Their largest single drops (0.2586
and 0.0888) landed on mid-material designs that were already dominated, so they bought
almost nothing: both arms' pooled fronts still hold 79 designs with the same best distance
they started with.

**This also connects to K2's structural problem.** The front's designs top out at material
4.5165 while the limit is 10.0, leaving ~2.5 raw HV of unclaimed area above it. Adam's win
was on the single heaviest design on the front — exactly the region where distance
improvements are worth the most, and the region a plain fixed step never reached.

---

## 4. Comparing the three methods, in plain terms

**Plain gradient descent** always moves every joint the same fixed multiple of its gradient.
That's the problem: one step size has to serve a joint whose gradient is tiny and a joint
whose gradient is huge. The big-gradient joint overshoots, pushes the design past the
distance or material limit, and the design is then **permanently retired** for that step size.
The small-gradient joint barely moves at all. Running three different step sizes is the
existing workaround, and the fact that it is needed is itself evidence the single fixed step
is the weak point.

**Adam** gives every coordinate its own step size. It keeps two running averages per
coordinate — one of the gradient (momentum, β₁) and one of the gradient *squared* (β₂) — and
steps roughly `lr · mean / sqrt(mean of squares)`. The division is what matters: a joint with
a small but *consistent* gradient still moves about `lr` per step instead of stalling, and a
joint with a huge gradient gets scaled back down instead of overshooting. Momentum also
carries a design through flat stretches where a plain step would crawl. That one change took
designs improved from 26 to 65 at identical compute.

**Basin hopping** attacks a different problem — being stuck at the bottom of the wrong
valley. It jumps to a nearby random point, walks downhill with the plain rule, and keeps the
jump only if it landed lower. It barely helped here (+0.0008 overall), and the result is
informative: **K2's refinement problem was never a local-minimum problem.** It was a
step-size problem. The adaptive method won; the restart method didn't.

One honest caveat on the basin arm as specified: *all five* of its restarts are noised, so it
never descends from the original un-noised position, and it splits its 2,000 calls into five
shorter 133-step descents instead of one 666-step descent. Some of its weakness is therefore
structural rather than a verdict on basin hopping in general. A version that kept one
un-noised restart, or spent more steps per restart, would be a fairer test — not run here.

---

## 5. Code changes

New behavior sits behind one setting whose default is today's behavior, so existing scores
stay reproducible.

| File | Change |
|---|---|
| `linkopt/config.py` | `refine_method: str = "plain"` + validation against `REFINE_METHODS = ("plain", "adam", "basin")` |
| `linkopt/refine.py` | `_descend_adam`, `_descend_basin`, `_clipped`, constants (`ADAM_BETAS`, `ADAM_EPS`, `GRAD_CLIP`, `BASIN_RESTARTS`, `BASIN_SIGMA`), dispatch in `refine()` |
| `run.py` | `--refine-method {plain,adam,basin}`; added to `RESUME_KEEPS` |
| `tests/test_refine.py`, `tests/test_config.py` | 6 new tests |

**`_descend` was deliberately not touched.**
`tests/test_refine.py::test_descend_is_the_notebooks_gradient_loop` execs the advanced
notebook's own loop cell and asserts our positions match it **bit for bit**; editing that
function would risk the equivalence proof. `_descend_basin` reuses it rather than
reimplementing it.

**`refine_method` is intentionally *not* a sweepable setting.** Any entry in `SETTING_FLAGS`
becomes sweepable, and every swept setting must be a column in `SETTING_COLUMNS` →
`JOB_COLUMNS` → `JOBS_LOG_COLUMNS`. `experiments.check()` compares the committed log's
header **exactly** and raises if it differs, so adding a column would make every run refuse
to start until `experiments_jobs.csv` was renamed. Arms are therefore identified by
`run_id`: the method is recorded in `runs/<id>/config.json` (verified: reads `adam`), not in
the DOE log.

---

## 6. Reproducing it

```bash
conda activate ps1

# gate
ruff check linkopt tests run.py
pytest -m "not slow"

# the three arms (direct refine() calls, in-process, ~3 min each)
PYTHONPATH=. python <scratchpad>/grad_arms.py

# what each arm would add, without touching best.npy
python merge.py --dry-run <scratchpad>/arm_plain.npy
python merge.py --dry-run <scratchpad>/arm_adam.npy
python merge.py --dry-run <scratchpad>/arm_basin.npy

# the flag, end to end through the pipeline
python run.py --targets 1 --refine-method adam --refine-best --no-update-best \
  --n-joints 7 --seeds 0 --n-start 16 --pop-size 16 --n-gen 1 --grad-steps 20
```

The harness calls `refine()` directly rather than going through `run.py --refine-best`, for
two reasons: `--refine-best` also creates a GA job whose designs get pooled into the same
`submission.npy`, which would contaminate a "best.npy's designs only" measurement; and it
runs **in-process**, avoiding the `ProcessPoolExecutor` that killed an earlier multi-worker
run on this 16 GB machine (`BrokenProcessPool`, three workers killed, no Python traceback).

---

## 7. Verification

- `ruff check linkopt tests run.py` clean; **161 passed, 3 deselected** (`pytest -m "not slow"`).
- `refine_method="plain"` asserted **bit-identical** to the current default — the regression
  guard that new behavior is opt-in.
- For `adam` and `basin`: every refined design asserted inside the limits and **never worse**
  than its input; `steps`/`step_size` bookkeeping asserted consistent with `moved()`.
- `basin` asserted reproducible under its fixed seed.
- `md5 submissions/best.npy` = `c56effd28185f153bb64f59f4fa80ca1` **before and after every
  arm** — unchanged.
- `update_best(write=False)` and `merge.py --dry-run` agree to 4 decimal places on all three
  arms, independently confirming the harness reproduces the pipeline's own pooling.

---

## 8. Limitations and next steps

**Limitations**

- One target (K2), one design set (the 79 already in `best.npy`). No seed replication — these
  methods are deterministic given the inputs, but the *pool* they were given is a single sample
  of what the GA produces.
- The basin arm's structure is unfavorable as specified (see §4).
- Adam's learning rates were inherited from `step_sizes` for comparability, not tuned. 1e-4 is
  in the set, but no sweep was run.
- Nothing was merged, so the K2 gain is a `--dry-run` projection, not a recorded score.

**Next steps, in order of expected value**

1. **Merge the adam arm.** `+0.0176` overall is sitting in `<scratchpad>/arm_adam.npy` and
   `best.npy` is monotone by construction, so pooling it cannot lose anything.
2. **Use adam inside a real GA run** (`--refine-method adam` without `--refine-best`). Adam has
   so far only touched 79 existing designs; behind the GA it would refine every design the GA
   produces.
3. **Tune the learning rates.** The three values came from plain descent's step sizes and have
   no reason to be optimal for an adaptive rule.
4. **Re-test basin fairly** — one un-noised restart, or fewer/longer restarts.
5. **Use the material gradient.** It is computed and discarded today; a method that trades
   distance against material would be a genuinely different experiment.
