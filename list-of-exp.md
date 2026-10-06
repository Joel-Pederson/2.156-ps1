# CP1: experiments tried so far and next experiments to run

Analysis of `experiments_jobs.csv`, `experiments_log.csv`, `linkopt/` and `submissions/best.npy` as of commit `24b55d3` (Oct 5, 2026). Current best overall score: **2.82**. Kangaroo 2 hypervolume: **6.42**, 54% of its scoring box.

## 1. Experiments already run

| When | What | Purpose / finding | Overall score after |
|---|---|---|---|
| Oct 2 | `smoke` and `quick` presets | Check the pipeline works | 0.78 → 0.84 |
| Oct 2–3 | Mechanism size sweep: 5, 7, 10 and 12 joints × 25 seeds, 150 gens, plus one `--refine-best` pass (plain gradient descent on `best.npy`) | 10–12 joints mostly found no valid design; refining `best.npy` helped | 2.45 |
| Oct 3 | Size sweep: 5, 6, 7 and 8 joints × 25 seeds, 150 gens | Commit 4b61271: 5 joints beat 6, 7 and 8 on the median job | 2.65 |
| Oct 3–4 | Budget sweep: GA length 150 vs 300 gens × refinement 250, 1000 or 3000 steps; 5 and 6 joints × 30 seeds | Commit d9a0ef9: 300 gens + 250 steps gives the most score per compute hour; extra refinement steps barely help | 2.77 |
| Oct 3–4 | Joel's laptop: 5 and 6 joints × 200 seeds, 150 gens | More random restarts (seeds), pooled | merged → 2.68 |
| Oct 5 | Kangaroo 3 only: 8 and 10 joints, 600 gens, 1000 steps (334 of 400 jobs) | Longer runs with bigger mechanisms on the hardest target | 2.82 |
| Oct 5 | Kangaroo 2 only, laptop: 6 and 7 joints, 600 gens, 250 steps (167 jobs), merged as "hailmary" | Same idea on Kangaroo 2; produced its best job ever (5.89) | 2.82 |
| — | `convergence.py` (records the score after every generation) | Built, but its results are in git-ignored run folders, not the CSVs | — |

**Held fixed in every job:**
- Population and starting designs: 200 each.
- Mutation: pymoo's default, never varied.
- Step sizes: the same three (4e-4, 1e-4, 3e-5).
- Refinement: distance-only gradient descent.
- Parent selection: random.
- Starting mechanisms: unscreened random designs.

**Never tried:**
- Kangaroo 1 beyond 300 generations.
- Kangaroo 2 with 8 or more joints at long runs.
- Different population sizes.
- Any of the code-change ideas below.

## 2. Key findings that set the priorities

- **The most accurate designs carry most of the score.** On Kangaroo 2, the most accurate design (distance ≈ 0.465 at material ≈ 4.55) gets credit for the whole strip from material 4.55 out to the limit of 10. That's about 62% of the 6.42.
- **A new design's value shrinks as its material grows.** It can add at most about (10 − material) × (how far it sits below the current staircase). Lower distance at moderate material (about 2.5–5) is worth the most.
- **Gaps in the front are cheap.** The gaps between mechanism families (four-bars, then six-bars, then 9-link designs) cost only a few hundredths of hypervolume.
- **The accurate end comes from more complex mechanisms.** It's held by six-bars and 9-link mechanisms; the four-bars sit at the cheap, low-value end.
- **Judge settings by the best jobs, not the median.** The pooled score comes from the best designs. 74 of Kangaroo 2's 79 designs come from 6- and 7-joint jobs, even though 5 joints won on the median.
- **Refinement has stalled.** Its gain on Kangaroo 2 fell from +0.73 (GA run for 150 gens) to +0.07 (600 gens), so designs are stuck at the bottom of local valleys.
- **Only distance-only optimization is ruled out as a fix.** Switching to "minimize distance only, material ≤ 10" collapses the population to a single design and doesn't reward using more material. Material bands (ε-constraint, experiment #6) are the better version of that idea.

## 3. Professors' hints (advanced notebook) vs. the code

| # | Hint | Status | Where |
|---|---|---|---|
| 1 | Preprocess random mechanisms before the GA | Not done | `ga.make_start_population` uses the random mechanisms as they come out of the generator |
| 2 | Use both gradients in gradient descent | Not done | `refine._descend` throws away the material gradient |
| 3 | Cycle through multiple runs | Partly | `--refine-best` and pooling exist, but the GA never starts from `best.npy` |
| 4 | Mix different kinds of GA | Not done | Only the GA that searches links and positions together |
| 5 | Change mutation and crossover | Partly | `mutation_prob` exists but was never varied; crossover can't be changed; parents are picked at random |
| 6 | A more efficient representation | Not done | Still one on/off switch per possible link |
| 7 | Smarter gradient methods | Partly | Three fixed step sizes; no Adam, backtracking, clipping, or way out of local valleys |

## 4. Next experiments: fastest first, impact scored 1–5

| # | Experiment | Time to a result | Impact | Hint |
|---|---|---|---|---|
| 1 | Mutation-rate sweep | ~30 min (no code) | 2 | 5 |
| 2 | Scale sweep of existing designs | ~30 min | 1 | — |
| 3 | Smarter gradient descent and basin hopping | ~1 h | 3 | 7 |
| 4 | Gradient descent using both distance and material | ~45 min | 2 | 2 |
| 5 | Screened starting population | ~1.5 h | 3 | 1 |
| 6 | Material-band runs (ε-constraint) | ~2–3 h | 3 | — |
| 7 | Warm-start GA from `best.npy` (cycling) | ~2–3 h | 4 | 3 |
| 8 | Positions-only GA on the best shapes | ~3 h | 4 | 4 |
| 9 | Bigger mechanisms, much longer runs | overnight (no code) | 4 | — |
| 10 | Dyad-based representation | 1 day+ | 4 (risky) | 6 |

**Adjusted for what's already been run:**
- **#9 is half done.** Kangaroo 2 has had 6–7 joints at 600 gens, and Kangaroo 3 has had 8–10 joints at 600. Narrow it to Kangaroo 2 with 7–8 joints at 1,000+ gens, plus Kangaroo 1 at 600 gens.
- **#3 has its baseline.** Plain refinement of `best.npy` was run on Oct 2; that's the control for the Adam and basin-hopping versions.

**Order before the deadline (Wed Oct 7):**
1. Tonight: #1, in the background.
2. Tomorrow: #3, #7 and #5.
3. If there's time: #8, then #4 (mainly for a report figure).
4. #10: describe as future work in the report unless someone has a free day.

**Kangaroo 1 and 3:** once something works on Kangaroo 2, rerun it by changing `--targets` (0 = Kangaroo 1, 2 = Kangaroo 3).
- **Kangaroo 1:** has never had a 600-generation run, so it's the natural next target for #9 and #7.
- **Kangaroo 3:** pair #9 with 10–14 joints and #5's screened starts.

## 5. Claude Code prompts

### Shared rules: paste above every prompt

```text
Context: repo 2.156-ps1 (MIT 2.156 CP1, planar linkage synthesis scored by hypervolume).
Our framework is in linkopt/ (config.py presets, ga.py NSGA-II, refine.py gradient
refinement, archive.py pooling, pipeline.py jobs); run.py runs experiments. Read the
README first.

Rules:
1. Do NOT modify LINKS/, submissions/best.npy, submissions/best_score.json, or the
   committed experiments_*.csv by hand. Use --no-update-best on every run. Measure what
   a run would add with `python merge.py --dry-run <submission.npy>`. Don't merge into
   best.npy until I say so.
2. Put new behavior behind a setting whose default is today's behavior, so existing
   tests still pass. Add a small test for the new option. Run `ruff check linkopt tests`
   and `pytest -m "not slow"` before any long run.
3. Focus on Kangaroo 2 (targets=1, the "Problem 2" key) unless the prompt says otherwise.
4. Compare arms fairly: same seeds and roughly the same compute in every arm. Change one
   thing at a time.
5. Report back with:
   - a table per arm: Kangaroo 2 hypervolume of the run alone, Kangaroo 2 best (lowest)
     distance among valid designs, wall-clock time, and the Kangaroo 2 score change
     from merge.py --dry-run
   - 3-5 plain-language sentences on what happened and why, suitable for my reflection
     report (I'm new to GAs and gradient descent).
Before any run longer than 15 minutes, show me the --dry-run job list and time estimate.
```

### 1. Mutation-rate sweep — ~30 min · Impact 2/5

**Why a 2:** this answers the "modify mutation" hint with zero code, and you've never varied it: all 3,314 logged jobs used the default. But the professors and your own `ga.py` note diminishing returns, so expect a small effect.

```text
Experiment: does the GA's mutation rate matter for Kangaroo 2? No code changes needed.

Run:
python run.py --preset full --targets 1 --n-joints 7 --n-gen 300 --grad-steps 250 \
  --seeds 0-9 --sweep mutation_prob=none,0.3,0.7 --no-update-best

Then:
- From the run's jobs.csv, compare the three mutation_prob arms: median and best
  hv_refined, plus each arm's best distance (compute it from the job files).
- Pool each arm separately into its own submission file and run merge.py --dry-run
  on each, to see which arm would add the most to best.npy.
- Explain in simple terms what mutation_prob controls in our ga._mating, and why
  pymoo's link switches and joint positions might respond differently.
```

### 2. Scale sweep of existing designs — ~30 min · Impact 1/5

**Why a 1:** distance is measured without rescaling the curve, so shrinking a design slightly cuts its material while its distance barely changes. That fills the gaps between mechanism families almost for free, but those gaps are worth only hundredths of hypervolume.

```text
Experiment: "scale sweep" of best.npy's Kangaroo 2 designs (gap filling).

Build a small script scripts/scale_sweep.py (don't change linkopt/ behavior):
- Load best.npy's Problem 2 designs.
- For each design, make copies with every joint position multiplied by
  s in [0.80, 0.85, 0.90, 0.95, 0.97, 1.03, 1.05]. Links and other fields unchanged.
- Score all copies in batches with linkopt.problem.evaluate against Kangaroo 2.
  Keep only valid designs (inside the limits, using problem.safe_limits).
- Save them as a submission file in runs/exp-scale/ (use linkopt.submission to build
  and save it; fill the other problems with best.npy's designs so the file is valid).
- Run merge.py --dry-run on it.
Also make one plot: the current Kangaroo 2 staircase with the new scaled points
overlaid. Explain why scaling changes material linearly but distance only a little.
```

### 3. Smarter gradient descent and basin hopping — ~1 h · Impact 3/5

**Why a 3:** it works directly on your most accurate designs, which carry most of the score. Your logs show those designs are stuck at the bottom of local valleys: refinement's gain fell from +0.73 to +0.07 as GA runs got longer, and 1,000 vs. 3,000 steps made no difference. It's cheap to try, but the gain is likely modest.

```text
Experiment: can a smarter gradient method lower Kangaroo 2's best distances?
Hint addressed: "Are there smarter gradient-based optimization methods?"

Add to refine.py (keep the default unchanged), via a new Config setting
refine_method in {"plain", "adam", "basin"}:
- "adam": the Adam update on joint positions using the distance gradient (lr around
  1e-4, beta1=0.9, beta2=0.999), with gradient clipping, and the same "stay inside the
  limits, keep each design's best position" rules as today.
- "basin": basin hopping. For each design, repeat R=5 times:
    1. add Gaussian noise (sigma = 0.01) to the movable joints
    2. run today's plain descent
    3. keep the result only if its distance improved
  Use a fixed RNG seed.

Run all three on best.npy's Problem 2 designs (like --refine-best, no GA). Give every
method the same total number of gradient calls (about 2,000 per design). Report per
method: how many designs improved, the median and best distance drop, and the
merge.py --dry-run Kangaroo 2 change. Pay special attention to the most accurate
designs (material above ~2.4 in grader units).

Explain in simple terms what Adam and basin hopping do differently from plain
gradient descent.
```

### 4. Gradient descent using both distance and material — ~45 min · Impact 2/5

**Why a 2:** it answers the "gradients of both functions" hint directly and makes a good report figure. But moving toward less material mostly helps the low-value left side of the front, so the hypervolume gain will be small.

```text
Experiment: refinement using both gradients (distance and material).
Hint addressed: "Can you use the gradients of both functions?"

In refine._descend, add an option (default keeps distance-only) for a weighted
direction:
  g = w * dgrad/||dgrad|| + (1-w) * mgrad/||mgrad||
Normalize each gradient per design before mixing; guard against zero norms.
Also add an option to save a snapshot of each design every 25 steps (not only its
best), so one design traces out several trade-off points.

Run on best.npy's Problem 2 designs, with w in {1.0, 0.9, 0.7, 0.5}, snapshots on,
step size 1e-4, 1,000 steps. For each w, report the merge.py --dry-run Kangaroo 2
change and the number of new non-dominated points. Make one plot with the staircase
before and after (best w).

Explain why lower w mostly adds cheap designs, and why that matters less for
hypervolume here than lowering the best distance.
```

### 5. Screened starting population — ~1.5 h · Impact 3/5

**Why a 3:** your 10- and 12-joint jobs found no valid design in 76–100% of runs, so starting quality clearly limits bigger mechanisms. For 7 joints, which already find valid designs, the gain is less certain.

```text
Experiment: does a better starting population help the GA?
Hint addressed: "Why not do some preprocessing to random mechanisms before running GA?"

Add a Config setting start_oversample (default 1 = today). In
ga.make_start_population:
- generate n_start * start_oversample random mechanisms
- score them in one batch with problem.evaluate
- keep the n_start with the lowest distance (break ties by lower material)
Optional second setting start_refine_steps (default 0): run that many plain gradient
steps on the kept starters before the GA.

Run:
- Kangaroo 2, 7 joints, 300 generations, grad_steps 250, seeds 0-9
- --sweep start_oversample=1,5,25
- also 8 joints with oversample 1 vs 25 (same seeds)
- --no-update-best

Report per arm: generation of the first valid design (record it with run_ga's
callback), hv_refined median and max, best distance, and the merge.py --dry-run
change. Note the extra time screening costs.
```

### 6. Material-band runs (ε-constraint) — ~2–3 h · Impact 3/5

**Why a 3:** it focuses the search on the moderate-material, high-accuracy region where new designs are worth the most. It's also the "material as a constraint" idea done properly. Whether it beats the 0.465 step depends on the GA finding better mechanisms, which isn't guaranteed.

```text
Experiment: focus the GA on material bands (the epsilon-constraint method, L05).

Add Config settings material_min and material_max (defaults: no band = today). In
MechanismProblem, add the band as extra constraints, keeping both objectives
(NSGA-II still spreads designs within the band):
  material >= material_min and material <= material_max
Make sure the job id and logs record the band.

Run Kangaroo 2, 7 joints, 400 generations, seeds 0-9, with these bands (grader
units, as in our plots): [2, 3], [3, 4.5], [4.5, 6], plus no band as the control.

Diagnostic arm: one extra arm where the GA minimizes distance only (single
objective; material <= 10 as a constraint), 7 joints, same seeds. This tests whether
ignoring material finds anything below today's best distance of about 0.465.

Report per band: best distance reached inside the band, and the merge.py --dry-run
Kangaroo 2 change from pooling each band's designs. Explain in plain words why we
banded material instead of only minimizing distance.
```

### 7. Warm-start GA from `best.npy` (cycling) — ~2–3 h · Impact 4/5

**Why a 4:** the GA always keeps the best of parents and children (elitism), so starting from your best designs can't make it worse. It starts the search beside designs that already work. The spare unconnected joints also let the GA grow four-bars into six-bars, which hold Kangaroo 2's accurate end.

```text
Experiment: warm-start the GA from best.npy (cycle GA -> refine -> pool -> GA).
Hint addressed: "Can you cycle through multiple optimization runs?"

Add a Config setting warm_start_fraction (default 0 = today: all random starters).
When > 0:
- That fraction of the starting population comes from best.npy's designs for that
  kangaroo, converted with problem.from_mech.
- Designs with fewer joints than n_joints are padded with extra UNCONNECTED joints at
  random positions in [0,1] (best.npy already contains valid designs with isolated
  joints).
- Designs with more joints than n_joints are skipped.
- Each copy after the first gets small Gaussian noise (sigma 0.01) on its positions.
- The rest of the population comes from MechanismRandomizer as today.

Check that a padded design scores exactly the same as the original before the GA
starts; add a test for that.

Run Kangaroo 2:
- 7 joints, 300 generations, seeds 0-9, --sweep warm_start_fraction=0,0.5,1.0
- then 8 joints with 0 vs 0.5
Report the merge.py --dry-run Kangaroo 2 change per arm, best distance, and how many
of the run's final designs have more links than the design they started from.
```

### 8. Positions-only GA on the best shapes — ~3 h · Impact 4/5

**Why a 4:** once a mechanism's shape is fixed, a 7-joint search shrinks from 42 mixed variables, mostly link switches that break the mechanism when flipped, to 14 smooth position numbers. That's much easier territory for crossover and mutation. Aiming it at the most accurate shapes targets the part of the front that sets the score.

```text
Experiment: mix two kinds of GA. The existing mixed GA finds good mechanism SHAPES;
a new positions-only GA fine-tunes the best shapes.
Hint addressed: "Can you mix different kinds of GA?"

Build linkopt/position_ga.py:
- A pymoo Problem whose only variables are the 2N joint coordinates (Real, bounds
  [0,1]) for ONE fixed design: edges, fixed_joints, motor and target_joint held
  fixed.
- Two objectives (distance, material), with constraints from the kangaroo's
  reference point.
- NSGA-II with SBX(eta=15) and PM(eta=20), standard tournament selection, pop 100.
- Starting population: the design's own positions plus noisy copies (sigma 0.01-0.03).
- After the GA, run today's refine on its final front.
- Wire it into run.py as a new job kind, "position_ga", that picks the top-K shapes
  from best.npy for a kangaroo, ranked by lowest distance (so the most accurate
  shapes come first).

Run Kangaroo 2 with K=15 shapes, 200 generations, 3 seeds each.
Control: spend the same wall-clock time on extra mixed-GA seeds (7 joints, 300 gens).
Report the merge.py --dry-run change for each, and per shape the best distance
before -> after. Explain in simple terms why a smaller, all-continuous search can
work better.
```

### 9. Bigger mechanisms, much longer runs — overnight · Impact 4/5

**Why a 4:** this has the strongest evidence of anything here.
- Median Kangaroo 2 hypervolume for 6-joint jobs rose from 3.84 to 4.66 to 5.17 as generations went from 150 to 300 to 600.
- Your best Kangaroo 2 job ever was 6 joints at 600 generations.
- On Kangaroo 3, 8–10 joints at 600 generations produced the top jobs.

Kangaroo 2 has never had 8 joints run long. The cost is a night of compute.

```text
Experiment: do bigger mechanisms with much longer GA runs beat today's best Kangaroo 2
accuracy? No new code, except combining it with #5 or #7 if those already helped.

Step 1 (~45 min):
python convergence.py --targets 1 --n-joints 7 8 --seeds 0-2 --n-gen 1500 \
  --grad-steps 0 250
Tell me where hypervolume stops rising for 7 and 8 joints.

Step 2 (overnight), with n_gen set from step 1 (likely 1000-1500):
caffeinate -i python run.py --preset full --targets 1 --n-joints 7 8 \
  --n-gen <from step 1> --grad-steps 250 --seeds 3000-3029 --no-update-best

Report: 90th-percentile and max hv_refined per joint count, best distance per joint
count, and the merge.py --dry-run Kangaroo 2 change. Compare against our 6-joint,
600-generation jobs in experiments_jobs.csv. Judge by the best jobs and the pooled
gain, not the median. Explain why the median is the wrong metric when pooling.
```

### 10. Dyad-based representation — 1 day+ · Impact 4/5 (risky)

**Why a 4 but risky:** this has the highest ceiling. Today's link-switch encoding has about 134 million combinations at 7 joints, and most aren't a valid mechanism one motor can drive. Building each new joint by attaching it to two earlier joints (a dyad), which is how `MechanismRandomizer` builds valid mechanisms, makes nearly every child usable. But it's a lot of new code and testing before Wednesday night. Run it only if 1–9 are done; otherwise describe it as future work in the report.

```text
Experiment: a more efficient mechanism representation (dyad-based encoding).
Hint addressed: "Is there a more efficient representation of mechanisms?"

Read LINKS/Optimization/_MechanismRandomizer.py to see how it builds valid mechanisms
(each new joint connects to earlier joints). Build linkopt/dyad_problem.py: a pymoo
problem where a mechanism of N joints is encoded as:
- the base four-bar's joint positions
- for each extra joint i: two Integer "parent" choices among earlier joints, a Binary
  fixed/free flag, and its (x, y) position
- an Integer target joint
Decoding must always produce a one-degree-of-freedom mechanism in our submission
format. Add round-trip tests (encode -> decode -> same score) and a test that 1,000
random encodings decode to valid one-degree-of-freedom graphs. Reuse problem.evaluate
for scoring.

Compare against the current MechanismProblem on Kangaroo 2, 7 and 8 joints, 300
generations, seeds 0-9, same population size. Report:
- the share of valid children per generation
- the generation of the first valid design
- hv_refined median and max
- the merge.py --dry-run change
Explain in plain words why this encoding wastes fewer evaluations.
```

## 6. Running the experiments fairly

- **Change one thing at a time** so you know what caused the effect.
- **Use the same seeds in every arm.** Then a difference comes from your change, not from luck. This is called a paired comparison.
- **Give every arm the same compute** (same evaluations or same minutes). Otherwise the arm that ran longer looks better for that reason alone.
- **Judge by how much an arm adds to `best.npy`** (`--no-update-best`, then `merge.py --dry-run`), and by each job's best distance. Not by the median job: the pooled score comes from the best designs.
- **Log each job's best distance** alongside its hypervolume. It's the single number that drives most of the score.
