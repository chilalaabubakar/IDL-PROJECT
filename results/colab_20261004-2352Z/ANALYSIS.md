# First full Colab run: analysis and next steps

Run: `results/colab_20261004-2352Z/` (pushed by the notebook on 2026-10-04). 256 samples
per row, DM2 sampler with 2,900 noisy + 100 final steps, every sample relaxed with FIRE
before scoring, 95% bootstrap CIs over samples. Training: unconditional EGNN-PBC 40,000
updates (3.0 h), conditional EGNN-PBC 50,000 updates (3.7 h). Validation loss was still
falling slowly at the end (σ = 0.02: 0.30 → 0.28 σ² from the midpoint to the end), so
longer training would help a little but is not the main issue. The logs show 0.27 s per
update, 7× slower than the A100 speed test (0.04 s), so training probably ran on a smaller
GPU; on an A100 a retrain takes about 30 min.

The diagnostics in §3 were run afterwards on a CPU with the pushed `ema.pt` weights.

## 1. Task B (generate around a requested site inside an existing glass): the headline

Success = the requested defect is present at the requested site **after relaxation**.
Local W1 = distance between the per-atom energy distribution near the site and that near
natural defects (lower is better; the floor is natural vs natural).

**D1− (under-coordinated A atom)**

| Method | Success after relax | Before relax | Local W1 (PE) | s / success |
|---|---|---|---|---|
| Conditional EGNN, w = 2 | **0.22 [0.17, 0.27]** | 0.66 | **0.019 [0.014, 0.026]** | 21 |
| Conditional EGNN, w = 0 | 0.09 [0.06, 0.12] | 0.16 | 0.054 [0.046, 0.061] | 27 |
| RePaint, unconditional (no label: chance) | 0.02 [0.00, 0.03] | 0.05 | 0.053 [0.048, 0.059] | 163 |
| Hand insertion (B2) | 0.06 [0.04, 0.09] | 0.34 | 0.032 [0.026, 0.039] | 3.6 |
| Local melt-quench (B6) | 0.00 [0.00, 0.00] | 0.01 | 0.024 [0.019, 0.031] | ∞ |
| Floor (natural vs natural) | — | — | 0.016 [0.011, 0.022] | — |

**D2 (B atom with ≥ 2 B neighbours)**

| Method | Success after relax | Before relax | Local W1 (PE) | s / success |
|---|---|---|---|---|
| Conditional EGNN, w = 2 | **0.36 [0.30, 0.42]** | 0.66 | **0.021 [0.017, 0.027]** | 13 |
| Conditional EGNN, w = 0 | 0.23 [0.18, 0.28] | 0.36 | 0.031 [0.026, 0.037] | 11 |
| RePaint, unconditional (no label: chance) | 0.03 [0.01, 0.05] | 0.03 | 0.062 [0.058, 0.067] | 92 |
| Hand insertion (B2) | 0.30 [0.24, 0.36] | 1.00 | 0.087 [0.077, 0.095] | 0.8 |
| Local melt-quench (B6) | 0.05 [0.03, 0.08] | 0.05 | 0.065 [0.056, 0.074] | 65 |
| Floor (natural vs natural) | — | — | 0.013 [0.009, 0.019] | — |

**D1+ (over-coordinated A atom):** every method ≤ 0.05 after relaxation (conditional
w = 2: 0.04 [0.02, 0.07]; hand insertion 0.02; local melt-quench 0.01), although
patch-based methods start at 1.00 before relaxation.

What this says:

1. **The conditional model with guidance is the only method that both places the defect
   and keeps the surroundings natural.** For D1− it succeeds about 3.5× as often as hand
   insertion and 10× as often as chance, with local realism at the natural floor (the CIs
   overlap). For D2 it matches hand insertion on success (0.36 vs 0.30, CIs overlap) and is
   4× closer to natural.
2. **Guidance matters and costs no realism.** w = 0 → 2 raises success 2.5× (D1−) and
   1.6× (D2), and the W1 distance *drops*. The sweep beyond w = 2 is still to do.
3. **Hand insertion is the cheapest per success** (it is only a relaxation), so the cost
   claim of RQ3 holds against local melt-quench but not against hand insertion. The honest
   framing is cost per *realistic* success.
4. **D1+ fails for every method.** See §3.1: natural D1+ sites are stable, so the problem
   is generating their surroundings, not the definition.

## 2. Task A (generate the whole box with a defect at the centre)

| Defect | Conditional w = 2 | Conditional w = 0 | Clamp patch (B4) | RePaint patch (B5, U = 1) | Unconditional (B1) |
|---|---|---|---|---|---|
| D1− success | **0.30 [0.24, 0.35]** | 0.09 | 0.04 | 0.07 | 0.01 |
| D2 success | 0.39 [0.33, 0.45] | 0.22 | 0.38 [0.32, 0.43] | 0.36 [0.30, 0.41] | 0.00 |
| D1+ success | 0.04 | 0.05 | 0.05 | 0.05 | 0.00 |
| Local W1 (PE), D2 | 0.055 | 0.063 | 0.052 | 0.050 | 0.087 |

The conditional model clearly wins on D1− and ties the patch methods on D2. But **the
generated glass itself is off**, which makes every Task A number suspect:

| | Generated (B1, 2,900 steps) | Natural test glasses |
|---|---|---|
| Mean energy per atom | −3.720 | −3.680 (slow quench: −3.706) |
| g_AB(r) L1 distance to test | 0.49 | 0.067 (the floor) |
| Nearest B atom to the centre is a D2 site | 0 of 256 | ~11% of B atoms are D2 |

The relaxed samples are *more stable than the training glasses*, more so even than the
10× slower quench. The figures show A-rich hexagonal patches and isolated B atoms, so
the glass is chemically ordered. The global |Ψ6| gate misses this because the patches are
local. The MVP gate "global g(r) within 2× the floor" is not met (7×).

## 3. Diagnostics (CPU, after the run)

### 3.1 Natural defects survive noise plus relaxation

Every atom of 100 natural test glasses was displaced by Gaussian noise σ, then the glass
was relaxed and each natural defect site checked with the success test
(`scripts/defect_stability.py`):

| σ | D1− (155 sites) | D1+ (135) | D2 (1,010) |
|---|---|---|---|
| 0.03 (first 40 glasses: 61 / 47 / 383 sites) | 1.00 | 1.00 | 1.00 |
| 0.06 | 1.00 | 1.00 | 1.00 |
| 0.10 | 1.00 | 1.00 | 1.00 |
| 0.15 | 1.00 | 0.99 | 0.99 |

Relaxation brings the atoms back to where they were: the mean displacement is 1.25σ
(0.125 at σ = 0.1), so σ = 0.1–0.15 means displacements of 0.125–0.19. Atoms near
generated defects move 0.14–0.19 during relaxation (`local.w1_relax_disp`), just as far,
yet the natural glasses keep 99–100% of their defects. **So the before → after success gap (e.g. 0.66 → 0.22) is the generator's
fault:** generated defects sit in different, shallower basins than natural ones. This
also rules out "D1+ is too fragile to place".

### 3.2 The number of sampler steps works like a cooling rate

The unconditional model was sampled on the CPU with fewer noisy steps (+100 final steps
each), then relaxed:

| Noisy steps | Samples | Energy / atom | B–B contacts per B | D2 / glass | D1− / glass | D1+ / glass | g_AB L1 |
|---|---|---|---|---|---|---|---|
| 100 | 16 | **−3.679** | 0.37 | 4.4 | 2.3 | 0.7 | 0.26 |
| 300 | 16 | −3.697 | 0.28 | 2.3 | 1.7 | 0.4 | 0.34 |
| 2,900 (Colab) | 256 | −3.720 | — | — | — | — | 0.49 |
| Natural test | 100 | −3.680 | 0.56 | 10.1 | 1.6 | 1.4 | 0.11–0.14 at 16 glasses |
| Natural, 10× slower quench | 200 | −3.706 | 0.48 | 8.7 | 1.1 | 1.3 | 0.13 |

The DM2 sampler adds fresh noise at every step, so a longer schedule is a slower anneal:
energy and B–B contacts fall as steps increase. **100 steps reproduce the natural
energy**, at 1/15 of the cost (200 steps in total instead of 3,000). B–B contacts stay at two-thirds of the natural
value even at 100 steps, so part of the chemical-order bias is in the model, not the
schedule. This finding is reportable on its own: DM2's default schedule over-anneals a
glass made at the dataset's cooling rate.

## 4. Hypotheses so far

| | Status |
|---|---|
| H2: trained conditioning beats chance | **Supported** on D1− and D2, both tasks. |
| H1: RePaint is more realistic than clamping | Untested: U = 1 is no resampling. D1− 0.071 vs 0.101 (CIs disjoint); D2 and D1+ tie. Needs the U sweep. |
| H3: unconditional is the most realistic | Rejected as run, confounded by the over-annealed backbone (§2). Re-test after fixing the step count. |
| RQ3: cheaper per success | Beats local melt-quench; loses to hand insertion on raw cost, wins on cost per realistic success. |

## 5. Next steps, in order

1. **Round 2 on Colab** (new cells in `notebooks/colab_train.ipynb`, about 3 h on an A100;
   finished runs are skipped, so it can be split across sessions):
   sampler steps 100 / 300 / 900 for B1 and the conditional model on both tasks;
   guidance w = 0, 1, 4, 6; RePaint U = 1, 5, 10 against Clamp; the defect-stability
   control. The summary now also shows energy, g(r) distance and B–B contacts.
   **Decision rule:** adopt the step count whose B1 energy and B–B contacts are closest to
   natural while conditional success after relaxation does not drop; use it for everything
   after.
2. **If B–B contacts stay low:** retrain with `noise.sigma_max=1.0` (the sampler starts at
   σ = 1.0, but the model never saw σ > 0.5) and compare.
3. **D1+:** keep it in the tables as the hard case. The natural sites are stable, but a
   1.5σ patch does not carry what holds them, so try a larger patch radius (2.5σ) before
   spending more compute.
4. **Stage 4, the symmetry ladder at N = 64** (MLP → MPNN → EGNN on `ka2d_64`): three
   short training runs, after the step count is fixed.
5. **Stage 0 (B-1):** reproduce DM2's pretrained SiO₂ sample. Still open.
6. **Midterm report:** the Task B tables in §1 are ready to use; add round 2's guidance and
   step sweeps and the stability control as the failure analysis.
