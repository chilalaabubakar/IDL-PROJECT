# Round 3: wider noise range, final table at 100 steps, symmetry ladder

Runs: three Kaggle sessions (T4), pushed into `results/kaggle_20261006-*` and
`results/kaggle_20261012-*`. The newest `summary.csv` (`results/kaggle_20261012-1807Z/`)
covers all 110 experiments of rounds 1–3. All rows: 256 samples, 100 sampler steps, every
sample relaxed before scoring, 95% bootstrap CIs over samples.

"Old model" = the round 1 EGNNs (trained with noise σ ≤ 0.5). "Retrained" = the same
EGNNs retrained with σ ≤ 1.0, since the sampler starts at σ = 1.0.

> **Sensitivity checks** ([`sensitivity/README.md`](sensitivity/README.md)) confirm the
> method ranking below under every evaluation setting tried, and qualify two statements:
> the retrained model's w = 6 surroundings match hand insertion only at the 3σ window
> (better within 2.5σ, worse within 3.5σ), and the D1− w = 2 surroundings are within 1–2×
> of the floor depending on the defect cutoff, not always at it.

## 1. Task B (inside an existing glass): the headline table

**D2** (natural floor of the local W1: 0.013)

| Method | Success after relax | Before relax | Local W1 (PE) |
|---|---|---|---|
| Hand insertion (B2) | 0.30 [0.24, 0.36] | 1.00 | 0.087 [0.077, 0.095] |
| Local melt-quench (B6) | 0.05 [0.03, 0.08] | 0.05 | 0.065 [0.056, 0.074] |
| Old model, w = 2 | 0.35 [0.29, 0.41] | 0.67 | 0.040 [0.032, 0.048] |
| Old model, w = 4 | 0.43 [0.37, 0.49] | 0.87 | 0.055 [0.047, 0.065] |
| Old model, w = 6 | 0.54 [0.48, 0.60] | 0.94 | 0.064 [0.056, 0.074] |
| Retrained, w = 2 | **0.56 [0.50, 0.62]** | 0.86 | 0.056 [0.048, 0.066] |
| Retrained, w = 6 | **0.79 [0.74, 0.84]** | 1.00 | 0.089 [0.081, 0.099] |

**D1−** (floor 0.016)

| Method | Success after relax | Before relax | Local W1 (PE) |
|---|---|---|---|
| Hand insertion (B2) | 0.06 [0.04, 0.09] | 0.34 | 0.032 [0.026, 0.039] |
| Local melt-quench (B6) | 0.00 [0.00, 0.00] | 0.01 | 0.024 [0.019, 0.031] |
| Old model, w = 2 | 0.22 [0.17, 0.27] | 0.62 | **0.021 [0.014, 0.028]** |
| Old model, w = 4 | 0.21 [0.16, 0.26] | 0.70 | 0.027 [0.019, 0.036] |
| Old model, w = 6 | **0.25 [0.20, 0.31]** | 0.76 | 0.034 [0.026, 0.043] |
| Retrained, w = 2 | 0.21 [0.16, 0.26] | 0.59 | 0.023 [0.016, 0.031] |
| Retrained, w = 6 | 0.22 [0.17, 0.27] | 0.43 | 0.067 [0.058, 0.076] |

**D1+** (floor 0.018): every method stays at or below 0.04 (hand insertion 0.02, local
melt-quench 0.01, old model 0.03–0.04, retrained 0.01–0.04).

What this says:

1. **For D2, the retrained model is the clear winner.** At w = 2 it succeeds 1.9× as often
   as hand insertion (0.56 vs 0.30) with surroundings closer to natural (0.056 vs 0.087).
   At w = 6 it reaches 0.79 (2.6×) with surroundings as natural as hand insertion's. It
   places the defect every time (1.00 before relaxation), and about 80% of those survive
   relaxation, against about 57% for the old model. **The wider noise range fixed most of
   the "defects vanish on relaxation" problem for D2.**
2. **Guidance trades realism for success inside a glass.** For both models, stronger
   guidance raises D2 success and moves the surroundings away from natural. A report should
   show this as a trade-off curve, not a single number.
3. **For D1−, the old model is best.** Its success is about 0.21–0.25 at every guidance
   strength, against 0.06 for hand insertion and 0 for local melt-quench, with surroundings
   at the natural floor at w = 2. The retrained model does not help D1−, and at w = 6 it
   places fewer defects and makes worse surroundings.
4. **D1+ remains a limitation.** Round 2 showed why: even a natural D1+ transplanted with
   its 47 nearest atoms relaxes away 97% of the time.

## 2. Task A (whole box with the defect at the centre)

| Method | D2 success | D2 W1 | D1− success | D1− W1 | D1+ success |
|---|---|---|---|---|---|
| Unconditional (B1, chance) | 0.01 | 0.055 | 0.00 | 0.069 | 0.00 |
| Clamp patch (B4) | 0.34 | 0.022 | 0.01 | 0.053 | 0.04 |
| RePaint patch (B5, U = 1) | 0.35 | 0.045 | 0.02 | 0.042 | 0.04 |
| Old model, w = 2 | 0.41 | 0.018 | 0.17 | 0.026 | 0.02 |
| Old model, w = 4 | 0.51 | **0.011** | 0.18 | 0.026 | **0.08** |
| Old model, w = 6 | 0.61 | **0.010** | **0.23** | 0.034 | 0.05 |
| Retrained, w = 2 | 0.54 | 0.015 | 0.16 | 0.029 | 0.05 |
| Retrained, w = 6 | **0.84** | 0.046 | 0.16 | 0.092 | 0.01 |

Floors: D2 0.013, D1− 0.016, D1+ 0.018.

- In the whole-box task, guidance helps both success and realism up to w = 4–6 for the old
  model. **The old model at w = 6 gives D2 success 0.61 with surroundings at the natural
  floor (0.010)**, the cleanest single result of the project.
- The retrained model at w = 6 reaches 0.84 success, but its surroundings are no longer
  natural (0.046).
- For D1− and D1+ the patch baselines (Clamp, RePaint) almost never work; the conditional
  model is the only method with meaningful D1− success.

## 3. Did the wider noise range fix the glass itself? No.

| Unconditional, 100 steps | Energy / atom | g_AB L1 | B–B contacts per B |
|---|---|---|---|
| Old model | −3.678 | 0.23 | 0.36 |
| Retrained (σ ≤ 1) | −3.687 | 0.24 | 0.34 |
| Natural glasses | −3.681 | 0.067 (floor) | 0.56 |

The retrained model makes defects that survive relaxation much better (§1), but its
whole glasses have the same B–B deficit and the same g(r) error. The noise range is not the
cause of the chemical-order bias; it stays a reported limitation for Task A.

## 4. Stage 4: symmetry ladder at N = 64

Three unconditional denoisers trained with the same recipe on `ka2d_64` (40,000 updates;
T4 time: EGNN 44 min, MPNN 33 min, flat MLP 3.4 min), sampled from scratch, relaxed.

| Model | Steps | Energy / atom | g_AB L1 | B–B per B | Local W1 | Crystallized | Final val loss / σ² (σ = 0.1) |
|---|---|---|---|---|---|---|---|
| EGNN (full symmetry) | 100 | −3.626 | **0.21** | **0.32** | **0.044** | 0% | 0.23 |
| EGNN | 900 | −3.664 | 0.39 | 0.19 | 0.076 | 0% | |
| MPNN (permutation only) | 100 | −3.645 | 0.25 | 0.26 | 0.064 | 0% | 0.14 |
| MPNN | 900 | −3.677 | 0.41 | 0.15 | 0.085 | 0% | |
| Flat MLP (no symmetry) | 100 | −3.341 | 0.67 | 1.31 | 0.304 | 16% | 1.02 |
| Flat MLP | 900 | −3.351 | 0.65 | 1.30 | 0.296 | 18% | |
| Natural (N = 64) | | −3.636 | 0.094 (floor) | 0.475 | 0.007 (floor) | 0% | |

- **The flat MLP does not learn.** Its validation loss stays at the predict-zero level
  (1.02 σ²), so its samples are relaxed random configurations: energy far above natural,
  B atoms clumped (1.3 B–B contacts per B), 16–18% partly crystalline. Its 0.11 D2
  "success" is chance from the clumping, not placement.
- **The EGNN makes the most natural glasses.** It beats the MPNN at 100 steps on g(r),
  B–B contacts and local realism, even though the MPNN reaches a *lower* validation loss.
  Lower denoising loss does not mean better samples here; built-in rotation symmetry
  matters for sample quality.
- **The cooling-rate effect and the B–B deficit reproduce at N = 64** for both graph
  models (900 steps: lower energy, fewer B–B contacts), so they are properties of the
  sampler and the training objective, not of one system size or architecture.
- Caveats: one seed per model; the MLP's learning rate and size were not tuned for it, and
  the MPNN was trained without rotation augmentation.

## 5. Practical note: the stalled Kaggle sessions

Two Kaggle sessions stalled at the start of the MLP training and used up their time
without output. After the notebook was changed to send training output to a log file
(with a time limit per model), the same MLP trained in 3.4 minutes. The stall was most
likely the training process blocking on the notebook's output stream.

## 6. Status of the plan

| Item | Status |
|---|---|
| Stages 1–3 (MVP, training-free placement, trained conditioning) | Done |
| Stage 4 (symmetry ladder; step count; noise-aware model) | Done (σ-blind ablation not run) |
| Stage 0 (course notebook; DM2's SiO₂ sample) | Done (`results/stage0/`) |
| Sensitivity (defect thresholds ±0.05σ, R_out / R_loc) | Done: `results/sensitivity/README.md` |
| Reports | The tables in §1, §2 and §4 are final |

For publication, the main additions would be 2–3 training seeds for the two conditional
models and the scale/generalisation tests (`ka2d_1024`, `ka2d_256_slow`).
