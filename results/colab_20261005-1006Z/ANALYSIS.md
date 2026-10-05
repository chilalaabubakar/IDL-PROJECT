# Round 2: sampler steps, guidance, RePaint resampling, stability control

Run: `results/colab_20261005-1006Z/` (34 new evaluations; `summary.csv` also holds round 1).
Same models as round 1, 256 samples per row, every sample relaxed before scoring, 95%
bootstrap CIs. Natural references: energy −3.681 per atom, B–B contacts 0.56 per B atom,
g_AB(r) floor 0.067, local W1 (PE) floor 0.016 (D1−) and 0.013 (D2).

## 1. Sampler steps: the cooling-rate effect is confirmed

**Unconditional model (B1), whole box:**

| Noisy steps | Energy / atom | g_AB L1 | B–B per B | Seconds (256 samples) |
|---|---|---|---|---|
| 100 | **−3.678** | **0.23** | 0.36 | 118 |
| 300 | −3.692 | 0.30 | 0.31 | 134 |
| 900 | −3.706 | 0.39 | 0.25 | 262 |
| 2,900 (round 1) | −3.720 | 0.49 | — | 643 |

On 256 samples, the trend matches the CPU check: every extra step cools the glass further.
100 steps give the natural energy and halve the g(r) error. B–B contacts stay at
two-thirds of the natural value even then, and g(r) is still 3.5× the floor, so the MVP
gate (2× the floor) is still not met.

**Conditional model, w = 2:**

| Task | Steps | D1− success | D2 success | Local W1 D1− / D2 | g_AB L1 | s / success D1− / D2 |
|---|---|---|---|---|---|---|
| A (box) | 100 | 0.17 [0.12, 0.22] | 0.41 [0.35, 0.46] | **0.026 / 0.018** | 0.21 | 3.5 / 1.4 |
| A | 300 | 0.17 [0.13, 0.22] | 0.34 [0.28, 0.40] | 0.030 / 0.032 | 0.27 | 5.7 / 2.8 |
| A | 900 | 0.19 [0.14, 0.24] | 0.45 [0.39, 0.51] | 0.049 / 0.046 | 0.35 | 9.7 / 3.9 |
| A | 2,900 | 0.30 [0.24, 0.35] | 0.39 [0.33, 0.45] | 0.055 / 0.055 | 0.43 | 16 / 12 |
| B (host) | 100 | 0.22 [0.17, 0.27] | 0.35 [0.29, 0.41] | 0.021 / 0.040 | 0.08 | 2.5 / 1.6 |
| B | 300 | 0.23 [0.18, 0.29] | 0.30 [0.25, 0.36] | 0.012 / 0.044 | 0.08 | 4.4 / 2.9 |
| B | 900 | 0.23 [0.18, 0.29] | 0.35 [0.29, 0.41] | 0.014 / 0.026 | 0.07 | 7.3 / 4.9 |
| B | 2,900 | 0.22 [0.17, 0.27] | 0.36 [0.30, 0.42] | 0.019 / 0.021 | 0.07 | 21 / 13 |

- **Task B success does not depend on the step count**, so 100 steps cost 8× less for the
  same result. D1− realism is at the floor at every step count. D2 realism is best at
  900–2,900 steps (0.021–0.026 vs 0.040 at 100).
- **On Task A, fewer steps give much more realistic surroundings** (W1 0.018–0.026 at 100
  steps, close to the floor, against 0.055 at 2,900). D2 success holds, but D1− success
  drops from 0.30 to 0.17: the over-annealed box made under-coordinated sites easier to
  keep.

**Decision: 100 steps from now on.** It gives natural energy and the best Task A realism,
leaves Task B success unchanged, and costs least. The one trade-off is D2 realism on
Task B.

## 2. Guidance strength (Task B, 300 steps)

| w | D1− success | before relax | D2 success | before relax | Local W1 D1− / D2 | Diversity (D2) | s / success D1− / D2 |
|---|---|---|---|---|---|---|---|
| 0 | 0.09 [0.06, 0.12] | 0.19 | 0.24 [0.19, 0.30] | 0.32 | 0.026 / 0.039 | 1.08 | 5.8 / 2.1 |
| 1 | 0.17 [0.13, 0.22] | 0.46 | 0.31 [0.26, 0.37] | 0.58 | 0.015 / 0.031 | 1.05 | 5.2 / 2.6 |
| 2 | 0.23 [0.18, 0.29] | 0.59 | 0.30 [0.25, 0.36] | 0.73 | 0.012 / 0.044 | 1.03 | 4.4 / 2.9 |
| 4 | 0.21 [0.16, 0.26] | 0.77 | 0.47 [0.41, 0.54] | 0.80 | 0.015 / 0.045 | 1.04 | 4.1 / 1.9 |
| 6 | 0.23 [0.18, 0.29] | 0.85 | **0.56 [0.50, 0.62]** | 0.97 | 0.017 / 0.048 | 1.03 | 3.6 / 1.5 |

- **D2 keeps improving with guidance.** At w = 6, 0.56 success is almost twice hand
  insertion (0.30 [0.24, 0.36], CIs disjoint), and the surroundings stay closer to natural
  (W1 0.048 against 0.087).
- **D1− saturates at w = 2.** More guidance puts the defect there before relaxation (0.85
  at w = 6), but the share that survives relaxation stays at ~0.22. The extra D1− sites
  sit in shallow basins.
- **No mode collapse.** The diversity spread ratio stays at 1.0 (natural = 1) and the
  nearest-training-window ratio also stays near 1, so the model is not copying training
  windows.
- **Cost:** at w = 6 the model needs 3.6 s per D1− success (hand insertion 3.6 s, with
  a quarter of the success rate) and 1.5 s per D2 success (hand insertion 0.8 s). These
  are at 300 steps; at 100 steps they drop by about another 40%.

## 3. RePaint resampling against Clamp (Task A patch requests, 300 steps)

| Method | Success D1− / D2 | Local W1 D1− | Local W1 D2 | Energy | g_AB L1 | Seconds |
|---|---|---|---|---|---|---|
| Clamp | 0.05 / 0.34 | 0.081 [0.074, 0.087] | **0.028 [0.024, 0.034]** | −3.689 | 0.26 | 157 |
| RePaint U = 1 | 0.02 / 0.34 | **0.056 [0.050, 0.062]** | 0.039 [0.035, 0.044] | −3.688 | 0.26 | 136 |
| RePaint U = 5 | 0.05 / 0.37 | 0.071 [0.065, 0.077] | 0.044 [0.039, 0.048] | −3.704 | 0.37 | 376 |
| RePaint U = 10 | 0.04 / 0.36 | 0.077 [0.072, 0.084] | 0.049 [0.044, 0.053] | −3.712 | 0.42 | 697 |

**H1 (RePaint is more realistic than clamping) gets a split verdict.** RePaint wins on D1−
and Clamp wins on D2, each with disjoint CIs. Resampling does not help: every resampling
pass re-noises and re-runs part of the schedule, so U = 5 and U = 10 cool the glass
further (energy −3.704 and −3.712, g(r) worse). On the score-dynamics sampler, RePaint's
resampling turns into extra annealing. Success does not change with U.

## 4. Control: natural defects under noise and relaxation (GPU, 100 test glasses)

| σ | Mean displacement | D1− (155 sites) | D1+ (135) | D2 (1,010) |
|---|---|---|---|---|
| 0.05 | 0.063 | 1.00 | 1.00 | 1.00 |
| 0.10 | 0.124 | 1.00 | 0.99 | 0.99 |
| 0.15 | 0.187 | 0.99 | 0.99 | 1.00 |
| 0.20 | 0.262 | 0.68 | 0.69 | 0.85 |
| 0.30 | 0.482 | 0.08 | 0.06 | 0.29 |

Natural defects hold until atoms are displaced by ~0.19 on average, then fall off
sharply. Generated samples move 0.16–0.24 during relaxation (`local.w1_relax_disp`),
which is at the edge of that cliff. Even there, natural defects keep 68–100%, while
generated ones keep about 25–60% (success after / before relaxation). The before → after
gap is mostly the generator's.

## 5. D1+ with larger patches (CPU, 64 hand insertions each)

Natural D1+ patches of growing radius were inserted into test hosts, then relaxed:

| Patch radius | Atoms per patch | Success before relax | After relax |
|---|---|---|---|
| 1.5 (as in the library) | 9 | 1.00 | 0.00 |
| 2.5 | 23 | 1.00 | 0.05 |
| 3.5 | 47 | 1.00 | 0.03 |

Even when a natural D1+ site is transplanted with its 47 nearest atoms, it relaxes away
97% of the time. Natural D1+ sites, by contrast, survive noise in their own glass (§4). So
an over-coordinated A atom is held in place by its host beyond 3.5σ, probably by local
compression, and nothing local can carry it: not a patch, not a pinned label. **D1+ goes
into the report as a limitation, with this experiment as the evidence**, rather than
getting more compute.

## 6. Where the main claims stand

| | Status |
|---|---|
| **Task B (headline)** | The conditional model beats hand insertion and local melt-quench on D1− (0.22 vs 0.06 vs 0.00) and D2 (0.56 at w = 6 vs 0.30 vs 0.05). Its D1− surroundings are at the natural floor, and its D2 surroundings are closer to natural than hand insertion's. |
| Task A | The conditional model has the most realistic surroundings at 100 steps, but the whole-box glass is still off (g(r) 3.5× the floor, too few B–B contacts). |
| H1 (RePaint > Clamp on realism) | Split: RePaint wins on D1−, Clamp on D2; resampling hurts. |
| H2 (trained > chance) | Supported. |
| H3 (unconditional most realistic) | Not supported: at 100 steps the conditional model's box is as natural as the unconditional one's (g(r) 0.21 vs 0.23, energy −3.674 vs −3.678), so conditioning costs no global realism. (Local W1 cannot decide this: B1's windows are not centred on defects.) |
| RQ3 (cost) | At 100–300 steps the model ties hand insertion per D1− success and is within 2× for D2, with 2–4× the success and closer-to-natural surroundings. |
| D1+ | Fails for every method, and larger patches do not help (§5): its stability depends on the host beyond 3.5σ. Reported as a limitation. |

## 7. Round 3 (notebook cells 3a–3c)

1. **Retrain with `noise.sigma_max=1.0`** (3a). The sampler starts at σ = 1.0, but the
   models never saw σ > 0.5, and that early phase is where atoms pick their neighbours.
   This is the most direct test of the B–B deficit. `sample.py` now clips σ at each
   checkpoint's own training σ_max, so the old and new models can be sampled side by side.
2. **The main table at 100 steps** (3b): w = 2, 4, 6 on both tasks for all three defects,
   Clamp and RePaint (U = 1) at 100 steps, and the retrained models at w = 2 and 6.
3. **Stage 4, the symmetry ladder at N = 64** (3c): flat MLP, MPNN and EGNN with the
   same recipe, sampled at 100 and 900 steps, against the N = 64 natural floor.

After round 3, the remaining planned items are Stage 0 (reproduce DM2's SiO₂ sample) and
the reports.
