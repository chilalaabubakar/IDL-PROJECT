# Gap 2, interim: how much does placement change on new host glasses?

Interim analysis of the two completed Gap 2 conditions (2 of 6), made with
`scripts/analyze_gap2_shift.py` on 2026-11-02:

- reference: fresh 256-atom glasses, standard cooling (`ka2d_256_standard`);
- main new condition: 1,024-atom glasses cooled 10 times more slowly (`ka2d_1024_slow`), the
  primary endpoint fixed in `docs/GAP2_GENERALIZATION.md`.

Source rows: `results/kaggle_gap2_kaggle_remaining_20261101-220126Z/study`. Same frozen
models (seeds 0-2), thresholds and sampling setting (w = 2, R_out = 6) in both conditions.
Primary denominator: requests whose site was not already D2. Intervals: 95%, whole-host
bootstrap, 1,000 draws; the two conditions have independent hosts and are resampled
separately.

## The new hosts are a different glass

| Per host | 256 standard | 1,024 slow |
|---|---|---|
| Energy per atom | -3.681 [-3.684, -3.678] | -3.702 [-3.703, -3.701] |
| Natural D2 per 256 atoms | 10.1 [9.4, 10.7] | 8.2 [7.9, 8.5] |
| Natural D1- per 256 atoms | 1.6 [1.3, 1.9] | 1.1 [1.0, 1.2] |
| Natural D1+ per 256 atoms | 1.5 [1.3, 1.8] | 1.3 [1.1, 1.4] |

The slowly cooled glasses sit lower in energy and hold about 19% fewer natural D2 sites, as
expected for better-annealed glass. (`hosts.csv` also lists global hexatic order, which falls
with system size and is not comparable across N.)

## Placement transfers; the margin over hand insertion narrows

| Success (initially non-D2) | 256 standard | 1,024 slow | Change |
|---|---|---|---|
| Model, seed 0 | 0.508 | 0.508 | -0.000 [-0.077, +0.081] |
| Model, seed 1 | 0.361 | 0.320 | -0.042 [-0.124, +0.053] |
| Model, seed 2 | 0.424 | 0.389 | -0.035 [-0.117, +0.048] |
| **Model, mean of seeds** | **0.431** | **0.406** | **-0.026 [-0.074, +0.026]** |
| Hand insertion | 0.206 | 0.279 | +0.073 [-0.001, +0.152] |
| Label removed (sham, seed 0) | 0.097 | 0.078 | -0.019 [-0.073, +0.036] |

| Advantage over hand insertion (same hosts and sites) | 256 standard | 1,024 slow |
|---|---|---|
| Seed 0 | +0.303 [+0.236, +0.377] | +0.230 [+0.139, +0.325] |
| Seed 1 | +0.155 [+0.073, +0.241] | +0.041 [-0.038, +0.119] |
| Seed 2 | +0.218 [+0.124, +0.310] | +0.111 [+0.033, +0.180] |
| Mean of seeds | +0.225 [+0.164, +0.292] | +0.127 [+0.061, +0.191] |

Change in the mean advantage: -0.098 [-0.196, -0.009].

- **Placement transfers without retraining.** No seed loses a detectable amount of success
  on 4 times larger, 10 times more slowly cooled glasses; the seed-mean change is -0.03 with
  an interval that includes zero and excludes losses larger than 0.07.
- **Every seed stays far above the label-removed sham**, so the label still drives placement.
- **The advantage over hand insertion shrinks** from 0.23 to 0.13 (still above zero for the
  mean, and for seeds 0 and 2; not for seed 1), mainly because hand insertion does better on
  these hosts. Which shift causes this, size or cooling history, cannot be told from these two
  conditions: `ka2d_256_slow` (running on Kaggle) and `ka2d_1024_standard` (PSC) separate them.

## Not covered yet

- Four conditions: 256 slow (Kaggle, running), 256 fast (Kaggle, queued), 1,024 standard and
  1,024 fast (PSC; results are collected with `scripts/collect_gap2.py`, not pushed).
- Structure of successful placements against natural D2 of the same condition, host
  displacement outside R_out, and survival in MD on the new hosts (the Gap 1 analyses,
  repeated per condition).

Re-run when more conditions are complete:

```bash
python scripts/analyze_gap2_shift.py --study <all study folders, newest last> --out results/gap2_shift_<date>
```
