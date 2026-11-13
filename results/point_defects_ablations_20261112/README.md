# Experiments on the point-defect finder

For the report on the two-dimensional study (`paper/point_defects_report.pdf`, Sections
"Experimental iterations" and "Ablations of the finder"). Every finder is scored on the same
750 real test defects: a defect is found when the atom the finder flags lies within 1.5σ of
it. Finders are compared in pairs on the same defects (McNemar's exact test), and intervals
resample whole test glasses.

- `scripts/pilots/finder_ablations.sh` trains the six ablation runs (about 4.5 hours on 4 CPU
  cores, resumable).
- `scripts/pilots/finder_ablations.py` scores every finder and writes `ablations.json` and
  `summary.md`.

## Ablations: one setting changed per run

Each run is `configs/train/detect_point_defects.yaml` with one change, trained for 2,000
updates (the finder of the report: 60,000). The rates are therefore lower than in the report's
main table, and the runs should be compared with each other only. One run per setting.

| Run | Parameters | Vacancy | Interstitial | Antisite | Frenkel pair | All 750 | p |
|---|---|---|---|---|---|---|---|
| **Reference:** starts from the denoiser, 4 layers, class weights | 0.52 M | 64% | 29.5% | 42.5% | 42% | **44.7%** [40, 49] | |
| Random starting weights | 0.52 M | 17% | 27.5% | 26.5% | 24% | 23.7% [20, 27] | < 0.001 |
| Random starting weights, 2 layers (reach 5σ) | 0.29 M | 15.5% | 27.5% | 25% | 24.7% | 23.1% [20, 27] | 0.46 |
| Random starting weights, 1 layer (reach 2.5σ) | 0.17 M | 13.5% | 23% | 15.5% | 17.3% | 17.3% [14, 20] | < 0.001 |
| No class weights | 0.52 M | 61% | 31.5% | 41% | 36.7% | 42.9% [39, 47] | 0.12 |
| Trained without the Frenkel pairs that are no defects | 0.52 M | 64% | 29.5% | 38% | 35.3% | 42.1% [38, 46] | 0.027 |

p: against the reference, or, for the runs with 1 and 2 layers, against the run with random
starting weights and 4 layers.

What it says:

1. **Starting from the denoiser matters most.** At the same number of updates it nearly
   doubles the defects found (23.7% → 44.7%). The gain is largest for vacancies (17% → 64%),
   the defect the denoiser alone already found best, and absent for interstitials. These
   short runs cannot say whether a network from random weights would catch up with much longer
   training.
2. **The finder has to see beyond the nearest neighbours.** One layer (reach 2.5σ) is clearly
   worse than two or four; two and four do equally well at this budget.
3. **Class weights change the labels, not the ranking.** Finding is the same within the noise
   (42.9% vs 44.7%). But without the weights the network calls nearly every atom intact: on
   validation glasses it labels 7% of the atoms next to a vacancy and no extra atom at all,
   against 27% and 18% with the weights (`*.log.jsonl`, last row).
4. **Cleaning the training set did not help.** Training without the 27% of Frenkel pairs that
   had undone themselves found no more defects (Frenkel pairs 35.3% vs 42%). The p value of
   0.027 compares two trained networks and does not cover the randomness of training: scored
   with the moving average of their weights the two runs are equal (44.5% and 44.8%). So the
   wrong labels are not what holds the finder back, at least early in training. This corrects
   the expectation written in `../point_defects_recheck_20261111/README.md`.

**Which weights are scored.** The ablation runs are scored with their trained weights, not
with the moving average of the weights that the long runs use. With a decay of 0.999, an
average over 2,000 updates is still 13% starting weights, which counts against a run that
starts from random weights (moving average: reference 44.8%, random start 13.3%, 2 layers
13.2%, 1 layer 17.6%, no class weights 44.3%, cleaned training set 44.5%; key
`moving_average` in `ablations.json`).

**Check of the faster CPU setup.** The runs use `model.k_max=32` instead of 64 (no atom has
more than 26 neighbours, so the padding is the only difference). The reference run's
validation scores at 2,000 updates equal those of the original Kaggle run at the same step
(vacancy 0.51, interstitial 0.30, Frenkel pair 0.40, antisite 0.37).

## The finders of the report

| Finder | Vacancy | Interstitial | Frenkel pair | Antisite | All 750 |
|---|---|---|---|---|---|
| After 20,000 updates | 67% | 47.5% | 36.7% | 46.5% | 50.3% [46, 54] |
| After 60,000 updates (the finder of the report) | 69.5% | 50% | 45.3% | 49% | **54.0%** [50, 58] |
| The same, not calibrated | 68.5% | 47.5% | 34.7% | 48% | 50.7% [47, 54] |

- **Training length.** 40,000 more updates at half the learning rate find 3.7 points more
  defects (49 defects found only by the longer run, 21 only by the shorter; p = 0.001). The
  gain is clear for Frenkel pairs (p = 0.015) and within the noise for each other type.
- **Calibration.** Subtracting the logarithm of the class weights from the finder's scores
  (logit adjustment) finds 3.3 points more defects (38 against 13; p < 0.001), nearly all of
  it in Frenkel pairs (34.7% to 45.3%, p = 0.002). It also cuts the defect-free glasses that
  are flagged from 68% to 2% (`../point_defects_recheck_20261111/glass_level.json`).

These repeat the comparisons of `../point_defects_finder_20261109/` and
`../point_defects_finder_v2_20261110/` on the real defects only (the earlier ones included the
50 Frenkel pairs that are no defects).

## Files

- `ablations.json`, `summary.md`: all rates, intervals and paired tests.
- `<run>.pt`: model settings and trained weights of each ablation run (readable by
  `glassdiff.models.registry.load_denoiser`); `<run>.config.yaml`: its settings;
  `<run>.log.jsonl`: its training log (loss and validation scores every 500 updates).
- `ablations.log`: when each run started. The full run folders (`runs/`) and the console logs
  are not kept in the repository.
