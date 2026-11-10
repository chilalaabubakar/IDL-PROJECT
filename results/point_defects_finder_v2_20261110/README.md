# A longer finder run: a small, real gain; the finder has now levelled off

The first finder (`../kaggle_pd_20261107-2024Z/training/pd_detect/`, 20,000 updates) was
continued for 40,000 more updates at a lower learning rate
(`configs/train/detect_point_defects_v2.yaml`, `notebooks/kaggle_point_defects_finder.ipynb`).
Everything else is unchanged: the same training data, model, test defects, calibration
(`--calibrate`) and repair routes as `../point_defects_finder_20261109/`.

Files:

- `../kaggle_pd_20261110-1042Z/training/pd_detect_v2/`: the checkpoint (`ema.pt`, with its
  class weights), config and training log. Training took 2.5 h on one T4.
- `../kaggle_pd_20261110-1230Z/eval_finder_v2.json`: the test evaluation
  (`scripts/eval_point_defects.py --calibrate --no-heal --no-energy-search`), 1.8 h.
- `paired_find.json`: the two finders compared defect by defect
  (`scripts/pilots/compare_finders.py`). A local CPU rerun gives exactly the finding rates of
  the Kaggle evaluation.

## Test results

The same 800 test defects (200 per type). Brackets: 95% Wilson intervals.

| Defect | Found: longer run | Found: first finder | Found: energy | Restored: longer run | Restored: first finder | Restored: energy route |
|---|---|---|---|---|---|---|
| Vacancy | **69.5%** [63, 75] | 67% | 29.5% | **53.5%** [47, 60] | 51% | 26% |
| Interstitial | **50%** [43, 57] | 47.5% | 28% | **38%** [32, 45] | 36% | 21% |
| Antisite | **49%** [42, 56] | 46.5% | 29% | **30%** [24, 37] | 27% | 13.5% |
| Frenkel pair | **37%** [31, 44] | 30% | 28.5% | 18% [13, 24] | 19.5% | 18.5%* |

\*Frenkel energy route: flag by energy, try every gap, keep the lowest energy
(`../point_defects_v2_20261109/`). Restored for interstitials is "remove, minimise"; for
antisites "swap, minimise", the pair picked by the finder.

For Frenkel pairs the longer run flags exactly the displaced atom 17% of the time (first
finder 14%) and picks the right gap 25% of the time (21.5%). That does not carry through to
restored glasses (18% vs 19.5%).

**Is the gain real?** Each row alone is within the noise of 200 defects, so the two finders
were compared on the same defects:

| Defect | Found only by the longer run | Found only by the first finder | McNemar p |
|---|---|---|---|
| Vacancy | 8 | 3 | 0.23 |
| Interstitial | 10 | 5 | 0.30 |
| Antisite | 12 | 7 | 0.36 |
| Frenkel pair | 21 | 7 | **0.013** |
| All 800 | 51 | 22 | **0.0009** |

The longer run finds more defects overall (51.4% vs 47.8%), and the gain is clear for Frenkel
pairs. For the other three types it points the same way in every case but is not significant
on its own.

**Is a defect-free glass told apart from a defected one?** Unchanged: the calibrated finder
flags 2% of defect-free glasses (first finder 1%) and 27% of defected ones (22%); AUC 0.73
(0.72).

## What this says

1. **More training helped a little, and that is now used up.** Finding improved by 2.5-7
   points, 3.6 points over all defects (p < 0.001, paired). Restored rates rose by 2-3
   points for vacancies, interstitials and antisites and fell by 1.5 for Frenkel pairs, all
   within the noise.
2. **The finder has levelled off.** On validation (200 defects per type, uncalibrated),
   finding moved little across the 40,000 extra updates (interstitials 39% → 44%, antisites
   47% → 49%, vacancies 54% → 53%, Frenkel pairs 41% → 39%), and over the last 20,000 by 2
   points or less. Further gains need a change of model or input, not more updates.
3. **This finder is the one to report.** It replaces the first finder in the main results:
   found 69.5, 50, 49 and 37%; restored 53.5, 38, 30 and 18% (vacancy, interstitial,
   antisite, Frenkel pair). Given the true defect location, minimisation restores 53-80%
   (`../point_defects_v2_20261109/`), so the gap to a perfect finder is still mostly in
   finding.
