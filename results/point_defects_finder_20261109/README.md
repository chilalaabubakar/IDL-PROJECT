# Improving the finder: calibration fixes the false alarms; Frenkel pairs stay hard

> **Correction, 11 November 2026.** A quarter of the Frenkel pairs in this dataset are not
> defects: the moved atom took the place of a neighbour of its own species, which filled the
> empty site, so the relaxed glass is the original again (50 of the 200 test pairs, 65 of 199
> validation pairs, 27% of the training pairs). The Frenkel numbers below include them.
> Corrected numbers, on the 150 real test pairs, are in
> [`../point_defects_recheck_20261111/`](../point_defects_recheck_20261111/README.md). The
> other three defects are not affected.

The next step after `../point_defects_v2_20261109/`, whose conclusion was that the finder
carries the result and that the headroom is in finding. Everything here uses the trained
finder of the first run (`../kaggle_pd_20261107-2024Z/training/pd_detect/`), so there is no
new training. Settings were checked on the validation split first; the test split was used
once, for the final numbers.

Files:

- `val_diagnosis.log`, from `scripts/pilots/finder_diagnosis.py`, on validation.
- `val_frenkel_raw.json` and `val_frenkel_calibrated.json`, the Frenkel routes on
  validation.
- `eval_calibrated.json` and `.log`, the test evaluation with the calibrated finder
  (`scripts/eval_point_defects.py --calibrate --no-heal --no-energy-search`).

## 1. Diagnosis on validation

**Over-confidence comes from the class weights.** The finder was trained with weights of 9-32
on the rare roles, which inflates their probabilities. As a result, 61% of defect-free
glasses had some atom above probability 0.5 of being defective. Subtracting the log weights
from the outputs is the standard correction (logit adjustment; Menon et al., ICLR 2021) and
needs no retraining.

**Frenkel pairs: the finder sees the empty site, not the displaced atom.** It labels 83% of
displaced atoms as intact, because after relaxation the moved atom sits comfortably in its
gap. Its "any defect" score still lands within 1.5σ of the Frenkel region 41% of the time,
mostly through the atoms around the empty site.

## 2. A new Frenkel route, tried on validation and dropped

**"Empty site first"** works in three steps:

1. Take the atom most likely to be next to an empty site.
2. Choose the nearby gap whose neighbours look most like that.
3. Move into it the atom (1-3.2σ away) after whose move the finder sees the least defect.

It finds the right site as often as the existing route, but rarely moves the right atom:

| Frenkel route, validation (199 examples) | Right site | Displaced atom moved back | Restored |
|---|---|---|---|
| Flag the displaced atom, pick the gap with the finder (existing) | 21% | 14% flagged | **24.6%** |
| Empty site first, the finder as critic | 24% | 3.5% | 4.5% |

The critic prefers moving a nearby atom into the site, which only shifts the hole. The
route stays in the code behind `--empty-site-first` so that this can be reproduced, but it is
off by default and was never run on the test split.

## 3. Test: the calibrated finder

Same 800 test defects as before (200 per type). Brackets: 95% Wilson intervals. The "raw"
column is the uncalibrated finder from the earlier runs.

**Is this glass defective at all?** The score is the highest defect probability of any
atom. It is compared on 800 defected glasses and the 100 defect-free originals.

| | Defect-free glasses flagged (> 0.5) | Defected glasses flagged (> 0.5) | AUC |
|---|---|---|---|
| Raw | 59% | 83% | 0.72 |
| **Calibrated** | **1%** | 22% | 0.72 |

**Finding the defect** (top atom within 1.5σ) **and restoring the glass:**

| Defect | Found: calibrated | Found: raw | Found: energy | Restored: calibrated | Restored: raw | Restored: energy route |
|---|---|---|---|---|---|---|
| Vacancy | **67%** [60, 73] | 66% | 29.5% | **51%** [44, 58] | 51% | 26% |
| Interstitial | **47.5%** [41, 54] | 46.5% | 28% | **36%** [30, 43] | 34.5% | 21% |
| Antisite | **46.5%** [40, 53] | 44.5% | 29% | **27%** [21, 34] | 27% | 13.5% |
| Frenkel pair | **30%** [24, 37] | 26% | 28.5% | 19.5% [15, 26] | 19% | 18.5%* |

\*Frenkel energy route: flag by energy, try every gap, keep the lowest energy
(`../point_defects_v2_20261109/`).

For Frenkel pairs, the calibrated finder flags exactly the displaced atom 14% of the time
(raw 10.5%) and picks the right gap 21.5% of the time.

## What this says

1. **Calibration fixes the over-confidence at no cost.** False alarms on defect-free glasses
   fall from 59% to 1%. Finding improves by 1-4 points for every defect, and the restored
   rates are unchanged within noise. Calibrated probabilities should be used from now on. New finder
   checkpoints store their class weights (`scripts/train_point_defects.py`).
2. **Telling a defective glass from a clean one remains weak (AUC 0.72).** Inside a glass
   known to be defective, the finder ranks the defect atoms well. But one relaxed point
   defect in 256 atoms looks much like the glass's natural variation, and the potential's
   own energies find it only about 30% of the time.
3. **Frenkel pairs stay the hard case.** Calibration moves finding from 26% to 30% (energy
   28.5%), and restoration stays at 19-20%. Finding the empty site first did not help:
   choosing which atom to move back is the hard part.
4. **What would still help.** The finder had plateaued for vacancies and Frenkel pairs, while
   interstitials and antisites were still improving slowly at 20,000 updates, so a longer run
   would add a few points for those two. The larger step is the materials one: the same
   study in amorphous silica, where vacancies and coordination defects are canonical.

Reproduce:

```bash
python scripts/pilots/finder_diagnosis.py > results/point_defects_finder_20261109/val_diagnosis.log
python scripts/eval_point_defects.py --data data/ka2d_256_point_defects/val.npz --kinds frenkel \
    --healer results/kaggle_pd_20261108-2014Z/training/pd_heal_v2/ema.pt \
    --detector results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt \
    --no-heal --no-energy-search --empty-site-first [--calibrate] --out <json>   # ~40 min
python scripts/eval_point_defects.py --data data/ka2d_256_point_defects/test.npz \
    --healer results/kaggle_pd_20261108-2014Z/training/pd_heal_v2/ema.pt \
    --detector results/kaggle_pd_20261107-2024Z/training/pd_detect/ema.pt \
    --calibrate --no-heal --no-energy-search \
    --out results/point_defects_finder_20261109/eval_calibrated.json            # ~35 min, 4 cores
```

The validation Frenkel files were made with `--empty-site-first` before it became optional.
