# Point defects after training (version 1): the finder works, the repair model does not yet

The first trained models of the point-defect study (`docs/POINT_DEFECTS.md`), run with
`notebooks/kaggle_point_defects.ipynb` on Kaggle (T4 x2) on 7 November 2026.

- Training: `../kaggle_pd_20261107-2024Z/training/`. `pd_heal` (repair) and `pd_detect`
  (finder) ran 20,000 updates each from the trained unconditional denoiser, taking 97 and
  74 min. Code at 640fdfd.
- Evaluation: `../kaggle_pd_20261107-2223Z/eval_trained.json`. The same 800 test defects
  (200 per type) and procedure as the before-training reference,
  `../point_defects_20261107/eval_untrained.json`.

Brackets: 95% Wilson intervals over the 200 examples (two per test glass, treated as
independent).

## Finding the defect

The top-flagged atom must lie within 1.5σ of the defect.

| Defect | Trained finder | Untrained denoiser | Highest-energy atom (needs the potential) | Random atom |
|---|---|---|---|---|
| Vacancy | **66%** [59, 72] | 44% [37, 51] | 29.5% [24, 36] | 3% |
| Interstitial | **46.5%** [40, 53] | 12% [8, 17] | 28% [22, 35] | 3% |
| Antisite | **44.5%** [38, 51] | 15% [11, 21] | 29% [23, 36] | 4% |
| Frenkel pair | 26% [20, 32] | 19% [14, 25] | 28.5% [23, 35] | 5% |

The trained finder beats the potential's own energies for three of the four defects, without
using the potential. For Frenkel pairs it does not: on validation glasses it labels the
displaced atom correctly only 5.5% of the time.

## Getting the original glass back

Restored = the final minimisation ends in the original glass (identical atoms may trade
places).

| Defect | Procedure | Trained | Untrained |
|---|---|---|---|
| Vacancy | flag with the model, put the missing atom in the nearest gap, minimise | **51%** [44, 58] | 41% |
| | same, with the repair model before minimising | **54%** [47, 61] | 43.5% |
| | flag with the energy, put the atom back, minimise | 26% [20, 32] | 26% |
| Interstitial | remove the flagged atom, minimise | **34.5%** [28, 41] | 6% |
| | same, with the repair model | **35%** [29, 42] | 6% |
| | remove the highest-energy atom, minimise | 21% [16, 27] | 21% |
| Antisite | swap back the pair the model prefers, minimise | **27%** [21, 34] | 0% |
| | swap back the lowest-energy pair (near the model's flag), minimise | 13.5% [9, 19] | 4% |
| Frenkel pair (region given) | shake, repair, minimise | 10% [6, 15] | 8% |
| | shake, minimise | 0.5% | 0.5% |

## Diagnosis: the repair model learned to remove noise, not the defect

**Most of the gain comes from finding the right atom.** The repair step adds 3 points for
vacancies, 0.5 for interstitials and 2 for Frenkel pairs.

**Validation confirms it.** With almost no noise (σ = 0.001), the repair model's largest
single-atom error barely differs from doing nothing:

| Defect | Repair model | Do nothing |
|---|---|---|
| Frenkel pair | 1.90σ | 1.90σ |
| Vacancy | 0.42σ | 0.49σ |
| Interstitial | 0.30σ | 0.37σ |
| Antisite | 0.36σ | 0.44σ |

**The training loss stayed flat** from step 1,000 to 20,000 (0.065).

**Why.** A defect moves 30-40 of the 256 atoms by more than 0.1σ, and for a Frenkel pair
87% of the squared shift sits in the top 10 atoms. The loss averaged over all atoms, with
noise up to σ = 0.5, so the random noise filled almost all of it. A Frenkel pair also asks
the model to predict a single jump of 1.2-3σ.

**The fix** (`configs/train/heal_point_defects_v2.yaml`,
`notebooks/kaggle_point_defects_v2.ipynb`):

1. **Defect atoms count more.** Atoms the defect moved by more than 0.1σ get 11 times the
   weight in the loss, about two-thirds of it in total.
2. **Mostly small noise.** Noise is drawn log-uniformly between 0.001 and 0.3, so in most
   examples the defect is what is left to remove.
3. **Frenkel pairs: put the atom back first.** The repair input has the displaced atom back
   in the gap at its empty site. The largest remaining shift drops from 1.90σ to 0.85σ on
   average, and the model cleans up the surroundings.
4. **Choosing the gap.** Which gap the atom goes back to is a separate choice in the
   evaluation, made without being told where the defect is. The candidates are up to 8 gaps
   1-3.2σ from the flagged atom. The choice is the gap whose neighbours the finder sees as
   "next to an empty site", or, with the potential, the lowest-energy outcome.

**Local check of the fix.** CPU, 400 updates from the version-1 repair model, raw weights.
Shown: error on the atoms the defect moved, one repair step at σ = 0.001, on 100 validation
examples per defect. Every row uses the version-2 inputs (Frenkel atom put back at its site).

| Defect | Do nothing | Version 1 (20,000 updates) | + 400 updates, old loss | + 400 updates, new loss |
|---|---|---|---|---|
| Frenkel pair | 0.301 (worst atom 0.867) | 0.281 (0.801) | 0.279 (0.796) | **0.265 (0.740)** |
| Interstitial | 0.192 (0.372) | 0.178 (0.307) | 0.178 (0.309) | **0.175 (0.299)** |
| Antisite | 0.200 (0.436) | 0.183 (0.350) | 0.183 (0.349) | **0.180 (0.341)** |
| Vacancy | 0.235 (0.542) | 0.216 (0.464) | 0.216 (0.464) | 0.216 (0.460) |

With the old loss, the extra updates change nothing; with the new loss, the defect error
falls, mostly for Frenkel pairs. This is 2% of a full run, so it shows the direction, not
the size of the final effect.

Other notes:

- The finder is over-confident. On defect-free validation glasses, some atom gets a
  defect probability above 0.5 in 61% of glasses. It ranks atoms well, but its
  probabilities should not be used to decide whether a glass has a defect at all.
- On Kaggle the notebook ran from `main`, so these results were pushed there and then
  merged into `dabo-point-defects`.
