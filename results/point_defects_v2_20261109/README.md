# Point defects, repair version 2: the finder does the work

> **Correction, 11 November 2026.** A quarter of the Frenkel pairs in this dataset are not
> defects: the moved atom took the place of a neighbour of its own species, which filled the
> empty site, so the relaxed glass is the original again (50 of the 200 test pairs, 65 of 199
> validation pairs, 27% of the training pairs). The Frenkel numbers below include them.
> Corrected numbers, on the 150 real test pairs, are in
> [`../point_defects_recheck_20261111/`](../point_defects_recheck_20261111/README.md). The
> other three defects are not affected.

Second Kaggle run of the point-defect study (`docs/POINT_DEFECTS.md`), with
`notebooks/kaggle_point_defects_v2.ipynb` on 8-9 November 2026:

- **Training:** `../kaggle_pd_20261108-2014Z/training/pd_heal_v2/`. The repair model,
  version 2, ran 20,000 updates in 90 min from the version-1 repair model, with code at
  fd79bd4 (`configs/train/heal_point_defects_v2.yaml`).
- **Evaluation:** `../kaggle_pd_20261109-0230Z/eval_trained_v2.json`. Version 2 with the
  version-1 finder (`../kaggle_pd_20261107-2024Z/training/pd_detect/`) on the same 800 test
  defects as before (200 per type). It took 6.3 h, mostly the Frenkel trials.
- **Repair without a potential** (this folder): `repair_no_potential.json` and `.log`,
  made with `scripts/eval_point_defect_repair.py`.

Brackets: 95% Wilson intervals over 200 examples (two per test glass, treated as
independent).

## 1. Version 2 learned the defect

Validation, one repair step at σ = 0.001. Shown: the RMS error on the atoms the defect
moved (> 0.1σ), with the Frenkel atom put back at its site, on the 200 validation examples
per type.

| Defect | Do nothing | Version 1 | Version 2 | Share removed: v1 → v2 |
|---|---|---|---|---|
| Frenkel pair | 0.294 | 0.276 | **0.218** | 6% → **26%** |
| Interstitial | 0.191 | 0.175 | **0.163** | 8% → **15%** |
| Antisite | 0.203 | 0.184 | **0.170** | 9% → **16%** |
| Vacancy | 0.234 | 0.216 | **0.205** | 8% → **12%** |

Other changes from version 1:

- The training loss kept falling (0.0133 → 0.0114) instead of staying flat.
- The defect error fell until the last update, so a longer run would likely help further.
- On defect-free glasses version 2 moves atoms less than version 1 (error 0.012 vs 0.027).

## 2. With minimisation, the repair model makes no difference

Restored means the final minimisation ends in the original glass (identical atoms may
trade places).

| Defect | Finder, fix the count or species, minimise | + repair v1 | + repair v2 | Physics only (energy) |
|---|---|---|---|---|
| Vacancy | **51%** [44, 58] | 54% [47, 61] | 51.5% [45, 58] | 26% [20, 32] |
| Interstitial | **34.5%** [28, 41] | 35% [29, 42] | 34% [28, 41] | 21% [16, 27] |
| Antisite | **27%** [21, 34] | – | – | 13.5% [9, 19] |
| Frenkel pair, region given (shake) | – | 10% [6, 15] | 9% [5, 14] | 0.5% (shake only) |

The finder's numbers are identical to version 1, as they should be: it is the same finder,
and every random choice is seeded. Once the right atom is added, removed or swapped,
minimisation completes the repair, and the repair model's improved single step does not
change which minimum the glass reaches.

## 3. Frenkel pairs found and repaired with nothing given

The procedure:

1. Flag the displaced atom.
2. List up to 8 gaps 1-3.2σ from it.
3. Move the atom into the gap chosen by the model (the finder's "next to an empty site"
   probability around each gap) or by the energy (minimise every try and keep the lowest).
4. Minimise.

| Flag by | Pick the gap by | Flagged the displaced atom | Chose the right gap | Restored |
|---|---|---|---|---|
| Finder | finder | 10.5% [7, 16] | 22% [17, 28] | **19%** [14, 25] |
| Finder | finder, then repair v2, then minimise | | | 17.5% [13, 23] |
| Finder | energy (8 trial minimisations) | | 11.5% [8, 17] | 19.5% [15, 26] |
| Energy | energy (physics only) | 6% [3, 10] | 6% [3, 10] | 18.5% [14, 24] |
| Energy | finder | | 11.5% [8, 17] | 13% [9, 18] |

Without being told where the defect is, about 19% of Frenkel pairs are restored. That is
more than the region-given test (9-10%) and far more than shaking alone (0.5%), but the
learned route only ties a pure physics search (18.5%).

The finder rarely flags exactly the displaced atom (10.5%). Many successes come from a
neighbouring atom of the same species being moved into the empty site. That creates a
closer vacancy-interstitial pair, which minimisation then closes; identical atoms trading
places still counts as restored.

## 4. Repair without a potential

The defect is given: the stored repair input, with the atom count and species right at the
true site, and the Frenkel atom put back in the gap at its site. So this measures repair alone,
not finding. Each model is applied to the whole glass, either one step at σ = 0.001 or 10
steps with σ falling from 0.05 to 0.001. No minimisation is used. Columns:

- **Defect atoms:** RMS distance to the original over the atoms the defect moved.
- **All atoms:** RMS distance over all atoms.
- **Overlap:** the fraction of glasses with two atoms closer than 0.56σ (the dataset's quality
  gate), i.e. physically broken structures.
- **Energy:** the median energy per atom above the original, scored with the potential.

Minimisation is the reference that needs the potential.

| Defect | Method | Defect atoms | All atoms | Overlap | Energy |
|---|---|---|---|---|---|
| Vacancy | do nothing | 0.233 | 0.084 | 0% | 0.16 |
| | repair v1, one step | 0.216 | 0.082 | 18% | 0.14 |
| | **repair v2, one step** | **0.203** | **0.077** | **20%** | 0.11 |
| | repair v2, 10 steps | 0.231 | 0.104 | 60% | 40 |
| | minimise (needs the potential) | 0.208 | 0.079 | 0% | 0 (67% restored) |
| Frenkel pair | do nothing | 0.280 | 0.117 | 0% | 0.23 |
| | repair v1, one step | 0.265 | 0.115 | 16.5% | 0.21 |
| | **repair v2, one step** | **0.220** | **0.102** | **50.5%** | 5.1 |
| | repair v2, 10 steps | 0.258 | 0.133 | 75% | 306 |
| | minimise (needs the potential) | 0.256 | 0.114 | 0% | 0 (53% restored) |
| Interstitial | do nothing | 0.190 | 0.078 | 0% | 0.02 |
| | repair v2, one step | 0.160 | 0.071 | 1% | 0.09 |
| | minimise (needs the potential) | **0.108** | **0.043** | 0% | 0 (80% restored) |
| Antisite | do nothing | 0.211 | 0.096 | 0% | 0.12 |
| | repair v2, one step | 0.180 | 0.087 | 2% | 0.14 |
| | minimise (needs the potential) | **0.150** | **0.069** | 0% | 0 (68.5% restored) |

The full table, including the untrained denoiser and 10-step runs for every model, is in
`repair_no_potential.json`.

- **Vacancies and Frenkel pairs:** in one step, repair version 2 brings the glass closer to
  the original than minimisation does. But it breaks many structures, by pushing atoms onto
  each other in 20% of vacancy glasses and half of the Frenkel glasses.
- **Iterating diverges:** repeated steps break most glasses (50-75%).
- **Interstitials and antisites:** minimisation is clearly better.
- **Version 1 and the untrained denoiser:** they move the defect atoms less and break fewer
  structures.

## What this says

1. **The finder is the result.**
   - Without using the potential, it finds vacancies, interstitials and antisites more
     often than the per-atom energy does: 66, 46.5 and 44.5% vs about 29%.
   - Fixing the atom count or species at the flagged atom and minimising then restores
     about twice as many glasses as the energy-based route: 51, 34.5 and 27% vs 26, 21 and
     13.5%.
   - This answers Prof. Dabo's point 7, identifying incorrect structures.
2. **Repair is done by minimisation once the defect is found.** Given the true defect
   location, minimisation restores 67% of vacancy glasses, 80% of interstitial, 53% of
   Frenkel and 68.5% of antisite glasses. The finder-based route reaches 51, 34.5, 19 and
   27%, so the remaining headroom is in finding, not in repair.
3. **The repair model is not yet useful in either setting.**
   - With a potential, it adds nothing measurable after minimisation.
   - Without a potential, version 2 moves the defect atoms closer to the original (13-22%
     of the error removed in one step), sometimes closer than minimisation. But it leaves
     overlapping atoms in up to half of the glasses.
   - A physical constraint on its output (no two atoms closer than a contact distance)
     would be the next thing to try.
4. **Frenkel pairs are the hardest defect.**
   - Found and repaired with nothing given: 19%. A pure physics search does as well
     (18.5%).
   - The finder rarely flags exactly the displaced atom.

For the report, the main result is: a learned finder that needs no potential, plus fixing
the atom count or species, plus minimisation. Prof. Dabo's point 6, repair, works through
that pipeline. The diffusion repair model is a secondary result, with a clear failure
mode.

## Reproduce

```bash
# Kaggle: notebooks/kaggle_point_defects_v2.ipynb (data ~45 min, training ~1.5 h, evaluation ~6.3 h)
python scripts/eval_point_defect_repair.py --data data/ka2d_256_point_defects/test.npz \
    --model untrained=results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt \
    --model v1=results/kaggle_pd_20261107-2024Z/training/pd_heal/ema.pt \
    --model v2=results/kaggle_pd_20261108-2014Z/training/pd_heal_v2/ema.pt \
    --out results/point_defects_v2_20261109/repair_no_potential.json        # ~80 min, 4 CPU cores
```
