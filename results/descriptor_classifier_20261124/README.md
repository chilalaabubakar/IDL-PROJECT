# Classifier on classical inputs (24 November 2026)

Item 2 of `docs/NPJ_EXTENSIONS.md`: does the learned finder still lead when the classical
side is also trained on the same labelled defects? Made by
`scripts/pilots/descriptor_classifier.py` (its docstring describes the inputs and the model).

## Answer, on the validation glasses

No, not by a measurable margin. Tuned gradient-boosted trees on hand-made per-atom inputs find
as many validation defects as the finder: 50.0% with positions and species only (version D),
against 52.2% for the finder, and 52.2% when the atom energy and local pressure are added
(version E). None of the differences is significant (McNemar p = 0.18-1 per defect type).
The finder's large lead over the single measures of the manuscript (52.2% against at most
30.5%) comes from training on labelled defects, not from the graph network.

Share of the 734 real validation defects found (%):

| | Vacancy | Interstitial | Frenkel | Antisite | All |
|---|---|---|---|---|---|
| Learned finder (EGNN) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 |
| Tuned trees, positions and species only (D, chosen before validation) | 56.0 | 46.5 | 46.3 | 50.0 | 50.0 |
| Tuned trees with energy and pressure (E) | 56.0 | 50.0 | 49.2 | 52.5 | 52.2 |
| First run: untuned trees, positions and species only (best of A, C) | 56.5 | 43.0 | 41.8 | 49.5 | 48.0 |
| Best single measure of the manuscript | 42.0 | 29.5 | 36.6 | 32.5 | 30.5 |

All versions are in `summary_tuned.md` and `validation_tuned.json`. Version C (D without the
hexatic order and the second ring) finds 51.4% on validation, but D was chosen before
validation because it was best on the held-out training glasses (50.3% against 49.8%), so D
is the one reported.

## How the trees were tuned

Every 8th training glass (100 glasses, 2,977 real defects) was held out. The trees were fitted
on the other 700 glasses with each of 24 settings (learning rate 0.05 or 0.1; 31, 127 or 255
leaves; at least 20 or 200 atoms per leaf; with or without the finder's class weights) for
version D, the 4 best settings for version E, and the best setting for A and C
(`tuning_trials.json`). The setting that found the most held-out defects was kept
(`tuning.json`): learning rate 0.05, 31 leaves, the finder's class weights; 200 atoms per leaf
for E, 20 for the others. Share of held-out defects found:

| Version | First settings | Tuned |
|---|---|---|
| A: shell measures | 44.1 | 46.2 |
| C: A + radial density | 45.3 | 49.8 |
| D: C + hexatic order, second ring | 45.9 | 50.3 |
| E: D + energy, pressure | | 50.9 |

The first settings stopped too early (21-212 rounds); the tuned ones run 380-600 rounds. Seven
settings of D lie within half a point of the best, so the choice is not fragile. The trees
were then refitted on all 800 training glasses and scored once on the validation glasses.
Tuning took 72 minutes and the refit and validation 11 minutes, on 4 CPU cores.

## What it means for the manuscript

- The claim "a learned finder locates relaxed defects 18-22 points better than classical
  measures" holds only against untrained single-atom rules. Against a classifier trained on
  classical inputs with the same labels, the graph network adds about 2 points (positions
  only) or nothing (with the potential), within the noise of 200 defects per type.
- The useful result that remains: exact labels of relaxed point defects make them findable.
  Any model trained on them, from classical inputs or from raw positions, finds two to three
  times as many as the classical rules, and needs no potential when it reads positions only.
- The final comparison is on the fresh test set with three finder runs (item 1). Run
  `score --tag tuned --restored` there once, at the end (`notebooks/kaggle_descriptor_classifier.ipynb`, step F).

## Reproduced on Kaggle (25 November 2026)

`notebooks/kaggle_descriptor_classifier.ipynb` redid the tuning and the validation from
scratch on Kaggle (`results/descriptor_classifier_kaggle_20261125-1731Z/`). It chose the same
settings, found the same share of held-out defects for every version (A 46.19%, C 49.78%,
D 50.29%, E 50.89%) and the same validation result, defect by defect, for every tree
version and single measure. Only the logistic regression, a reference, differs by 0.1
points, from rounding in its solver.

## First run, untuned (kept for the record)

`summary.md`, `validation.json`, `models.pkl`: the same trees with the first settings
(learning rate 0.1, 63 leaves), which stopped after 21-212 rounds. They found 46.6-48.0% of
validation defects. The tuned models are `models_tuned.pkl`.
