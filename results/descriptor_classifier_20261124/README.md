# Classifier on classical inputs: first result, validation (24 November 2026)

Item 2 of `docs/NPJ_EXTENSIONS.md`. Made by

```bash
python scripts/pilots/descriptor_classifier.py run --out results/descriptor_classifier_20261124
```

(16 min on 4 CPU cores; the training split was already present and matched its SHA256).

**What was trained.** Gradient-boosted trees on per-atom classical inputs, with the finder's
five roles as labels, on every atom with a defect role and a random tenth of the intact atoms
of the 25,577 training defects (745,635 rows). Version A uses positions and species only
(cell area, number of neighbours, number of unlike neighbours, nearest distance, and their
mean, minimum and maximum over the neighbour shell); B adds atom energy and local pressure
(needs the potential); C adds the radial density of each species within 2.5σ. Default
settings, early stopping on a tenth of the training rows (39-101 rounds). Ranking by the
log-odds of the role against "intact" beat ranking by its probability for every model on
validation, so the log-odds are used.

**Result on the validation defects** (734 real defects; `summary.md`, `validation.json`).
Share found (%):

| | Vacancy | Interstitial | Frenkel | Antisite | All |
|---|---|---|---|---|---|
| Learned finder (EGNN) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 |
| Trees, positions and species only (best of A, C) | 56.5 | 43.0 | 41.8 | 49.5 | 48.0 |
| Trees with energy and pressure (B) | 56.0 | 42.5 | 45.5 | 47.0 | 48.0 |
| Best single measure of the manuscript | 42.0 | 29.5 | 36.6 | 32.5 | 30.5 |

**What it means.**

- A classifier trained on classical inputs recovers most of the finder's lead over the
  single measures. Over all defects the finder leads it by about 4 points (52.2% against
  48.0%), where it leads the best single measure by about 22.
- Vacancies: no difference (56.5% against 55.5-56.5%; McNemar p ≥ 0.85).
- Interstitials, antisites and Frenkel pairs: the finder leads by 5-10 points, significant
  only for interstitials against version A (p = 0.001); against version C, p = 0.08 for
  interstitials, 0.06 for antisites and 0.29 for Frenkel pairs (not corrected for multiple
  tests).
- So most of the 18-22 points of the manuscript come from training on labelled defects, not
  from the graph network. By the test written in the plan before the run (drop the claim for
  a type where version A comes within 5 points), the claim is lost for vacancies, is at the
  edge for antisites (5.0 points), and stands only weakly for interstitials and Frenkel pairs.

**Limits of this first result.** Validation only (200 defects per type, intervals of about
±7 points); the choice of log-odds was made on the same defects, which favours the trees
slightly; the trees were not tuned, so a tuned version may close more of the gap; one finder
run. The decision belongs on the fresh test set, with the three finder runs, at the end.

**Next.** Give the classical side its best chance before the final test: tune the trees on
validation (learning rate, size of the trees, inputs from the second neighbour shell). If
the finder's lead stays at about 5 points, the paper's claim becomes: training on exact
defect labels is what makes defects findable; a graph network adds a few points for
interstitials, antisites and Frenkel pairs.
