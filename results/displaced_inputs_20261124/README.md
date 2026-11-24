# Defects found against a random displacement of the atoms (24 November 2026)

Item 3 of `docs/NPJ_EXTENSIONS.md`, first part: the methods that exist before any finder is
trained on displaced inputs. Validation glasses only (734 real defects); the fresh test set
is scored once, at the end. Made by

```bash
python scripts/pilots/displaced_inputs.py --out results/displaced_inputs_20261124 \
    --finder main=results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt \
    --trees tuned=results/descriptor_classifier_20261124/models_tuned.pkl \
    --trees displaced=results/descriptor_classifier_20261124/models_displaced.pkl
```

(11 minutes on 4 CPU cores). Every atom of each relaxed defected glass is shifted by
Gaussian noise of standard deviation s, the same shifts for every method. The trees are those
of item 2 (`results/descriptor_classifier_20261124/`): tuned on clean inputs, and the same
settings refitted on training glasses displaced by a size drawn uniformly in [0, 0.1σ]
(`descriptor_classifier.py run --input-noise 0.1 --tag displaced`). D uses positions and
species only; E adds the atom energy and local pressure.

Share found (%), all defect types:

| Method | s = 0 | 0.01σ | 0.02σ | 0.05σ | 0.1σ |
|---|---|---|---|---|---|
| Main finder (EGNN, trained on relaxed glasses) | 52.2 | 50.1 | 39.5 | 16.5 | 8.2 |
| Trees D, trained on relaxed glasses | 50.0 | 43.2 | 38.4 | 26.7 | 16.4 |
| Trees D, trained on displaced glasses | 48.8 | 46.7 | 44.7 | 40.6 | 30.0 |
| Trees E, trained on displaced glasses | 48.4 | 48.5 | 46.2 | 40.5 | 28.5 |
| Unlike neighbours (best single measure under displacement) | 29.0 | 29.3 | 28.7 | 27.9 | 10.0 |
| Atom energy | 30.5 | 28.1 | 12.7 | 3.4 | 4.9 |

All methods, the results by defect type at 0.02σ and the McNemar p values are in `summary.md`;
every outcome is in `found.json`.

**What it shows.**

- Every model trained on relaxed glasses breaks down under small displacements, the finder
  most: from 52.2% to 39.5% at 0.02σ and 16.5% at 0.05σ. The interstitials and Frenkel pairs
  go first (27.0% and 23.1% at 0.02σ, from 48.5% and 47.8%); the vacancies hold (56.5%).
- Training on displaced inputs fixes most of it at a small cost. The trees trained on
  displaced glasses lose 1-4 points on relaxed glasses (48.4-48.8% against 50.0-52.2%), keep
  44.7-46.2% at 0.02σ and 40.5-40.6% at 0.05σ, and beat the main finder at 0.02σ by 5-7
  points (p = 0.004 and p < 0.001).
- The atom energy collapses (12.7% at 0.02σ), because the energy rises steeply when two atoms
  approach. The count of unlike neighbours is unchanged up to 0.05σ, because a shift that
  small rarely moves an atom across the neighbour cutoff.

**Next.** The finder trained on displaced inputs (`notebooks/kaggle_finder_displaced.ipynb`,
three runs on a GPU) is scored by the same script, with the same shifts, next to these
methods. The question is whether it keeps a lead over the trees trained on displaced inputs
at every size, or matches them as on relaxed glasses (item 2).

**Limits.** Validation glasses only, 200 defects per type. Independent Gaussian shifts are
not thermal vibration, where neighbouring atoms move together; thermal snapshots are the next
part of item 3. The training range (up to 0.1σ) was set in advance, not tuned.
