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

## Second part: the finder trained on displaced inputs (25 November 2026)

Three runs, trained on Kaggle with `notebooks/kaggle_finder_displaced.ipynb`: the main finder
continued for 20,000 updates with every input atom shifted by Gaussian noise of a size drawn
per glass uniformly in [0, 0.1σ] (`configs/train/detect_point_defects_displaced.yaml`; seeds
10, 11, 12). Weights and logs: `results/kaggle_displaced_20261125-*/training/`. The notebook
scored every method with the same shifts as above: `results/kaggle_displaced_20261125-0556Z/displaced_inputs/`
(its numbers for the main finder and the trees equal those of the first part exactly, on a
GPU and with Kaggle's software, so the evaluation reproduces).

Share of the 734 real validation defects found (%), all types:

| Method | s = 0 | 0.01σ | 0.02σ | 0.05σ | 0.1σ |
|---|---|---|---|---|---|
| Finder trained on displaced inputs (mean ± spread, 3 runs) | 49.2 ± 0.2 | 50.0 ± 0.6 | 47.0 ± 0.6 | 43.1 ± 0.2 | 33.6 ± 0.6 |
| Main finder (trained on relaxed glasses) | 52.2 | 50.1 | 39.5 | 16.5 | 8.2 |
| Trees D trained on displaced inputs (positions only) | 48.8 | 46.7 | 44.7 | 40.6 | 30.0 |
| Trees E trained on displaced inputs (with potential) | 48.4 | 48.5 | 46.2 | 40.5 | 28.5 |
| Unlike neighbours | 29.0 | 29.3 | 28.7 | 27.9 | 10.0 |

**What it shows.**

- Training on displaced inputs works for the finder too. It costs 3 points on relaxed glasses
  (49.2% against 52.2%) and keeps 47.0% at 0.02σ (main finder 39.5%), 43.1% at 0.05σ
  (16.5%) and 33.6% at 0.1σ (8.2%). The three runs agree within 1.2 points at every size.
- By the test written in the plan, it passes: at 0.02σ it loses 2 points against its own
  result without shifts (at most 5 allowed), and it leads the best single measure by 18 points.
- Against the trees trained on the same displaced inputs, the finder leads by 1-2 points at
  0.02σ (not significant in any run: p = 0.07-0.92), by 2.5 points at 0.05σ (p = 0.06-0.16),
  and by 4-5 points at 0.1σ (p < 0.001 to 0.04, significant in every run). The learned network gains on
  hand-made inputs only when the displacement is large.

Training was stable: no non-finite update, 0-8 gradient spikes per run.

**Still open in item 3.** Thermal snapshots (neighbouring atoms moving together) instead of
independent shifts; the fresh test set at the end.
