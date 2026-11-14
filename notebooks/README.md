# notebooks/

The notebooks of the point-defect study. Each one clones this branch, runs on Kaggle or
Colab, and pushes its results back to `dabo-point-defects` (GitHub token in the platform's
secrets, never in the notebook). The notebooks of the first study are on the branch `main`.

- `kaggle_point_defects.ipynb`: makes the training split, trains the first repair network
  and the finder (one per GPU), evaluates them and pushes the results (~4 h on T4 x2).
  First run: `results/point_defects_trained_20261107/`.
- `kaggle_point_defects_v2.ipynb`: retrains the repair network with the fix for what the
  first run showed (atoms moved by the defect weighted in the loss, mostly small noise, the
  Frenkel atom put back first), and evaluates it with the first run's finder, including
  Frenkel pairs found and repaired without being told where they are (~8-9 h on one GPU,
  mostly the evaluation).
- `kaggle_point_defects_finder.ipynb`: continues the finder for 40,000 updates at half the
  learning rate and evaluates it with calibrated probabilities (~4.5 h on one GPU). Run on
  10 November 2026: `results/point_defects_finder_v2_20261110/`.
- `kaggle_silica_point_defects.ipynb`: the same question in amorphous silica, stage 1
  (`docs/SILICA_POINT_DEFECTS.md`): oxygen vacancy, oxygen interstitial, oxygen Frenkel pair
  and silicon vacancy in DM2's test glasses, found by the standard coordination analysis or
  the per-atom energy and repaired by minimisation (CPU only, ~11-12 h, so it may need a
  second session).
- `colab_silica_point_defects.ipynb`: continues the same silica run on Colab (CPU runtime)
  when the Kaggle hours are used up. It keeps every saved step, does only what is left,
  pushes every 20 minutes and continues after a disconnect.

The ablation runs and the final test evaluation ran on CPU:
`scripts/pilots/finder_ablations.sh` and `scripts/pilots/recheck_real_defects.sh`.

Keep notebooks thin: import from `glassdiff` rather than copying code into cells.
