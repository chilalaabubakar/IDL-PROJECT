# notebooks/

The notebooks of the point-defect study. Each one clones this branch, runs on Kaggle or
Colab, and pushes its results back to `dabo-point-defects` (GitHub token in the platform's
secrets, never in the notebook). The notebooks of the first study are on the branch `main`.

**Extension work** (branch `npj-extensions`, `docs/NPJ_EXTENSIONS.md`; these push to
`npj-extensions`):

- `kaggle_finder_seeds.ipynb`: trains the main finder twice more with the same two-stage
  recipe and new seeds (runs 2 and 3; 3.5-5 h per run on a T4, both usually in one
  session) and pushes `results/kaggle_seeds_<date>/`. No test scoring: the fresh test set
  is scored once, at the end. Item 1.
- `kaggle_descriptor_classifier.ipynb`: the classifier on classical inputs
  (`scripts/pilots/descriptor_classifier.py`) on 4 CPU cores: training split, fit,
  tuning and validation (~2.5 h); with `SCORE_FRESH_TEST = True`, the fresh test set with
  repair and minimisation. Pushes `results/descriptor_classifier_kaggle_<date>/`. Item 2.
- `kaggle_finder_displaced.ipynb`: continues the main finder with every input atom
  shifted at random by up to 0.1 sigma (`configs/train/detect_point_defects_displaced.yaml`),
  three runs (~1.5 h each on a T4), then scores every method against the size of the
  displacement on the validation glasses (`scripts/pilots/displaced_inputs.py`). Pushes
  `results/kaggle_displaced_<date>/`. Item 3.

**The final study:**

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
- `kaggle_attention_finder.ipynb`, `kaggle_cnn_finder.ipynb`: other finders against the EGNN
  (`docs/POINT_DEFECTS.md`, "Other finders"). Each trains its finder and the EGNN control from
  random weights with seeds 0, 1 and 2 (one run per GPU at a time; 3-5 h on T4 x2), tests them
  on the 750 real test defects and pushes `results/<platform>_arch_<date>/`. Runs on Kaggle or
  Colab; a session that runs out of time pushes checkpoints and the next run continues. Run the
  two one after the other: the second skips the control runs the first pushed.
  `QUICK_TEST = True` runs everything in a few minutes on a CPU, without pushing. Both were
  run on 15 November 2026: `results/kaggle_arch_20261115-*/`, evaluated together in
  `results/point_defects_architectures_20261115/`.
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
