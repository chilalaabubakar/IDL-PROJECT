# notebooks/

- `colab_train.ipynb`: **full-size training and all experiments on Colab** (GPU runtime;
  runs persist in Google Drive and resume after disconnects). Open it from
  <https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_train.ipynb>.

- `kaggle_round3.ipynb`, `kaggle_sensitivity.ipynb`, `kaggle_seeds.ipynb`: round 3, the
  sensitivity checks and the training seeds on Kaggle (no Drive; models and results come from
  this repository, progress is pushed back).
  The sensitivity notebook re-runs the headline Task B rows and baselines, re-scores them
  with `scripts/sensitivity.py`, and varies R_out. The seeds notebook trains both
  conditional models with seeds 1 and 2 (one per GPU, resumable across sessions) and repeats
  the five headline rows with them.

- `kaggle_extras.ipynb`: extra 2D checks for the paper that need no new training, one
  session on T4 x2 (~4.5 h): **E1** choosing the guidance weight w per trained model on
  validation glasses (rules fixed beforehand in `configs/eval/select_w.yaml`), then testing
  it; **E2** the seam check (realism shell by shell around the site, `scripts/seam_profile.py`)
  for R_out = 3–6; **E3** D1+ with a larger rebuilt region (R_out = 5, 6).
- `kaggle_rout_range.ipynb`: trains both conditional recipes again with R_out drawn from 3–6σ
  for each training request, then compares them with the fixed-R_out models at R_out = 3, 4,
  5 (~6 h on T4 x2).
- `kaggle_silica_train.ipynb`: 3D silica training on DM2's glasses
  (`docs/SILICA_QUICKSTART.md`).
- `kaggle_point_defects.ipynb`: the point-defect study (`docs/POINT_DEFECTS.md`): makes the
  training split, trains the repair and finder models (one per GPU), evaluates them and
  pushes the results (~4 h on T4 x2). First run: `results/point_defects_trained_20261107/`.
- `kaggle_point_defects_v2.ipynb`: retrains the repair model with the fix for what the first
  run showed (defect atoms weighted in the loss, mostly small noise, Frenkel atom put back
  first), and evaluates it with the first run's finder, including Frenkel pairs found and
  repaired without being told where they are (~8-9 h on one GPU, mostly the evaluation). Pushes to the
  `dabo-point-defects` branch.
- `kaggle_point_defects_finder.ipynb`: continues the finder for 40,000 updates at half the
  learning rate (it was still improving for interstitials and antisites) and evaluates it with
  calibrated probabilities (~4.5 h on one GPU). Pushes to `dabo-point-defects`. Run on
  10 November 2026: `results/point_defects_finder_v2_20261110/`.
- `kaggle_silica_point_defects.ipynb`: the point-defect study in amorphous silica, stage 1
  (`docs/SILICA_POINT_DEFECTS.md`): oxygen vacancy, oxygen interstitial, oxygen Frenkel pair
  and silicon vacancy in DM2's test glasses, found by the standard coordination analysis or the
  per-atom energy and repaired by minimisation (CPU only, ~5-7 h). Pushes to `dabo-point-defects`.

- `stage0_poc.ipynb`: the course proof-of-concept notebook, as provided (unchanged, with its
  Colab outputs): MLP denoisers on a 3×3 jittered grid, plus Frenkel-pair and antisite demos
  by naive clamping. `scripts/stage0_poc.py` reruns its cells and reproduces the proposal's
  Fig. 1 (`results/stage0/`).
- `figures.ipynb`: report figures built from `results/summary.csv` (ticket E-5).

Keep notebooks thin: import from `glassdiff` rather than copying code into cells.
