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

- `stage0_poc.ipynb`: the course proof-of-concept notebook, as provided (unchanged, with its
  Colab outputs): MLP denoisers on a 3×3 jittered grid, plus Frenkel-pair and antisite demos
  by naive clamping. `scripts/stage0_poc.py` reruns its cells and reproduces the proposal's
  Fig. 1 (`results/stage0/`).
- `figures.ipynb`: report figures built from `results/summary.csv` (ticket E-5).

Keep notebooks thin: import from `glassdiff` rather than copying code into cells.
