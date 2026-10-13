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

- `stage0_poc.ipynb`: the course proof-of-concept notebook, as provided (unchanged, with its
  Colab outputs): MLP denoisers on a 3×3 jittered grid, plus Frenkel-pair and antisite demos
  by naive clamping. `scripts/stage0_poc.py` reruns its cells and reproduces the proposal's
  Fig. 1 (`results/stage0/`).
- `figures.ipynb`: report figures built from `results/summary.csv` (ticket E-5).

Keep notebooks thin: import from `glassdiff` rather than copying code into cells.
