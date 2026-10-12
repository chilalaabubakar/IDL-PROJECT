# notebooks/

- `colab_train.ipynb`: **full-size training and all experiments on Colab** (GPU runtime;
  runs persist in Google Drive and resume after disconnects). Open it from
  <https://colab.research.google.com/github/chilalaabubakar/IDL-PROJECT/blob/main/notebooks/colab_train.ipynb>.

- `stage0_poc.ipynb`: the course proof-of-concept notebook, as provided (unchanged, with its
  Colab outputs): MLP denoisers on a 3×3 jittered grid, plus Frenkel-pair and antisite demos
  by naive clamping. `scripts/stage0_poc.py` reruns its cells and reproduces the proposal's
  Fig. 1 (`results/stage0/`).
- `figures.ipynb`: report figures built from `results/summary.csv` (ticket E-5).

Keep notebooks thin: import from `glassdiff` rather than copying code into cells.
