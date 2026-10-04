# Colab results 20261004-2352Z

Copied from Google Drive by notebooks/colab_train.ipynb.

- `summary.md` / `summary.csv`: every experiment with 95% bootstrap CIs
- `training/<model>/: log.csv, config, training curve; `ema.pt` loads with
  `glassdiff.models.registry.load_denoiser` or `scripts/sample.py ckpt=...`
- `experiments/`: 33 runs (metrics.json, timing, config, figure)
