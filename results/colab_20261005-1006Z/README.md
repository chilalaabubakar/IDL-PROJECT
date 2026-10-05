# Colab results 20261005-1006Z

Copied from Google Drive by notebooks/colab_train.ipynb.

- `summary.md` / `summary.csv`: every experiment with 95% bootstrap CIs
- `training/<model>/`: log.csv, config, training curve; `ema.pt` loads with
  `glassdiff.models.registry.load_denoiser` or `scripts/sample.py ckpt=...`
- `experiments/`: 34 runs not in an earlier push (metrics.json, timing, config, figure); `summary.*` lists all of them
- `diagnostics/`: control runs such as `defect_stability`
