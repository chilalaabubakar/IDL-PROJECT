# The main finder in three runs: validation (25 November 2026)

Item 1 of `docs/NPJ_EXTENSIONS.md`. Runs 2 and 3 of the main finder were trained on Kaggle
with `notebooks/kaggle_finder_seeds.ipynb`, with the same two-stage recipe as run 1 (stage 1:
20,000 updates from the denoiser; stage 2: 40,000 at half the learning rate from stage 1) and
new seeds (run 2: 4 and 5, run 3: 6 and 7; run 1 used 0 and 1). Weights and logs:
`results/kaggle_seeds_20261125-*/training/pd_detect_run{2,3}/` and `.../pd_detect_v2_run{2,3}/`.
Scored here on the validation glasses only (the fresh test set waits for the end) with

```bash
python scripts/pilots/displaced_inputs.py --out results/finder_runs_20261125 --sizes 0 0.02 \
    --finder run1=results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt \
    --finder run2=results/kaggle_seeds_20261125-1154Z/training/pd_detect_v2_run2/ema.pt \
    --finder run3=results/kaggle_seeds_20261125-1555Z/training/pd_detect_v2_run3/ema.pt
```

Share of the 734 real validation defects found (%), calibrated:

| | Vacancy | Interstitial | Frenkel | Antisite | All | All, atoms shifted by 0.02σ |
|---|---|---|---|---|---|---|
| Run 1 (the manuscript's finder) | 56.5 | 48.5 | 47.8 | 54.5 | 52.2 | 39.5 |
| Run 2 | 57.5 | 49.0 | 47.8 | 54.5 | 52.6 | 40.3 |
| Run 3 | 59.5 | 47.5 | 50.7 | 54.0 | 53.1 | 38.4 |
| Mean ± spread | 57.8 ± 1.5 | 48.3 ± 0.8 | 48.8 ± 1.7 | 54.3 ± 0.3 | 52.6 ± 0.5 | 39.4 ± 1.0 |

**What it shows.** The recipe is repeatable: the three runs agree within a point overall and
within 3 points for every defect type, and run 1 is not a lucky run (it is the lowest of the
three). The same holds for the sensitivity to small displacements.

**Training stability** (`scripts/pilots/finder_seeds.py` `stability`, on `steps.csv`): no
update gave a non-finite loss or gradient. Stage 1 showed 33 and 54 gradient spikes (a
gradient more than 5 times the median of the 100 updates before it), stage 2 showed 5 and 7;
85-87% of the updates were clipped at norm 1. This matches what the manuscript reports for
the EGNN: spikes, but no failed run.

**Next for item 1.** The three runs are scored on the fresh test set once, at the end, with
every other method.
