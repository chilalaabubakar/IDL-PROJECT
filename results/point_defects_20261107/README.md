# Point defects before training: what the existing denoiser already does

The reference for the point-defect study (`docs/POINT_DEFECTS.md`). Every test here uses the
**trained unconditional denoiser** `egnn_uncond_smax1`
(`results/kaggle_20261006-0454Z/training/`). It was trained with DM2's loss on defect-free
glasses shaken by random noise and has never seen a defect. The trained repair and detection
models (`notebooks/kaggle_point_defects.ipynb`) will be compared with these numbers.

Files: `noise_restoration.json` and `.log` (part 1); `eval_untrained.json` and `.log`
(part 2).

## 1. What the network computes, and restoring shaken glasses

`scripts/pilots/noise_restoration.py`, first 50 test glasses of `data/ka2d_256`.

**The learned correction is a restoring force.** On test glasses with Gaussian noise of
size σ, the correction −d points along the physical Kob-Andersen force more closely than the
straight line back to the original does:

| σ | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 | 0.8 |
|---|---|---|---|---|---|---|
| Share of the added shift predicted (R²) | 0.67 | 0.78 | 0.78 | 0.64 | 0.21 | 0.05 |
| cos(−d, force) | 0.79 | 0.81 | 0.73 | 0.65 | 0.36 | 0.09 |
| cos(way back to the original, force) | 0.70 | 0.68 | 0.63 | 0.51 | 0.20 | 0.03 |

**Restoring shaken glasses.** Each atom is shaken by δ per coordinate, or the stored T = 0.01
thermal snapshot is used. Shown: the fraction of glasses that return exactly to the original
minimum (every atom within 0.05σ).

| Perturbation | FIRE minimisation | Denoiser, then FIRE |
|---|---|---|
| δ = 0.05 | 100% | 96% |
| δ = 0.1 | 98% | 86% |
| δ = 0.2 | 38% | **50%** |
| δ = 0.3 | 0% | 0% |
| Thermal snapshot (T = 0.01) | 100% | 96% |

At δ = 0.3, neither method recovers the original, but denoising first ends 0.016 above the
original energy per atom, against 0.085 for minimisation alone.

**Local damage.** About 23 atoms within 2.5σ of a site are shaken by 0.3σ. Afterwards, 56% of
the damaged atoms are back within 0.1σ with minimisation alone, and 61% when denoising
first. The size of the denoiser's output separates damaged from intact atoms with AUC 0.98.
That matches the per-atom force (0.985) and the per-atom energy change (0.984), both of
which need the potential.

## 2. The four standard point defects

`scripts/eval_point_defects.py` on the test split of `data/ka2d_256_point_defects`: 200
relaxed examples per defect, from 100 test glasses with 2 sites each. Brackets: 95% Wilson
intervals. They treat examples as independent, although two come from each glass.

**Finding the defect** (the top-scored atom lies within 1.5σ of it):

| Defect | Denoiser output (no potential) | Highest-energy atom (needs the potential) | Random atom |
|---|---|---|---|
| Vacancy | **44%** [37, 51] | 29.5% [24, 36] | 3% |
| Interstitial | 12% [8, 17] | **28%** [22, 35] | 3% |
| Frenkel pair | 19% [14, 25] | 28.5% [23, 35] | 5% |
| Antisite | 15% [11, 21] | **29%** [23, 36] | 4% |

**Repairing it.** Restored means minimisation ends in the original glass; identical atoms
may trade places.

| Defect | Procedure | Restored |
|---|---|---|
| Vacancy | flag with the denoiser, put the missing atom in the nearest gap, minimise | **41%** [34, 48] |
| | same, with 30 denoising steps before minimising | **43.5%** [37, 50] |
| | flag with the energy, put the atom back, minimise | 26% [20, 32] |
| Interstitial | remove the atom the denoiser flags, minimise (with or without denoising) | 6% [3, 10] |
| | remove the highest-energy atom, minimise | **21%** [16, 27] |
| Frenkel pair (region given) | shake the region by 0.35σ, minimise | 0.5% [0, 3] |
| | shake, denoise, minimise | **8%** [5, 13] |
| Antisite | swap back the A-B pair near the flagged atom that the denoiser prefers | 0% [0, 2] |
| | swap back the pair with the lowest energy | 4% [2, 8] |

For Frenkel pairs, denoising also leaves more atoms back within 0.1σ (64% vs 47%). The
glass ends 0.009 above the original energy per atom, against 0.029 for shaking alone.

## What this says

1. **The learned field is physical.** The denoiser's correction is a learned restoring force,
   closer to the true force than the direct way back to the original.
2. **A potential-free vacancy finder.** Without a potential, the untrained denoiser finds
   vacancies more often than the per-atom energy, and so restores the original glass
   about 1.6 times as often (41-43.5% vs 26%).
3. **Other defects need training.** For the other three defects the energy is the better
   finder, and the denoiser cannot tell which species belongs where (antisites). For
   Frenkel pairs it helps, but rarely restores the glass (8%). It was never shown a defect: this is the case
   for training it on defects (`configs/train/heal_point_defects.yaml`,
   `configs/train/detect_point_defects.yaml`).
4. **Relaxed defects are hard to find in a glass.** Even the potential's own energies find
   the defect only about 30% of the time.

## Reproduce

```bash
python scripts/pilots/noise_restoration.py results/point_defects_20261107/noise_restoration.json   # ~45 min, 4 cores
python scripts/make_point_defects.py --config configs/data/ka2d_256_point_defects.yaml "splits=[val,test]"
python scripts/eval_point_defects.py --data data/ka2d_256_point_defects/test.npz \
    --healer results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/ema.pt \
    --out results/point_defects_20261107/eval_untrained.json                                  # ~90 min, 4 cores
```

The committed generator reproduces `val.npz` and `test.npz` byte for byte (SHA256 in
`data/ka2d_256_point_defects/meta.json`); the files were first made from the working copy
that became this commit.

An earlier exploratory version of part 2 (50 glasses per defect, PyTorch FIRE for making the
defects, a stricter "restored" that required every atom back on its own site) gave the
numbers quoted in discussion on 6-7 November. The finding rates were similar; Frenkel
restoration was 0% for every method under the stricter test. The tables above replace them.
