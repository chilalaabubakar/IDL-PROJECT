# data/

The generated datasets are committed here (~20 MB in total), so you can start without
regenerating them:

| Directory | Contents |
|---|---|
| `ka2d_256/` | 800 / 100 / 100 glasses, N = 256, quench 1e-2/τ; `patches_train.npz` (10,545 patches) |
| `ka2d_64/` | 1600 / 200 / 200 glasses, N = 64 (MLP baseline, matched-N symmetry ladder); patches |
| `ka2d_1024/` | 100 test glasses, N = 1024 (size transfer) |
| `ka2d_256_slow/` | 200 test glasses, N = 256, quench 1e-3/τ (better annealed, out of distribution) |
| `sio2_dm2/` | 3D silica: DM2's 24 published glasses (3,000 atoms; 0.1, 1, 10, 100 K/ps), split 16 / 4 / 4 and labelled (`scripts/silica/import_dm2.py`; DM2, MIT licence). Our 576-atom glasses will go in `sio2_576/` (`docs/SILICA_QUICKSTART.md`) |

Every directory has `meta.json` with the protocol, LAMMPS version, rejection log and
the frozen defect thresholds (`"defects"`). File format: docs/DATASET.md §1.7.

They are reproducible exactly from the configs (same seeds give bit-identical glasses):

```bash
python scripts/make_dataset.py --config configs/data/ka2d_256.yaml
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256
python scripts/build_patches.py --config configs/defects/v0.yaml data=data/ka2d_256
```

Do not regenerate or re-label them once models have been trained on them: the thresholds
are frozen (digest `1c8714e00c26` for `ka2d_256`).
