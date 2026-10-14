# Compute estimate for the rest of the project

Method (as in an ACCESS proposal): for each piece of work, count the model variants we expect
to train, the epochs each needs, and the measured time per epoch.

    GPU-hours = variants × epochs × epoch time

Credits follow the PSC Explore guide: **200,000 credits ≈ 3,700 GPU-hours** (54 credits per
GPU-hour), **10,000 credits ≈ 1 TB of storage**. ACCESS describes one credit as roughly one
CPU core-hour; the exact rate for Bridges-2 RM should be checked in the
[exchange calculator](https://allocations.access-ci.org/exchange_calculator).

All tables are printed by `python docs/compute/budget.py`; change an assumption there and
rerun.

## Bottom line

| | Base plan | Upper estimate |
|---|---|---|
| GPU (3D models, sampling, checks) | 347 GPU-h → 18.8K credits | 627 GPU-h → 33.9K credits |
| CPU (MD simulations, baselines) | 1,551 core-h → ~1.6K credits | 6,551 core-h with the optional DFT check → ~6.6K credits |
| Storage | 1 TB → 10K credits | 1 TB → 10K credits |
| **Total** | **≈ 30K credits** | **≈ 50K credits** |
| 2D work left (Kaggle, free) | 22 T4 GPU-h | |

**An Explore allocation (200K credits) covers the upper estimate four times over.** The spare
credits leave room for larger cells (DM2 uses 3,000 atoms), more ablations, or the DFT check.

## What the numbers rest on

| Quantity | Value | Source |
|---|---|---|
| 2D training, EGNN | 0.27–0.37 s per update (batch 16 × 256 atoms), 16 s per epoch | measured, our 11 Kaggle/Colab T4 runs |
| 2D evaluation row (256 samples, 200 steps, relaxed) | 15 min | measured, median of 73 rows |
| 3D training, NequIP (DM2 settings, 0.60 M parameters) | 1.2 s per update (batch 16 × 576 atoms), V100/A6000 class | DM2: 30,000 updates on 3,000-atom cells in ~40 h on an A6000, scaled by atom count |
| Cross-check: NequIP vs our EGNN, per atom | 2.9× per training step | measured on CPU, `bench_models.py`; predicts ~0.6 s per update, so 1.2 s is conservative |
| NequIP forward pass vs training step | 0.21× | measured on CPU, `bench_models.py` |
| 3D EGNN | 0.35 s per update | 2D EGNN measurement, 3D neighbour count |
| Silica MD (8 Å Born + DSF Coulomb) | 200,000 atom-steps per second per core (2.9 ms per step for 576 atoms) | measured, LAMMPS, `in.silica_md_bench` |
| One 3D epoch | 600 training glasses / batch 16 = 37.5 updates: NequIP 45 s, EGNN-3D 13 s | derived |
| One full training run | 50,000 updates = 1,333 epochs = 16.7 GPU-h (NequIP) | derived (same length as our 2D runs) |

These are planning numbers. The first week of the 3D work replaces them with timings on the
GPU we are given.

## 3D models: training (base plan)

The screening grid has 32 variants: maximum training noise (0.75 or 1.0 Å) × training region
size (fixed 6 Å or a 5–8 Å range) × network size (DM2's or half) × number of layers (3 or 4)
× learning rate (2×10⁻⁴ or 5×10⁻⁴). Short runs pick the best, as in the guide's example
(100 ablations × 5 epochs, top 10 × 10 more, top 2 × 50 more).

| NequIP stage | Variants | Epochs each | Epochs | GPU-h |
|---|---|---|---|---|
| Screening | 32 | 133 | 4,256 | 53.2 |
| Top 6 continue | 6 | 267 | 1,602 | 20.0 |
| Top 2 continue to full length | 2 | 933 | 1,866 | 23.3 |
| Best variant, 2 more seeds | 2 | 1,333 | 2,666 | 33.3 |
| Model without labels (for baselines) | 1 | 1,333 | 1,333 | 16.7 |
| **Total** | | | **11,723** | **146.5** |

| EGNN-3D stage (symmetry test in 3D) | Variants | Epochs each | Epochs | GPU-h |
|---|---|---|---|---|
| Screening | 4 | 133 | 532 | 1.9 |
| Best variant, 3 seeds | 3 | 1,333 | 3,999 | 14.6 |

## 3D models: sampling and checks (base plan)

Each row is 256 generated samples. Rows with guidance run the network twice per step. The
number of sampler steps is set by the calibration experiment; the base plan assumes 1,000
steps (DM2 uses 3,000; our 2D model needed 100).

| Item | Rows | GPU-h |
|---|---|---|
| Main table: 3 defect types × w = 2, 6 × 3 seeds | 18 | 40.3 |
| Guidance sweep: 3 types × w = 0, 1, 4 | 9 | 20.2 |
| Region-size sweep: 3 types × 2 extra sizes | 6 | 13.4 |
| Baselines without labels (chance, clamp, RePaint) | 9 | 10.1 |
| EGNN-3D rows: 3 types × 3 seeds | 9 | 5.9 |
| Step-count calibration (100–3,000 steps, 64 samples) | 4 | 1.2 |
| Physics checks: MACE re-relaxation of ~2,300 structures, vibrations | | 20 |
| Setup, benchmarks and debugging | | 15 |

Training + sampling + checks = 289 GPU-h; **with 20% contingency, 347 GPU-h**.

**Upper estimate (627 GPU-h):** 64 screening variants (label dropout 0.1 or 0.2 added) and
3,000 sampler steps. The step count is the largest single uncertainty: sampling costs 28,
91 or 271 GPU-h at 300, 1,000 or 3,000 steps.

## CPU: simulations and baselines

| Item | Core-h |
|---|---|
| MD dataset: 550 glasses at 10 K/ps (0.58 h each), 200 at 100 K/ps (0.24 h), 50 at 1 K/ps (3.96 h), 10 large (1,536 atoms) at 1 K/ps; +10% reruns | 735 |
| Local melt-quench baseline: 3 types × 2 sizes × 256 attempts (0.42 h each) | 641 |
| Relaxing generated samples in LAMMPS: ~60 rows × 256 | 25 |
| Hand-built baseline and energy-barrier (NEB) application | 150 |
| **Total** | **1,551** |
| Optional DFT check (20–30 cells of 192 atoms) | 2,000–5,000 |

The melt-quench baseline can be cut by about two-thirds by excluding frozen–frozen atom pairs
from the force calculation.

## 2D work left (free Kaggle T4 GPUs)

| Item | Runs/rows | GPU-h |
|---|---|---|
| Train with a range of region sizes (D2 and D1− recipes) | 2 | 8.9 |
| Size test on 1,024-atom glasses | 5 | 5.0 |
| Slowly cooled hosts | 5 | 1.2 |
| Choose w per model on validation data | 18 | 4.5 |
| Evaluate the region-range models | 6 | 1.5 |
| Larger region for D1+ | 3 | 0.8 |
| **Total** | | **22** |

## Storage

Datasets (~800 glasses plus dumps for analysis, ~10 GB), checkpoints (~2.4 MB each, a few
hundred), generated samples (~0.2 GB) and optional DFT outputs (~50 GB) fit well under 1 TB;
1 TB (10K credits) leaves room for trajectories.
