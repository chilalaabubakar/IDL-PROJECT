# Compute estimate for the rest of the project (upper plan)

Method (as in an ACCESS proposal): for each piece of work, count the model variants we expect
to train, the epochs each needs, and the measured time per epoch.

    GPU-hours = variants × epochs × epoch time

The team chose the **upper plan** as the plan of record: 64 screening variants for the 3D
model, 3,000 sampler steps, the optional DFT check at its upper end, the remaining 2D work
included, and a 20% margin on all GPU and CPU work (except DFT, already taken at its upper
end).

Credits follow the PSC Explore guide: **200,000 credits ≈ 3,700 GPU-hours** (54 credits per
GPU-hour), **10,000 credits ≈ 1 TB of storage**. ACCESS describes one credit as roughly one
CPU core-hour; the exact rate for Bridges-2 RM should be checked in the
[exchange calculator](https://allocations.access-ci.org/exchange_calculator).

All tables are printed by `python docs/compute/budget.py`; change an assumption there and
rerun.

## Bottom line

| Resource | Amount | ACCESS credits |
|---|---|---|
| GPU | (522 3D + 22 2D) × 1.2 = **653 GPU-h** | 35,288 |
| CPU | 1,551 × 1.2 + 5,000 (DFT) = **6,861 core-h** | ~6,861 |
| Storage | **1 TB** | 10,000 |
| **Total** | | **≈ 52,100 (26% of 200K)** |

**Safety check:** if the week-1 benchmarks show every GPU job running twice as slowly as
planned, the GPU line doubles to 1,306 GPU-h (70.6K credits) and the total is about 87K
credits, still 44% of an Explore allocation.

For reference, the lower bound (32 screening variants, 1,000 sampler steps) is 373 GPU-h.

## By phase (weeks of the 3D plan)

| Weeks | Milestone | GPU-h | Core-h | Credits |
|---|---|---|---|---|
| 1–3 | Finish the 2D paper (optional runs) | 26 | 0 | 1,420 |
| 1–2 | M0 Setup: allocation, potential check, benchmarks | 18 | 0 | 973 |
| 2–5 | M1 Data: silica glasses, detectors, stability checks | 0 | 882 | 882 |
| 4–8 | M2 Model: 64-variant screening, top 6, sampler-step calibration | 156 | 0 | 8,406 |
| 7–11 | M3 Main results: full runs, seeds, all sampling rows, baselines | 429 | 799 | 23,990 |
| 10–13 | M4 Physics checks, energy barriers, DFT | 24 | 5,180 | 6,477 |
| 12–16 | M5 Paper: figures, code and data release | 0 | 0 | 0 |
| | **Total** (+ 10,000 for storage) | **653** | **6,861** | **52,148** |

The busiest phase (M3) needs 429 GPU-hours in about five weeks: one GPU running 12 hours a
day, or 4 GPUs for under five days. The 882 core-hours of M1 take about 7 hours on one
128-core Bridges-2 node.

## What the numbers rest on

| Quantity | Value | Source |
|---|---|---|
| 2D training, EGNN | 0.27–0.37 s per update (batch 16 × 256 atoms), 16 s per epoch | measured, our 11 Kaggle/Colab T4 runs |
| 2D evaluation row (256 samples, 200 steps, relaxed) | 15 min | measured, median of 73 rows |
| 3D training, NequIP (DM2's conditional settings, 1.0 M parameters) | 1.2 s per update (batch 16 × 576 atoms) on a V100 | our CPU ratio applied to our T4 timings: NequIP costs 2.9× our EGNN per atom, and our EGNN takes 66–90 µs per atom per update on a T4, so ~1.8–2.4 s per update on a T4 and ~0.9–1.2 s on a V100 (about twice a T4). We plan with the slow end. DM2's paper gives no training time |
| NequIP vs our EGNN, per atom | 2.9× per training step | measured on CPU, `bench_models.py` (DM2's model, 8.3 s per step for 3,000 atoms); our conditional model takes 6.7 s on the same kind of CPU (`scripts/silica/train.py`), so it is no slower |
| NequIP training memory | ~0.5 GB per 576-atom glass (2.5 GB per 3,000 atoms) | measured on CPU; batch 16 × 576 atoms fits a 16 GB T4 |
| NequIP forward pass vs training step | 0.21× | measured on CPU, `bench_models.py` |
| 3D EGNN | 0.35 s per update | 2D EGNN measurement, 3D neighbour count |
| Silica MD (8 Å Born + DSF Coulomb) | 200,000 atom-steps per second per core (2.9 ms per step for 576 atoms) | measured, LAMMPS, `in.silica_md_bench` |
| One 3D epoch | 600 training glasses / batch 16 = 37.5 updates: NequIP 45 s, EGNN-3D 13 s | derived |
| One full training run | 50,000 updates = 1,333 epochs = 16.7 GPU-h (NequIP) | derived (same length as our 2D runs) |

These are planning numbers. The first week of the 3D work replaces them with timings on the
GPU we are given: every training run writes its seconds per update and GPU name to
`timing.json` (`docs/SILICA_QUICKSTART.md`). On PSC's H100 nodes, which reportedly cost twice as many credits per hour,
the faster GPU should roughly cancel the higher rate.

## 3D model training

The 64 screening variants are every combination of two values each for: maximum training
noise (0.75 or 1.0 Å), training region size (fixed 6 Å or a 5–8 Å range), network size
(DM2's or half), number of layers (3 or 4), learning rate (2×10⁻⁴ or 5×10⁻⁴) and label
dropout (0.1 or 0.2). Short runs pick the best, as in the guide's example (100 ablations × 5
epochs, top 10 × 10 more, top 2 × 50 more).

| NequIP stage | Variants | Epochs each | Epochs | GPU-h |
|---|---|---|---|---|
| Screening | 64 | 133 | 8,512 | 106.4 |
| Top 6 continue | 6 | 267 | 1,602 | 20.0 |
| Top 2 continue to full length | 2 | 933 | 1,866 | 23.3 |
| Best variant, 2 more seeds | 2 | 1,333 | 2,666 | 33.3 |
| Model without labels (for baselines) | 1 | 1,333 | 1,333 | 16.7 |
| **Total** | | | **15,979** | **199.7** |

| EGNN-3D stage (symmetry test in 3D) | Variants | Epochs each | Epochs | GPU-h |
|---|---|---|---|---|
| Screening | 4 | 133 | 532 | 1.9 |
| Best variant, 3 seeds | 3 | 1,333 | 3,999 | 14.6 |

## 3D sampling and checks

Each row is 256 generated samples; rows with guidance run the network twice per step. The
plan assumes DM2's 3,000 sampler steps. If the calibration experiment picks fewer steps,
sampling costs drop to 91 GPU-h (1,000 steps) or 28 GPU-h (300 steps).

| Item | Rows | GPU-h |
|---|---|---|
| Main table: 3 defect types × w = 2, 6 × 3 seeds | 18 | 121.0 |
| Guidance sweep: 3 types × w = 0, 1, 4 | 9 | 60.5 |
| Region-size sweep: 3 types × 2 extra sizes | 6 | 40.3 |
| Baselines without labels (chance, clamp, RePaint) | 9 | 30.2 |
| EGNN-3D rows: 3 types × 3 seeds | 9 | 17.6 |
| Step-count calibration (100–3,000 steps, 64 samples) | 4 | 1.2 |
| Physics checks: MACE re-relaxation of ~2,300 structures, vibrations | | 20 |
| Setup, benchmarks and debugging | | 15 |

3D total before the margin: 522 GPU-h.

## CPU: simulations and baselines

| Item | Core-h |
|---|---|
| MD dataset: 550 glasses at 10 K/ps (0.58 h each), 200 at 100 K/ps (0.24 h), 50 at 1 K/ps (3.96 h), 10 large (1,536 atoms) at 1 K/ps; +10% reruns | 735 |
| Local melt-quench baseline: 3 types × 2 sizes × 256 attempts (0.42 h each) | 641 |
| Relaxing generated samples in LAMMPS: ~60 rows × 256 | 25 |
| Hand-built baseline and energy-barrier (NEB) application | 150 |
| Subtotal × 1.2 margin | 1,861 |
| DFT check (20–30 cells of 192 atoms, upper end) | 5,000 |
| **Total** | **6,861** |

The melt-quench baseline can be cut by about two-thirds by excluding frozen–frozen atom pairs
from the force calculation.

## 2D work left

Planned on Kaggle's free T4 GPUs; it is included in the request so that it can also run on
the cluster.

| Item | Runs/rows | GPU-h |
|---|---|---|
| Train with a range of region sizes (D2 and D1− recipes) | 2 | 8.9 |
| Size test on 1,024-atom glasses | 5 | 5.0 |
| Slowly cooled hosts | 5 | 1.2 |
| Choose w per model on validation data | 18 | 4.5 |
| Evaluate the region-range models | 6 | 1.5 |
| Larger region for D1+ | 3 | 0.8 |
| **Total** | | **21.9** |

## Storage

Datasets (~800 glasses plus dumps for analysis, ~10 GB), checkpoints (~2.4 MB each; the 64
screening runs and their continuations add a few hundred), generated samples (~0.5 GB at
3,000 steps with saved trajectories for figures) and DFT outputs (~50 GB) fit well under
1 TB. 1 TB (10K credits) leaves room for MD trajectories.
