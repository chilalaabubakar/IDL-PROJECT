# 3D silica: how to start training

This guide covers the code for the 3D step of the project (`docs/SILICA_3D_PLAN.md`): data,
defect labels, the conditional 3D model and its training, on Kaggle or on PSC Bridges-2.
Sampling and scoring in 3D come next and are not here yet.

## What there is

| Piece | File | Status |
|---|---|---|
| DM2's 24 glasses, split and labelled | `data/sio2_dm2/` (`scripts/silica/import_dm2.py`) | in the repository |
| Defect detectors (Si(III), Si(V), NBO, O(III), 2-rings, 3-rings) | `src/glassdiff/silica/defects.py`, `configs/silica/defects.yaml` | frozen: Si–O cutoff 2.2 Å |
| Training requests (centre / patch / host; R_out drawn from 5–8 Å; label dropout 15%) | `src/glassdiff/silica/requests.py` | as in 2D |
| Conditional model: DM2's NequIP plus noise level, held atoms, defect label and cooling rate | `src/glassdiff/silica/model.py` | 1.0 M parameters; equivariance checked in `tests/test_silica.py` |
| Training script (resumable, EMA, screening variants, timing) | `scripts/silica/train.py`, `configs/silica/train_*.yaml` | tested on CPU |
| Our 576-atom glasses (LAMMPS melt-quench with DM2's potential) | `src/glassdiff/silica/md.py`, `scripts/silica/make_glasses.py`, `configs/silica/md.yaml` | potential checked; glasses not made yet |
| Launchers | `notebooks/kaggle_silica_train.ipynb`, `slurm/silica/*.sbatch` | Kaggle notebook dry-run locally |

Species are 0 = O and 1 = Si, lengths are in Å and cooling rates in K/ps throughout.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -e ".[silica,dev]"      # e3nn 0.4.4, torch_geometric, ase, DM2's graphite
pip install -e ".[md]"              # only to make glasses (LAMMPS; see docs/DATASET.md §4)
pytest -q tests/test_silica.py      # 12 tests, ~10 s
```

## Step 1: train on DM2's glasses (can start now)

DM2 published 24 glasses of 3,000 atoms, 6 each at 0.1, 1, 10 and 100 K/ps, made with the
potential we use. `data/sio2_dm2` splits every rate 4 / 1 / 1 into train / val / test
(16 / 4 / 4 glasses). Defects in the 16 training glasses: 965 Si in 3-rings, 160 NBO,
19 Si(III) or Si(V), 30 Si in 2-rings. That is enough to get the training pipeline and the
timings right, but too few rare defects for the main study, which uses our own glasses
(step 3).

**On Kaggle** (free T4 GPUs): open `notebooks/kaggle_silica_train.ipynb`, set *Accelerator
→ GPU T4 x2* and *Internet → On*, add the `GITHUB_TOKEN` secret, and run *Save & Run All*.
It trains the conditional model on one GPU and the unconditional baseline on the other,
pushes checkpoints to `results/kaggle_silica_<date>/` every 3 hours, and continues from
there in the next session. `SMOKE = True` gives a 5-minute test first.

**Anywhere else** (one GPU):
```bash
python scripts/silica/train.py --config configs/silica/train_cond_dm2.yaml run_dir=runs/silica/dm2_s0
python scripts/silica/train.py --config configs/silica/train_uncond_dm2.yaml run_dir=runs/silica/uncond_dm2_s0
```
Run the same command again to continue after a stop. `max_minutes=...` stops cleanly before
a time limit.

**What to look at.** `runs/.../log.csv` has, every 500 updates, the training loss and
`valn_<sigma>`: the fraction of the added noise the EMA model fails to remove at noise level
sigma (Å) on the validation glasses. 1.0 means nothing learned; lower is better. On CPU, a
small model trained for 150 updates already brings `valn_0.2` from 1.0 to 0.39.
`timing.json` holds the measured seconds per update and the GPU name: send these numbers to
whoever updates `docs/compute/README.md`.

## Step 2: check the potential, then make our glasses

DM2 does not publish its LAMMPS input, so `src/glassdiff/silica/md.py` rebuilds it from the
papers: Bouhadja–Jakse–Pasturel Born–Mayer–Huggins potential (Table III of
arXiv:1403.1640), charges Si +2.4 and O −1.2, DSF Coulomb (0.25 Å⁻¹), 8 Å cutoffs, units
real, 1 fs steps, and DM2's melt-quench protocol. The check runs 20 ps of NPT at 300 K on
two of DM2's glasses; a wrong unit or parameter would make them expand, collapse or lose
their tetrahedra:

```bash
python scripts/silica/check_potential.py --config configs/silica/md.yaml     # ~10 min
```
Result (`results/silica/potential_check.json`): density changes by less than 0.5% and every
Si stays four-fold, so the rebuilt input holds DM2's glasses where they are.

Then make the 1,000 glasses of `configs/silica/md.yaml` (576 atoms; train 400 at 10 K/ps +
200 at 100 K/ps, val 50, test 100 at 10 K/ps + 50 at 1 K/ps), about 565 core-hours:

```bash
python scripts/silica/make_glasses.py --config configs/silica/md.yaml dry=true workers=4   # 1 min test
python scripts/silica/make_glasses.py --config configs/silica/md.yaml workers=64
sbatch -A <allocation> slurm/silica/md.sbatch          # or on Bridges-2 (RM-shared, 64 cores)
```
Each glass is saved as soon as it is done, so a stopped job loses at most the glasses in
progress; run it again to continue. Glasses that fail the gates (density 2.0–2.4 g/cm³,
≥ 95% four-fold Si, no pair closer than 1.2 Å) are replaced with new seeds. When a split is
complete it is assembled into `data/sio2_576/<split>.npz` and labelled, and the defect
counts are printed with the week-5 gate (≥ 300 training examples per main defect type).
`scripts/silica/label_defects.py` relabels a dataset on its own.

## Step 3: train on our glasses (the main model)

```bash
python scripts/silica/train.py --config configs/silica/train_cond.yaml run_dir=runs/silica/cond_s0
sbatch -A <allocation> slurm/silica/train.sbatch      # Bridges-2 V100; resubmits itself until done
```
Batch 16 × 576 atoms uses about 8 GB, so it fits a T4 or a V100. A full run is 50,000
updates (1,333 epochs of 600 glasses), about 17 GPU-hours on a V100 by our estimate.

**Screening funnel** (`configs/silica/screen.yaml`, the compute plan's upper plan): 64
variants (2 values each of maximum noise, R_out range, network width, layers, learning rate
and label dropout), 5,000 updates each, then the best 6 to 15,000, the best 2 to 50,000,
then 2 more seeds of the winner:

```bash
sbatch -A <allocation> slurm/silica/screen.sbatch                       # 64 variants, 8 at a time
python scripts/silica/rank_runs.py runs/silica_screen/v* --top 6 --out runs/silica_screen/top6.txt
sbatch -A <allocation> --array=0-5 --export=ALL,LIST=runs/silica_screen/top6.txt,N=15000 slurm/silica/screen.sbatch
python scripts/silica/rank_runs.py $(cat runs/silica_screen/top6.txt) --top 2 --out runs/silica_screen/top2.txt
sbatch -A <allocation> --array=0-1 --export=ALL,LIST=runs/silica_screen/top2.txt,N=50000 slurm/silica/screen.sbatch
```
Runs are ranked by `valn_0.2` on the last row of their `log.csv`. A single variant can also
be run by hand: `scripts/silica/train.py ... variant=17 n_updates=5000 run_dir=...`.

**Bridges-2 setup, once:** clone the repository to `$PROJECT`, edit the two paths at the top
of `slurm/silica/env.sh`, run `bash slurm/silica/env.sh install` and `mkdir -p slurm/logs`,
and submit every job from the repository root.

## Measured so far (CPU, 4 cores)

| What | Value |
|---|---|
| Model size (DM2's conditional settings) | 1,017,840 parameters |
| One training step, one 3,000-atom glass | 6.7 s |
| Training memory | ~2.5 GB per 3,000-atom glass, ~0.5 GB per 576-atom glass |
| Defect labelling | 0.01 s per 3,000-atom glass |
| LAMMPS, 576 atoms | 2.9 ms per MD step on one core (0.58 h per glass at 10 K/ps) |
