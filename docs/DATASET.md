# Dataset plan: 2D Kob–Andersen glasses for controlled defect generation

## Bottom line

1. **No public dataset matches the proposal.** The proposal needs about 1,000 independent,
   quenched 2D binary Kob–Andersen (KA) 65:35 glasses with N = 256. I checked the DM2
   data repository, GlassBench, and the datasets cited in Yang & Schwalbe-Koda (2025). None
   of them are 2D binary KA glasses (see the table of external datasets below).
2. **We generate the dataset ourselves with LAMMPS, and it is cheap.** A pilot run in this
   repo (`docs/pilot/`) shows that one N = 256 glass takes **2.9 s on one CPU core** with the
   proposal's fast quench. The full 1,000-glass set therefore takes **under 1 CPU-hour**,
   and every auxiliary set together takes under 2 CPU-hours. None of the pilot glasses
   crystallized.
3. **External data has two supporting uses:** the DM2 a-SiO2 structures and pretrained
   models for the Stage 0 reproduction and the later move to 3D, and (optionally)
   GlassBench's 2D ternary KA as an out-of-distribution check.

---

## 1. Primary dataset: `ka2d_256`

### 1.1 Physical model (fixed; do not change after generation)

| Quantity | Value |
|---|---|
| Model | Binary Lennard-Jones, Kob–Andersen parameters (Kob & Andersen 1995) at the 2D composition of Brüning et al. (2009) |
| Composition | 65:35 → **166 A + 90 B** (exactly 64.8 : 35.2) |
| σ_AA, σ_AB, σ_BB | 1.0, 0.8, 0.88 |
| ε_AA, ε_AB, ε_BB | 1.0, 1.5, 0.5 |
| Cutoff | 2.5 σ_αβ, energy-shifted (`pair_modify shift yes`) |
| Masses | all 1.0 |
| Number density ρ | 1.2 → box L = √(256/1.2) = **14.606 σ** (square, periodic in x and y) |
| Units | LJ (σ_AA, ε_AA, m, τ = σ√(m/ε)); time step 0.005 τ |

Why 65:35: the standard 80:20 KA mixture crystallizes in 2D, but 65:35 does not
(Brüning et al., arXiv:0811.2995).

### 1.2 Generation protocol (one glass per seed)

| Step | LAMMPS | Notes |
|---|---|---|
| 1. Initialize | `lattice sq 1.2`, 16×16 cells, `set type/subset 2 90 <seed>` | Gives an exact atom count and composition. Random insertion lost atoms in the pilot, so don't use it. |
| 2. Remove overlaps | `min_style fire; minimize 0 1e-6 ...` | |
| 3. Melt | NVT Nosé–Hoover, T = 2.0, Tdamp = 0.1 τ, 20,000 steps (100 τ) | `fix enforce2d`. Record the MSD to confirm the system forgets the lattice (MSD ≫ 1 σ²). |
| 4. Quench | Linear ramp T = 2.0 → 0.01 at **1e-2 per τ** (39,800 steps) | This is the proposal's "fast cooling". |
| 5. Inherent structure | `minimize 0 1e-8 100000 1000000` (FIRE) | **Training target = the inherent structure (T = 0).** Relaxing a natural glass then moves no atoms, so "movement during relaxation" is a clean realism metric for generated samples. |
| 6. Save | wrapped positions, types, box, per-atom PE, seed, protocol hash | Optionally also save the T = 0.01 thermal snapshot taken before minimization. |

### 1.3 Size, splits and seeds

* 1,000 glasses split 800 / 100 / 100 (train / val / test) **by glass**, with disjoint seed
  ranges per split (e.g., train 0–799, val 10000–10099, test 20000–20099).
* Each glass yields 256 local environments, and the denoiser is local. DM2 trained on
  24 glasses with 3,000 atoms, so 800 × 256 ≈ 205k training environments is plenty.

### 1.4 Quality gates applied to every glass (reject, log and replace)

| Check | Threshold | Pilot result |
|---|---|---|
| Composition | exactly 166 A / 90 B | ✓ |
| Crystallinity: global hexatic order \|Ψ6\| | < 0.3 | 0.007–0.116 for N = 256 (12 glasses) |
| Minimum pair distance | > 0.7 σ_AB | — |
| PE/atom | within 5 SD of the ensemble mean | −3.675 ± 0.014 |
| Residual max force after minimization | < 1e-6 | — |

### 1.5 Labels: defect definitions, measured in the pilot

All thresholds are **computed on the train split only and frozen in `meta.json` before any
model is trained**, as the proposal requires. The numbers below come from 12 pilot glasses
(3,072 atoms) and will shift slightly on the full set.

**Coordination statistics (N = 256, quench at 1e-2/τ):**

| Descriptor | Distribution |
|---|---|
| Voronoi coordination (all atoms) | 4: 0.42% · 5: 27.0% · 6: 44.8% · 7: 27.7% · 8: 0.07% |
| Cutoff coordination, A atoms (r < 1.4 σ_αβ) | 5: 1.05% · 6: 55.3% · 7: 42.6% · 8: 1.10% |
| Cutoff coordination, B atoms | 4: 16.1% · 5: 68.6% · 6: 15.1% · 7: 0.19% |
| B atoms: number of B neighbours (r < 1.2) | 0: 54.1% · 1: 34.8% · 2: 10.2% · 3: 0.93% |
| Delaunay circumradius (void size) | median 0.571 · 95%: 0.674 · 99%: 0.722 · 99.9%: 0.766 |

**Proposed definitions.** These meet the proposal's target of about 1–5% of atoms:

| Label | Definition | Frequency (pilot) | Per glass |
|---|---|---|---|
| **D1−** (under-coordinated) | A atom with cutoff CN ≤ 5 | 1.05% of A | ≈ 1.7 |
| **D1+** (over-coordinated) | A atom with cutoff CN ≥ 8 | 1.10% of A | ≈ 1.8 |
| D1 (combined) | D1− ∪ D1+ | ≈ 1.4% of all atoms | ≈ 3.6 |
| **D2** (B–B cluster) | B atom with ≥ 2 B neighbours within 1.2 σ | 11.1% of B ≈ 3.9% of all atoms | ≈ 10 |
| D2-strict (optional, harder) | B atom with ≥ 3 B neighbours | 0.93% of B | ≈ 0.8 |
| **D3** (stretch) | Delaunay triangle with circumradius > the train-split 99.9% quantile (≈ 0.77 σ). Its location is the circumcentre. | 0.1% of triangles | — |

Two pilot findings change the definitions:
* **Voronoi coordination is a poor basis for D1.** Its tails (CN 4 or 8) cover only
  about 0.5% of atoms, which is below the proposal's 1–5% target.
* **D1 should be restricted to A atoms.** For B atoms, CN = 4 is common (16%).

In the sensitivity analysis, vary the cutoffs by ±0.05 σ and report how the success rates
move.

### 1.6 Reference-defect patch library

* Built from the **train split only**: for every D1, D2 (and D3) site, store the centre atom
  and its neighbours within 1.5 σ (first shell), as coordinates relative to the centre,
  along with their types.
* Used by the hand-insertion, noise-patch, clamping and RePaint baselines, and by the
  "patch" request mode of the trained model.
* The test split supplies the **natural-defect reference distributions** for the realism
  metrics, so references never overlap with the patches.

### 1.7 File format

```
data/ka2d_256/
  train.npz  val.npz  test.npz
     pos          float64 [M, 256, 2]   inherent structure, wrapped into [0, L)
     types        int8    [M, 256]      0 = A, 1 = B
     box          float64 [M, 2]
     pe_atom      float64 [M, 256]      LAMMPS compute pe/atom
     seed         int64   [M]
     thermal_pos  float64 [M, 256, 2]   T = 0.01 snapshot before minimization
     labels       int8    [M, 256]      0 none, 1 D1-, 2 D1+, 3 D2 (added when frozen)
  meta.json          protocol, quality gates, LAMMPS version, git state, per-split
                     stats and rejection log; "defects": frozen thresholds, digest, rates
  patches_train.npz  rel_pos, types, defect, sizes (concatenated patches)
```

Positions are stored in float64 so that re-relaxing a dataset glass is a no-op. With the
thermal snapshots the N = 256 set is about 10 MB. All four sets are committed in `data/`
(~20 MB); they can also be regenerated exactly from the configs (same seeds give
bit-identical glasses).

### 1.8 Generated sets (actual numbers)

Generated with `scripts/make_dataset.py` (LAMMPS 22 Jul 2025, 3 worker processes):

| Set | Splits | Rejected | PE/atom (train or test) | Wall time |
|---|---|---|---|---|
| `ka2d_256` | 800 / 100 / 100 | 0 | −3.6802 ± 0.0136 | 14 min (train) |
| `ka2d_64` | 1600 / 200 / 200 | 4 (global \|Ψ6\| 0.30–0.36, finite-size fluctuations) | −3.6356 ± 0.0284 | 5 min (train) |
| `ka2d_1024` | test 100 | 0 | −3.6732 ± 0.0074 | 8 min |
| `ka2d_256_slow` | test 200 | 0 | −3.7059 ± 0.0105 | 30 min |

Frozen defect thresholds for `ka2d_256` (digest `1c8714e00c26`, fitted on train):
D1− = A with CN ≤ 5, D1+ = A with CN ≥ 8, D2 = B with ≥ 2 B neighbours within 1.2,
D3 radius 0.766. Per glass on train: D1− 1.74, D1+ 1.43, D2 10.0 (1.05%, 0.86% of A and
11.1% of B atoms, matching the pilot). The train patch library holds 10,545 patches
(1,388 D1−, 1,146 D1+, 8,011 D2) of 6–11 atoms. `ka2d_64` (digest `cb3b93c7c030`):
per glass D1− 0.53, D1+ 0.30, D2 2.09. The slow-cooled set, labelled with the `ka2d_256`
thresholds, has fewer defects (D1− 0.69% of A, D2 9.7% of B, against 0.93% and 11.2% for
the fast-cooled test split), as expected for a better-annealed glass.

The PyTorch potential reproduces the stored LAMMPS per-atom energies to 1e-5
(`tests/test_potential.py`).

---

## 2. Auxiliary sets (same model and protocol)

| Set | Purpose | Size | Measured cost |
|---|---|---|---|
| `ka2d_64` (N = 64: 42 A + 22 B, L = 7.303) | MLP baseline from Table 1, plus an EGNN/MPNN control at the same N so the symmetry ladder is not confounded by system size | 2,000 glasses | 0.6 s/glass |
| `ka2d_1024` (L = 29.21) | Size-transfer test: does a model trained at N = 256 generate at N = 1024? (DM2 shows this works in 3D) | 100 glasses | 15 s/glass |
| `ka2d_256_slow` (quench at 1e-3/τ) | Better-annealed reference / out-of-distribution realism check | 200 glasses | 16 s/glass |
| Hosts for host-conditioned inpainting | The 100 **test** glasses | — | — |

Finite-size note: PE/atom is −3.642 at N = 64 against −3.675 at N = 256. This is why the
symmetry-ladder comparison has to be made at matching N.

---

## 3. External datasets located

| Dataset | Contents | Format / size | License | How we use it |
|---|---|---|---|---|
| **DM2 a-SiO2 data**: [`digital-synthesis-lab/2025-dm2-data`](https://github.com/digital-synthesis-lab/2025-dm2-data) (`simulated_a-SiO2/`); the same files ship in `DM2/demo/demo_training/simu_data/` | 24 a-SiO2 glasses, 3,000 atoms each, 6 per cooling rate (0.1, 1, 10, 100 K/ps), Jakse potential | LAMMPS data, real units, with charges; ~430 kB each | MIT | Stage 0 reproduction; future 3D extension |
| **DM2 pretrained models**: [`digital-synthesis-lab/DM2`](https://github.com/digital-synthesis-lab/DM2) `demo/model/` | `gen-a-sio2-uncond-v1.pt`, `gen-a-sio2-cond-v1.pt`, `gen-cu50zr50-v1.pt` | Full pickled torch models, ~2.7 MB each | MIT | Stage 0 sanity check: generate a 300-atom a-SiO2 sample and compare g(r) to the training data (≈ 2 min on a GPU, per the DM2 README) |
| CuZr metallic glass (Wang et al. 2020): [figshare 12485795](https://figshare.com/articles/dataset/Heterogeneous_thermal_activation_energy_in_Cu-Zr_metallic_glasses/12485795) | Cu50Zr50, 5,000 atoms | — | see figshare | Not needed; future 3D work |
| AET experimental nanoparticle: [`AET-MetallicGlass/Supplementary-Data-Codes`](https://github.com/AET-MetallicGlass/Supplementary-Data-Codes) | Experimental 3D atomic structure | — | — | Not used |
| **GlassBench** ([Zenodo 10118191](https://zenodo.org/records/10118191)) | 3D KA 80:20 and a **2D ternary** KA mixture (KA2D); equilibrium supercooled-liquid trajectories for dynamics prediction | `GlassBench.zip`, 6.0 GB | CC-BY 4.0 | Optional stretch: an out-of-distribution test of the detectors and realism metrics on a different 2D glass former. **Not usable as training data**: the system is different (ternary, not quenched inherent structures). |

---

## 4. Tooling (verified in this session)

* **LAMMPS 22 Jul 2025** installs with `pip install lammps mpich`. The wheel needs
  `libmpi.so.12`, which the `mpich` wheel provides. Add that library's directory to
  `LD_LIBRARY_PATH` (here it was `/usr/local/lib`). The wheel includes the `VORONOI`,
  `EXTRA-COMPUTE` and `OPENMP` packages. conda-forge `lammps` is an alternative.
* Analysis uses `numpy` and `scipy` (Delaunay with periodic images; see `docs/pilot/pilot_ka2d.py`).
* Evaluation uses our own **PyTorch KA energy + batched FIRE minimizer**
  (`glassdiff.physics`), tested against the LAMMPS per-atom energies to 1e-5.

To regenerate the datasets:

```bash
pip install -e ".[dev,md]"
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH   # wherever libmpi.so.12 landed
for c in ka2d_256 ka2d_64 ka2d_1024 ka2d_256_slow; do
  python scripts/make_dataset.py --config configs/data/$c.yaml workers=4
done
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_64
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_1024 thresholds_from=data/ka2d_256
python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256_slow thresholds_from=data/ka2d_256
python scripts/build_patches.py --config configs/defects/v0.yaml data=data/ka2d_256
python scripts/build_patches.py --config configs/defects/v0.yaml data=data/ka2d_64
```

To reproduce the pilot:

```bash
pip install lammps mpich numpy scipy
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH   # wherever libmpi.so.12 landed
python docs/pilot/pilot_ka2d.py 0.01 12        # N=256, fast quench
python docs/pilot/pilot_ka2d.py 0.001 4        # N=256, slow quench
python docs/pilot/pilot_ka2d.py 0.01 12 64     # N=64
```

Raw outputs: `docs/pilot/out_*.txt`.
