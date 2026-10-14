# Plan: location-specific defects in 3D amorphous silica

The 2D Kob–Andersen work answers the method questions: a conditional diffusion model can
place a requested defect at a chosen site inside an existing glass. It does so more often
than hand insertion or local melt-quench, and the result holds under every evaluation
setting we varied. A materials journal (npj Computational Materials, where DM2 was published)
will ask for three more things:

1. a real 3D material;
2. defects that matter physically;
3. evidence that the generated defects behave like natural ones.

This plan adds those three things for amorphous silica (a-SiO₂), reusing as much of the
2D pipeline and of DM2 as possible.

**Target paper (working title):** *Placing rare defects in well-annealed amorphous silica with
a conditional diffusion model.*

**Estimated effort:** 14–16 weeks for the six of us. Compute (§8, `docs/compute/`):
~350 GPU-hours (up to ~630) and ~1,550 CPU core-hours, about 30K–50K ACCESS credits with
1 TB of storage. This needs a cluster allocation; Kaggle alone is not enough.

---

## 1. Claims the paper should make

| # | Claim | Evidence |
|---|---|---|
| C1 | The model places a requested silica defect at a chosen site inside an existing glass, after relaxation, more often than hand construction or local melt-quench. | Task B success table per defect class, 3 training seeds, sensitivity checks (§5) |
| C2 | The surroundings of generated defects are as natural as those of defects that form in MD. | Local realism against natural-vs-natural floors (§5.2) |
| C3 | Generated defects are physically equivalent to natural ones, not just geometrically. | Re-relaxation with an independent machine-learned potential, defect energies, vibrational signatures (§5.4) |
| C4 | This makes rare defects in slowly cooled glass cheap: something MD cannot do at a chosen site. | Cost per success against MD, and one property distribution computed from ≥ 100 generated defects (§6) |

C4 is the selling point. At 1 K/ps an MD glass has about one Si with the wrong coordination
per 2,000 Si atoms (measured in Stage 0 on DM2's glasses), so waiting for MD to put one at a
chosen site is hopeless. The model can be asked for exactly that.

## 2. System and data

### 2.1 Potential and protocol

To stay comparable with DM2, we use the potential its silica data was made with:
- **Potential:** Bouhadja, Jakse and Pasturel (J. Chem. Phys. 138, 224510, 2013), a
  Born–Mayer–Huggins form fitted to ab initio data, with charges Si +2.4 and O −1.2.
- **Cutoffs:** 8 Å short-range cutoff; Coulomb by the damped shifted force (DSF) method with
  damping 0.25 Å⁻¹ and an 8 Å cutoff.
- **Melt-quench** (from the DM2 Methods), with a 1 fs timestep:
  1. random SiO₂ placement (PACKMOL);
  2. 50 ps NVT and 50 ps NPT at 300 K;
  3. 50 ps NPT melt at 5000 K;
  4. NPT quench to 300 K at the chosen rate;
  5. 100 ps NPT at 300 K.

The DM2 repository contains neither LAMMPS inputs nor the potential table, so we write the
LAMMPS input from the paper. We check it by reproducing the density (2.13 g/cm³ at 1 K/ps)
and the coordination numbers of DM2's published glasses.

We also use, as is:
- DM2's 24 published glasses: 3,000 atoms each, 6 at each of 0.1, 1, 10 and 100 K/ps.
  They serve as a held-out global reference.
- DM2's pretrained models, as an unconditional baseline. They already run on our CPUs
  (Stage 0).

### 2.2 Our dataset

| Split | Glasses | Atoms each | Cooling rates | Use |
|---|---|---|---|---|
| train | 400 + 200 | 576 (192 SiO₂; box ≈ 20.7 Å) | 10 K/ps (400), 100 K/ps (200) | training; patch library |
| val | 50 | 576 | 10 K/ps | model selection |
| test | 100 + 50 | 576 | 10 K/ps (100), 1 K/ps (50) | Task B hosts, natural reference defects |
| transfer | 10 | 1,536 | 1 K/ps | size transfer (stretch) |

- **Why 576 atoms:** Task B rebuilds a sphere of radius R_out ≈ 6–8 Å, which must fit well
  inside half the box (10.3 Å). It also keeps MD and training cheap.
- **Why the 100 K/ps glasses:** they make rare defects about 3× more common (§2.3). The model
  is conditioned on cooling rate (as DM2's conditional model is), so it can still be asked
  for 1 K/ps surroundings.
- **Cost, from a LAMMPS benchmark** (`docs/compute/in.silica_md_bench`: 200,000
  atom-steps per second per core with 8 Å Born + DSF Coulomb): 2.9 ms per step for 576
  atoms, so 0.24 h per glass at 100 K/ps, 0.58 h at 10 K/ps and 4.0 h at 1 K/ps on one core,
  ~735 core-hours for the whole set including reruns.

### 2.3 Defect classes

Fractions measured in Stage 0 on DM2's glasses (`results/stage0/metrics.json`):

| Class | Definition (cutoff Si–O 2.2 Å, first minimum of g_SiO) | 1 K/ps | 10 K/ps | 100 K/ps | Why it matters |
|---|---|---|---|---|---|
| **Si(V)** / **Si(III)** | Si with 5 or 3 O neighbours | 0.05% of Si | 0.10% | 0.32% | Si(III) is the precursor of the E′ centre (oxygen vacancy); Si(V) appears under stress and densification |
| **NBO** / **O(III)** | O with 1 or 3 Si neighbours | 0.5% of O are not 2-fold | 1.2% | 1.9% | the NBO is the precursor of the non-bridging oxygen hole centre; optical absorption |
| **2-ring** (edge-sharing tetrahedra) | two Si sharing two O (Si–Si < 2.8 Å) | 0.0017 per Si | 0.0047 | 0.0083 | strained, reactive; DM2's own samples over-produce them (Stage 0) |
| **3-ring** | 3-membered Si–O ring (shortest-path ring) | to measure | | | gives the Raman D2 line at 606 cm⁻¹, which is used experimentally to estimate fictive temperature |

- **Primary classes:**
  - **3-rings** and **NBO**, which have enough training examples and an experimental link;
  - **Si(V)/Si(III)**, the rare, high-value class. In 400 glasses at 10 K/ps there are only
    about 80 of them, so the 100 K/ps glasses are needed.
- **Stretch class:** 2-rings.
- **Gate (end of week 5):** every primary class has ≥ 300 training instances, or we merge or
  drop it.

As in 2D:
- The detectors are frozen on the train split.
- Patches come from train, and reference defects and hosts from test.
- We run the defect-stability check before training: does a natural defect survive noise
  plus relaxation? In 2D, D1+ failed every method because natural D1+ sites did not survive
  being transplanted. Checking first saves us from repeating that.

## 3. Model

### 3.1 Backbone

- **Main backbone:** DM2's E(3)-equivariant NequIP denoiser (`graphite`: 3 message-passing
  layers, 5 Å cutoff, 12 neighbours). It reproduces silica's tetrahedra (Stage 0), which a
  distance-only network may struggle with.
- **Ablation:** a 3D port of our EGNN-PBC. It is cheaper and extends the symmetry ladder to
  3D: does higher-order equivariance matter for silica defects?

### 3.2 What we add to DM2 (from the 2D model)

| Input | How | 2D lesson behind it |
|---|---|---|
| Noise level σ | Fourier embedding, as in our 2D model | DM2 is σ-blind; RePaint and guidance need a defined noise level |
| Pinned atoms | per-atom flag embedding | Task B pins everything outside R_out |
| Defect label at the site | per-atom label embedding, dropped 15% of the time for classifier-free guidance | the main conditioning of the 2D work |
| Cooling rate | global embedding (as in DM2's conditional model) | ask for 1 K/ps surroundings while training on faster glasses |

### 3.3 Training recipe

- **Noise range σ ∈ [0.001, 1.0] Å, matching where the sampler starts.** DM2 trains with
  σ ≤ 0.75 Å but samples from 1.0. In 2D, the same mismatch made defects vanish on
  relaxation, and retraining with σ ≤ 1 raised D2 success from 0.54 to 0.79.
- **Request types** as in 2D (centre / patch / host), but **R_out drawn from 5–8 Å** rather
  than fixed. In 2D the model was trained at one R_out, and the sensitivity checks showed
  R_out strongly affects realism. Training on a range lets us choose R_out at sampling time.
- **Three training seeds from the start.** In 2D, two training runs with the same validation
  loss gave D2 success 0.56 vs 0.34. Seed 0 alone is not enough to report.
- **Infrastructure:** the resumable multi-session trainer from `notebooks/kaggle_seeds.ipynb`.

### 3.4 Sampler

DM2's score dynamics with our clamp / RePaint / guidance strategies.

The step count acts like a cooling rate (2D round 2), and DM2's own 3,000-step samples look
like a 100 K/ps glass (Stage 0). A **calibration experiment** therefore comes first:
- sweep the step count (100 / 300 / 1,000 / 3,000);
- match the fictive temperature of the generated surroundings to the requested cooling rate
  (density, Si–O–Si angle width, 3-ring fraction).

## 4. Tasks and baselines

**Task B (headline):** in a held-out glass, rebuild the sphere of radius R_out around a
requested site and ask for a defect of class X at its centre. **Task A** (secondary): a whole
box with the defect at the centre.

| Baseline | What it is |
|---|---|
| B1 unconditional + filter | DM2-style generation; keep samples where the defect appears by chance |
| B2 hand construction | current practice: remove or insert an O, or insert a ring patch from the library, then relax |
| B4 clamp, B5 RePaint | training-free placement with the unconditional model |
| B6 local melt-quench | melt the sphere R_out at 5000 K with the outside fixed, quench at 10 K/ps, relax (LAMMPS) |
| MD waiting time | how long plain MD at a given cooling rate needs to produce class X at a chosen site; the reference for C4 |

## 5. Evaluation

Everything is ported from the 2D pipeline: 256 samples per row, bootstrap CIs, three
training seeds (mean ± spread).

**5.1 Success.** Relax every sample with the training potential, then apply the frozen
detector at the requested site (r_tol). Relaxation runs as a batched PyTorch implementation
of the potential with FIRE on GPU (as for KA), checked against LAMMPS minimisation.

**5.2 Local realism.** In windows of two radii (5 and 7 Å; in 2D the window size changed
the comparison with hand insertion), compute W1 against natural defects of the same class.
The features are:
- per-atom energy;
- Si–O bond lengths;
- O–Si–O and Si–O–Si angles;
- ring sizes.

The floor is natural vs natural. A **seam check** measures the same features in a shell
around R_out, because in 2D a larger R_out may only have moved the seam out of view.

**5.3 Global realism.** Compared with MD at the matched cooling rate:
- partial g(r) and the neutron structure factor S(q), also compared with experiment;
- density;
- coordination fractions;
- the ring-size distribution.

**5.4 Physical validation (new; the part a materials journal will weigh most).**
1. **Independent potential.** Re-relax generated and natural defects with a machine-learned
   silica potential (GAP-22, Erhard et al., npj Comput. Mater. 2022) or a foundation model
   (MACE-MP-0). Defects must survive, and their excess energies must match the natural ones
   (two-sample test).
2. **Vibrational signature.** Compute the local vibrational density of states of 3-ring
   windows from the dynamical matrix. The ring-breathing mode near the D2 line should match
   between generated and natural rings, and hand-built rings are the comparison.
3. **DFT spot check (stretch).** 20–30 periodic cells of 192 atoms with NBO or Si(III)
   defects (CP2K, PBE). Generated defects should show the same gap states as natural ones.
   This needs cluster CPU time.

**5.5 Sensitivity.** Si–O cutoff ±0.05 Å, r_tol, r_loc and R_out, using
`scripts/sensitivity.py` once it works in 3D.

## 6. Application (choose one at the end of M3)

| Option | What we show | Cost |
|---|---|---|
| **A. Barrier statistics** (preferred) | NEB barriers for Si(V) → Si(IV) relaxation or bond switching, for ≥ 100 generated and ≥ 100 natural defects: the distributions agree, and the generated set costs far less | ~2,000 NEB runs with the classical potential, CPU |
| B. Cost per success | CPU/GPU-hours per correctly placed defect at a chosen site: model vs local melt-quench vs MD waiting, at 1 K/ps. Unlike 2D, MD is expensive here, so the cost claim becomes meaningful | small; the numbers come from §4 |
| C. Pore surfaces | NBO and silanol-site precursors on pore walls (DM2 showed NBOs at pores) | stretch |

Option B goes in the paper whatever we pick; option A or C is the showcase.

## 7. Milestones and gates

| Weeks | Milestone | Gate (go / adjust) |
|---|---|---|
| 1–2 | **M0 Setup.** Cluster allocation; LAMMPS input for the potential, checked against DM2's glasses; benchmark MD per glass and NequIP time per update on the GPU we get; port DM2 training into our trainer (resumable) | Measured costs fit the budget, or we shrink cells, data or model |
| 2–5 | **M1 Data.** Train / val / test glasses; detectors (coordination, NBO, rings) frozen; defect counts per class; patch library; natural floors; defect-stability check per class | ≥ 300 train instances per primary class; natural defects survive noise + relaxation |
| 4–8 | **M2 Model.** 3D code (structures, minimum image, neighbour lists, batched silica potential + FIRE, detectors); conditional NequIP, seed 0; unconditional sanity (g(r), S(q), rings) and step-count calibration | Generated glass within 2× the natural floor on global metrics, or a documented fictive-temperature offset |
| 7–11 | **M3 Main results.** Task B table for the three classes against all baselines; guidance and R_out sweeps; seeds 1–2; sensitivity | C1 holds for ≥ 2 classes across seeds |
| 10–13 | **M4 Physics.** Machine-learned-potential re-relaxation; VDOS; DFT spot check (stretch); the chosen application | C3 holds; application results |
| 12–16 | **M5 Paper.** Figures, writing, code and data release (Zenodo), AI-use statement | — |

**Workstreams** (same five as in 2D):
- **WS1** data and MD (LAMMPS, baselines B2/B6);
- **WS2** detectors, rings, evaluation;
- **WS3** models (NequIP conditioning, EGNN-3D);
- **WS4** sampler and experiments;
- **WS5** physical validation, application, compute and writing.

## 8. Compute budget

Full method, assumptions and tables: `docs/compute/README.md`, printed by
`docs/compute/budget.py` (GPU-hours = variants × epochs × epoch time). Measured anchors:
our 2D training and evaluation times, a CPU benchmark of DM2's NequIP against our EGNN
(`bench_models.py`), DM2's reported A6000 training time, and a LAMMPS silica benchmark.

| Item | Base plan | Where |
|---|---|---|
| NequIP training: 32-variant screen → top 6 → top 2 → 3 seeds, plus a model without labels (11,723 epochs × 45 s) | 147 GPU-h | cluster GPUs |
| EGNN-3D ablation: 4-variant screen, 3 seeds | 17 GPU-h | Kaggle / Colab or cluster |
| Sampling: ~55 rows of 256 samples at 1,000 steps (28 / 91 / 271 GPU-h at 300 / 1,000 / 3,000 steps) | 91 GPU-h | cluster GPUs |
| Physics checks (MACE re-relaxation, vibrations) and setup | 35 GPU-h | either |
| **GPU total with 20% contingency** | **347 GPU-h (upper 627)** | ≈ 19K (34K) credits |
| MD dataset | 735 core-h | cluster CPUs |
| Local melt-quench baseline (1,536 attempts) | 641 core-h | cluster CPUs |
| Relaxations, hand-built baseline, NEB application | 175 core-h | cluster CPUs |
| **CPU total** | **1,551 core-h** (+2,000–5,000 for DFT) | ≈ 1.6K credits |
| Storage | 1 TB | 10K credits |

**Action for week 1:** apply for an NSF ACCESS *Explore* allocation (200K credits ≈ 3,700
GPU-hours on PSC Bridges-2), which covers the upper estimate about four times over. Also ask
about CMU and CMU-Africa clusters. Kaggle's ~30 GPU-hours a week covers the 2D work left
(22 GPU-h) and the EGNN-3D ablation, not the main model.

## 9. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Rare classes (Si(V)/Si(III)) have too few training examples | High | 100 K/ps glasses with cooling-rate conditioning; defect-centred request sampling; merge Si(V)+Si(III) into one "coordination defect" class; gate at week 5 |
| Natural defects do not survive noise + relaxation (as D1+ in 2D) | Medium | defect-stability check before training; define classes by the relaxed state; drop classes that fail |
| NequIP training too slow for 3 seeds | Medium | smaller angular order or fewer layers; smaller cells; EGNN-3D as the main model if it is close enough |
| Seam artifacts at R_out | Medium | train over an R_out range; seam-shell metric; optional short constrained anneal after sampling |
| Results tied to one classical potential | Medium | independent machine-learned potential (§5.4); report both |
| Generated glass hotter than requested (Stage 0: DM2 samples look like 100 K/ps) | High | step-count calibration (§3.4); report the fictive temperature of generated surroundings |
| Training-seed variance (2D D2: 0.34–0.79) | High | three seeds from the start; mean ± spread in every table |
| Scope creep | Medium | core = 3-rings + NBO + Si coordination; everything else stretch |

## 10. What carries over from the 2D code

| Reuse as is | Generalise to 3D | New |
|---|---|---|
| request sampler (modes, pins, labels, CFG dropout); guidance / clamp / RePaint strategies; bootstrap and metrics plumbing; `sensitivity.py`; aggregation; the Kaggle notebooks (resumable training, pushes, dry runs) | `Structures`, minimum image and neighbour lists (2D-specific in `types.py` and `geometry.py`); EGNN-PBC; batched FIRE; local-realism features (angles in 3D); hand insertion; local melt-quench (LAMMPS) | silica potential in PyTorch; coordination / NBO / ring detectors; NequIP conditioning (σ, pins, labels, cooling rate); S(q); VDOS; NEB pipeline; validation with a machine-learned potential |

## 11. Paper outline

1. **Figure 1.** The task: a well-annealed glass, a requested site, and Task B regenerating
   the sphere. Generated vs natural vs hand-built 3-ring and Si(V).
2. **Figure 2.** Success and local realism per class against all baselines, with seed spread
   and floors.
3. **Figure 3.** Physical validation: energies under the independent potential, the
   ring-breathing VDOS, and DFT gap states (if done).
4. **Figure 4.** The application (barrier distributions) and cost per success vs MD at
   1 K/ps.
5. **Supplementary.** 2D Kob–Andersen results as the controlled method study (symmetry
   ladder, step count as cooling rate, sensitivity, seeds), and the Stage 0 reproduction of
   DM2.

## 12. Decisions for the team

1. **Potential:** stay with DM2's (comparable with DM2), or train on a machine-learned
   potential from the start (more physical, slower MD)? *Recommendation: DM2's potential for
   data, a machine-learned potential for validation.*
2. **Cell size:** 576 atoms (cheap) or 1,000–3,000 (closer to DM2)? *Recommendation: 576,
   plus a size-transfer test.*
3. **Application:** barriers (A) or pore surfaces (C)?
4. **Compute:** who applies for the ACCESS allocation in week 1?
5. **Collaboration:** contact the DM2 authors (Schwalbe-Koda group, UCLA)? Their 3D tooling
   and silica data would shorten M0–M2, and a joint paper would be stronger.
