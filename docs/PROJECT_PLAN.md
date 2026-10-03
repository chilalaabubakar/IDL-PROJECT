# Project plan: Controlled Defect Generation in Amorphous Materials via Conditional Diffusion

Team: S. Cheruto, N. Ladislaus, A. Chilala, G. Uwera, M. Sangwa (CMU LTI)
Base paper: Yang & Schwalbe-Koda, *A generative diffusion model for amorphous materials*,
npj Comput. Mater. 12, 29 (2026). Code: [`digital-synthesis-lab/DM2`](https://github.com/digital-synthesis-lab/DM2)
Dataset spec: [`DATASET.md`](DATASET.md) · Implementation plan: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) · Feasibility pilot: [`pilot/`](pilot/)

---

## 0. Summary

| # | Finding or decision | Consequence |
|---|---|---|
| 1 | **No public dataset fits the proposal; we generate it.** The pilot measured 2.9 s per N = 256 glass on one CPU core, and no glass crystallized. | All data fits in under 2 CPU-hours (details in `DATASET.md`). Data is not on the critical path. |
| 2 | **DM2 is not a standard DDPM.** Training adds noise x + σε with σ ~ U(0.001, 0.75 Å), the model predicts the displacement σε, and it gets **no time or noise-level input**. Sampling is about 3,000 steps of "score dynamics" starting from uniform random positions. | Our method section must state which formulation we use. Recommendation: reproduce DM2's formulation, and give the main model a σ embedding (§3.1). RePaint and CFG need a well-defined noise level at each step. |
| 3 | **The EGNN in DM2/graphite is not periodic-safe.** It uses raw `x[i] - x[j]` rather than minimum-image vectors. | We write an EGNN that uses minimum-image (MIC) edge vectors. This was the proposal's top risk, and the fix is small (§3.2). |
| 4 | **Pilot defect statistics change the definitions.** Voronoi coordination tails (CN 4 or 8) cover only 0.5% of atoms, which is too rare. Cutoff coordination of A atoms (≤ 5 or ≥ 8) gives 1.4% of atoms, and B atoms with ≥ 2 B neighbours give 3.9%. | Use the definitions in `DATASET.md` §1.5; thresholds are frozen from the train split. |
| 5 | **"At a chosen location" is trivial in an empty periodic box.** Any naturally formed defect can be moved to any target point by translating the whole box, and in 2D MD is cheap. | **Recommended addition: Task B, host-conditioned inpainting** (§2), plus an honest MD baseline (local melt-quench). This keeps RQ1 and RQ3 meaningful. **This needs a team decision.** |
| 6 | A bug in DM2's `denoise_train_conditional.py`: `in_8` is listed twice and `in_18` is missing, so a 10 K/ps sample is labelled 0.1 K/ps. | Fix it if we reproduce DM2's conditional training. Not needed for our 2D work. |

---

## 1. Goals, research questions and hypotheses

**Goal:** a diffusion model that, when asked for defect type *k* at location *p* in a 2D
binary glass, generates realistic surroundings in which that defect is still present
**after physical relaxation**.

| RQ | Question | Primary metric | Hypotheses tested |
|---|---|---|---|
| RQ1 | Can the model reliably place a requested defect at a requested location? | Success rate after relaxation | H2 (trained beats the 1–5% chance rate) |
| RQ2 | Are the surroundings as realistic as those of naturally formed defects? | Local W1 distances against natural defects | H1 (RePaint is more realistic than clamping), H3 (unconditional is the most realistic) |
| RQ3 | Is it cheaper per success than random sampling or hand insertion? | Wall-clock seconds per success | H2 |

---

## 2. Problem setup: two tasks (decision needed)

**Task A: full-box generation (as in the proposal).** Generate a whole 256-atom box from
noise. One atom is pinned at *p* and carries the label *k*.
*Caveat:* the box is periodic and contains nothing else, so the model is translation-
invariant. A single pinned atom therefore carries **only the label**, and its position is
arbitrary. "Melt-quench by MD, then translate the box so a natural defect sits at *p*" is a
near-free baseline: one glass costs 2.9 s and contains about 3.6 D1 sites and 10 D2 sites.
We should still run Task A, but we can't claim location control from it.

**Task B (recommended main evaluation): host-conditioned local inpainting.** Take a
test-split glass as the host. Pin every atom farther than R_out = 4σ from *p*, delete and
regenerate the inner disk, and require defect *k* at its centre. This is the real use case:
editing an existing structure, as in the crystal-inpainting work of Zhong et al. (2025).
* Location becomes meaningful, because the host fixes the frame.
* Hand insertion becomes exactly "current practice".
* The honest physics baseline becomes **local melt-quench**: freeze the host, heat the
  inner disk, quench, minimize, and repeat until success.
* **No new machinery is needed.** The pin mask simply covers the host atoms, and every
  conditioning method in Table 1 applies unchanged.

Also consider (stretch): **two defects at a chosen separation**, which translation cannot
provide for free.

**Recommendation:** Task B is the headline for RQ1 and RQ3. Task A is reported for
continuity with the proposal and for RQ2.

---

## 3. Method design

### 3.1 Noise process and sampler (what we take from DM2, and what we change)

| Aspect | DM2 (code) | Proposal text | Our plan |
|---|---|---|---|
| Forward noise | x_σ = x₀ + σ·ε, with σ ~ U(0.001, 0.75 Å) **per structure** (variance-exploding style); no wrapping | DDPM | **Same variance-exploding form, in LJ units.** Starting point σ_max ≈ 0.5 σ_AA (DM2 used ≈ 50% of a bond length). Wrap only for graph building, using MIC. |
| Target / loss | MSE(model(x_σ), σε), i.e. predict the displacement | ε-MSE over unpinned atoms U | Displacement MSE over U, normalized per graph. This equals a σ²-weighted ε-MSE; state it that way in the report. |
| Noise-level input | **None** (the network has to infer σ from local disorder) | time embedding | **Main model: Gaussian-Fourier σ embedding added to node features.** Ablation: σ-blind, as in DM2. |
| Sampler | Start from uniform random positions. 2,900 steps of x ← x − D(x) − η, where η ~ N(0, s²), s ~ U(σ_min, σ_k) per coordinate, and σ_k falls linearly from 1.0 to 0.001. Then 100 noise-free steps. | DDPM ancestral | Reuse DM2's sampler (a loop of about 20 lines). Ablation over steps: 200 / 1,000 / 3,000. |
| Graph | ASE neighbour list on the CPU at every step (slow) | — | GPU brute-force MIC graph. N = 256 means 65k pairs, which is trivial, and it batches 256 samples at once. |
| Global condition | cooling rate → Gaussian basis → added at every layer | — | Reuse the same pattern for **per-node** labels. |

### 3.2 Denoisers (the symmetry ladder from Table 1)

All denoisers output a 2D displacement per atom. Inputs: atom type, σ embedding, pin flag
and label token.

| Model | Symmetry | Implementation notes | Role |
|---|---|---|---|
| MLP on flattened coordinates | none (fixed N and atom order) | Input sin/cos of fractional coordinates (keeps periodicity); atoms ordered by type, then by index. **N = 64 only.** | Baseline |
| MPNN (Gilmer et al.) | permutation | Edge features = raw MIC vector (dx, dy) plus distance RBF. No rotation augmentation; a "with augmentation" run is a control. | Ablation |
| **EGNN-PBC** (Satorras et al.) | permutation + E(2) locally | Messages use \|r_ij\|² from **MIC edge vectors**. Output Δx_i = Σ_j r_ij · φ_x(m_ij) (translation-invariant, rotation-equivariant). 4–6 layers, hidden size 128, graph cutoff 2.5–3 σ. | **Main backbone** |
| NequIP (DM2 code, e3nn 0.4.4) | permutation + O(3) | Embed 2D as 3D with z = 0 and drop the output's z component. | Stretch |

Note for the write-up: a square periodic box keeps only the 90° rotations globally.
Rotation equivariance is a **local** prior here.

**Fairness rule for the ladder:** the MLP exists only at N = 64. Train MPNN and EGNN at
N = 64 **as well as** at N = 256. Ladder conclusions come from the N = 64 comparison;
N = 256 shows how the result scales.

### 3.3 Conditioning methods (Table 1)

| Method | Training needed? | How it works in our sampler |
|---|---|---|
| Unconditional + filter (B1) | unconditional model | Generate, relax, keep the samples that have the defect at *p*. Gives the chance rate. |
| Hand insertion + relax (B2) | none | Delete the atoms within r_patch of *p* in a host, paste a library patch at a random rotation, relax with FIRE. |
| Patch in starting noise (B3) | unconditional | DM2 "pore" style: the initial configuration contains the patch atoms, everything else is random, and **nothing is pinned** during sampling. |
| Naive clamping (B4) | unconditional | Reset the patch (or host) atoms to their target positions at every step. |
| **RePaint** (B5) | unconditional (σ-aware preferred) | At step k, set known atoms to x₀ + σ_k·z. Resampling: with U resamples and jump length j, re-noise the unknown atoms by √(σ²_{k−j} − σ²_k). Sweep U ∈ {1, 5, 10} and j ∈ {1, 10}. |
| Local melt-quench MD (B6, Task B) | none | LAMMPS: freeze the host, heat the inner disk (T = 2.0), quench at 1e-2/τ, minimize, repeat until success. |
| **Pin mask + label + CFG (main)** | trained | See below. |
| Classifier guidance (stretch) | separate noise-aware GNN classifier | Add s·σ_k²·∇ₓ log p_φ(y_c = k \| x, σ_k) to the update. |

**Training for the main method** (DM2-style objective plus conditioning):
1. Pick a request mode per structure: *centre only* (p = 0.5), *centre + first shell*
   (p = 0.3), or *host annulus*, i.e. pin all atoms farther than R_out (p = 0.2, Task B).
2. Pick the centre atom: with probability 0.5 a uniformly chosen defect class and then a
   random site of that class; otherwise a uniform random atom carrying its true label.
3. Pinned atoms get no noise, pin flag = 1, and are excluded from the loss. The centre
   gets a label token from {none, D1−, D1+, D2}; every other atom gets an "unlabelled"
   token.
4. **CFG dropout:** with probability 0.15, replace the label with a null token while
   **keeping the pinned geometry**. Guidance then amplifies the label, not the context.
5. Sampling: d̃ = (1 + w)·d(x, σ, c) − w·d(x, σ, ∅). Sweep w ∈ {0, 0.5, 1, 2, 4}.

### 3.4 Training defaults (starting points, to be tuned on val)

AdamW, lr 2e-4 (as in DM2), batch of 16 graphs, 30k–50k updates, EMA 0.999, gradient
clipping at 1.0. Draw a new σ for every structure at every step (DM2 instead duplicated
each structure 128× per epoch; the two are equivalent). Use 3 seeds for the main model if
the GPU budget allows.

---

## 4. Evaluation protocol

All metrics are computed **after FIRE relaxation with every atom free**: no pins, the same
force tolerance as the dataset, and our PyTorch KA potential validated against LAMMPS.
Each (method × defect × task × architecture) cell gets **256 samples**. 95% CIs come from
**bootstrapping over samples** (10,000 resamples), never over atoms: atoms in one sample
are correlated.

| # | Metric | Definition |
|---|---|---|
| 1 | **Success rate** | c* = the atom of the requested species nearest *p*. Success means \|x_c* − p\| < 0.5 σ and detector(c*) = k **after relaxation**. Also report success *before* relaxation; the gap measures how often defects vanish. |
| 2 | **Local realism** | Atoms within R_loc = 3σ of *p* versus the same window around **natural** defects of class k in the test split. Report Wasserstein-1 on per-atom PE, pair distances (AA/AB/BB), bond angles, and relaxation displacement \|Δx\|. Also report tail metrics: the fraction of atoms above the reference 99th-percentile PE, and the minimum pair distance (overlap check). |
| 3 | **Global realism** | g_αβ(r), PE/atom, coordination histograms, global \|Ψ6\| (crystallization check). Stretch: QUESTS entropy/overlap, as in DM2. |
| 4 | **Diversity & memorization** | Mean pairwise distance between local-environment descriptors versus the natural spread, and the distance from each sample to its nearest training patch versus the test-to-train distance. |
| 5 | **Cost per success** | (GPU-s for sampling + s for relaxation) / number of successes, on stated hardware. MD baselines are measured the same way. |

**Calibration rows in every results table:**
* *Floor*: natural (train) against natural (test). This is the best possible W1.
* *Ceiling*: hand insertion **without** relaxation.

Without these two rows, the W1 values can't be interpreted.

**Framing RQ3 honestly:** MD is cheap for a 256-atom 2D glass (3 s). We will report
measured costs for all methods. The claim we can test is cost **per success under
constraints** (Task B, rare defects, multiple defects), not the absolute speed-ups of
1000× that DM2 reports for 3D silica.

---

## 5. Experiment stages and exit criteria

| Stage | Contents | Exit criterion |
|---|---|---|
| **0. Reproduction** | (a) Run the course's proof-of-concept notebook (MLP, 3×3 jittered grid; **please add it to the repo**). (b) Run DM2's pretrained `gen-a-sio2-uncond-v1.pt` on the 300-atom demo on a GPU and compare g(r) to the training data. Note: `torch.load(..., weights_only=False)` with torch ≥ 2.6, `e3nn==0.4.4`, and `graphite` on the path. | Notebook figure reproduced; DM2 sample g(r) matches |
| **1. MVP (week 4)** | Dataset and quality gates; frozen detectors; patch library; PyTorch KA energy + FIRE (tested against LAMMPS); evaluation pipeline that produces floor and ceiling numbers; unconditional EGNN-PBC with sampler. | Unconditional samples after relaxation: no crystallization, < 1% overlapping atoms, global g(r) W1 within 2× the floor; B1 chance rates measured |
| **2. Training-free placement** | B2, B3, B4, RePaint (+ B6 for Task B), on Tasks A and B, for D1± and D2. RePaint U/j sweep. | Results with CIs for H1 and H3 → **midterm report** |
| **3. Trained conditioning** | Pin mask + label + CFG; w sweep; request-mode ablation. | Results for H2; failure catalogue |
| **4. Symmetry ladder** | MLP → MPNN → EGNN at N = 64 (and MPNN/EGNN at N = 256); σ-aware against σ-blind; 200 / 1,000 / 3,000 steps. | Ladder table with CIs |
| Stretch | Classifier guidance; NequIP-2D; D3 voids; two-defect requests; size transfer to N = 1024; GlassBench OOD check. | — |

**Failure analysis (all stages):** vanishing defects (the success gap before and after
relaxation), bad atoms at the patch boundary (PE as a function of distance from *p*),
overlapping atoms, ignored labels (low w), mode collapse (high w; diversity metric). Show
3–5 rendered examples per failure type.

**Sensitivity, in priority order:** RePaint U/j → guidance w → number of steps → defect
thresholds ±0.05σ → R_out / R_loc.

---

## 6. Timeline (9 weeks) and workstreams

| Week | Milestone |
|---|---|
| 1 | Repo scaffold, environment, CI tests. Stage 0. Generator script, plus the full `ka2d_256` and `ka2d_64` sets with quality gates. |
| 2 | Detectors and frozen thresholds; patch library; PyTorch KA + FIRE validated; evaluation pipeline on natural data (floor and ceiling); EGNN-PBC with equivariance tests. |
| 3 | Unconditional EGNN training and sampler; global realism; B1 chance rates; B2 hand insertion. |
| **4** | **MVP checkpoint** (Stage 1 exit criterion). |
| 5 | B3, B4, RePaint, B6; RePaint sweep. |
| **6** | **Midterm:** H1 and H3 with CIs. |
| 7 | Trained pin mask + CFG; w sweep. |
| 8 | Symmetry ladder; step-count and σ ablations. |
| 9 | Failure analysis, stretch goals, final report and figures. |

**Five workstreams (one owner each; the team assigns names):**
1. **Data & physics:** LAMMPS generator, quality gates, PyTorch KA potential + FIRE, B6 local melt-quench.
2. **Defects & evaluation:** detectors, thresholds, patch library, all metrics, bootstrap, floor and ceiling rows.
3. **Models & training:** EGNN-PBC, MPNN, MLP, training loop, EMA, checkpoints, experiment configs.
4. **Sampling & conditioning:** DM2 sampler, clamping, RePaint, CFG, classifier guidance.
5. **Baselines, experiment tracking & writing:** B1–B4 runs, results database, figures, report.

Integration points: the data format (end of week 1), the `denoiser(x, σ, cond) → Δx`
interface (week 2), and the `evaluate(samples) → metrics.json` interface (week 2).

---

## 7. Repository layout

The scaffold is in place: see the top-level `README.md` for the layout (package
`src/glassdiff/`), and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for conventions,
tickets and the module each ticket owns.

**Dependencies:** python 3.10–3.11, torch 2.x, numpy, scipy, lammps + mpich (data
generation only), matplotlib, pyyaml, pytest; optionally wandb.
Keep `torch_geometric` and `e3nn` out unless we do the NequIP stretch goal. Brute-force
graphs at N ≤ 1024 make them unnecessary.

**Required tests before any training result counts:**
* PyTorch KA energy and forces match LAMMPS per atom to 1e-5.
* EGNN-PBC output rotates with the input (arbitrary rotation on a local, non-periodic
  patch; 90° on the box), is unchanged when all atoms are translated by any vector
  (including wrap-around), and is permutation-equivariant.
* Detectors give the expected labels on hand-built configurations.

---

## 8. Compute budget

| Item | Estimate | Basis |
|---|---|---|
| All datasets | < 2 CPU-hours | **Measured** (pilot) |
| One EGNN training run at N = 256 | ~1–4 GPU-hours | Estimate: 800 graphs × 256 atoms, 30k–50k updates (DM2 needed 20–40 h for 3D SiO2 with NequIP at 3,000 atoms) |
| All training runs (~15) | ~30–60 GPU-hours | Estimate |
| Sampling: 256 samples × ~40 cells × ≤ 3,000 steps | a few GPU-hours (batched) | Estimate |
| Relaxation of ~10k samples | < 1 GPU-hour (batched FIRE) or about 1 CPU-hour in LAMMPS | Estimate |

A single modern GPU (cluster, Colab or PSC) is enough. This cloud container has **no GPU**
(4 cores, 15 GB), so only data generation and CPU tests can run here.

---

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| EGNN with a periodic box | Medium → low (cause found) | MIC edge vectors (§3.2) plus the PBC unit tests. Fallback: MLP at N = 64, as the proposal says. |
| A σ-blind denoiser breaks RePaint/CFG | Medium | σ-aware main model; σ-blind kept as an ablation. |
| Defect thresholds too rare or too common | Resolved by the pilot | Definitions in `DATASET.md` §1.5, frozen on train; ±0.05σ sensitivity. |
| 2D glass crystallizes | Low (pilot \|Ψ6\| ≤ 0.15) | \|Ψ6\| gate on every glass and every generated sample. |
| Location claim is trivial (Task A) | High if left unaddressed | Task B as the headline (§2). |
| RQ3 cost claim fails because MD is cheap in 2D | High | Report measured costs honestly, and frame the claim around constrained requests (§4). |
| Finite-size confound in the ladder | Medium | Ladder at N = 64 for every architecture. |
| Leakage between patches and references | Medium | Patches from train only; references and hosts from test only. |
| Schedule | Medium | Hard MVP gate in week 4; stretch goals only after Stage 3. |

---

## 10. Open questions for the team

1. **Adopt Task B (host-conditioned inpainting) as the headline evaluation?** (Recommended.)
2. Training targets: inherent structures (T = 0, recommended) or low-T thermal snapshots?
3. Main denoiser σ-aware (recommended) or σ-blind as in DM2?
4. Where is the Stage 0 proof-of-concept notebook? Please commit it to `notebooks/`.
5. What GPU access do we have (cluster, Colab, PSC)? This sets how many seeds we run.
6. One cooling rate (1e-2/τ) only, or also add cooling rate as a global condition, as in DM2?

---

## References and links

* Yang & Schwalbe-Koda (2026), npj Comput. Mater. 12, 29. Code: https://github.com/digital-synthesis-lab/DM2 · data: https://github.com/digital-synthesis-lab/2025-dm2-data
* Hsu et al. (2024), score-based denoising for atomic structures: https://github.com/LLNL/graphite
* Kob & Andersen (1995), PRE 51, 4626 · Brüning et al. (2009), arXiv:0811.2995 (KA 65:35 in 2D)
* Ho et al. (2020), DDPM · Ho & Salimans (2022), CFG · Lugmayr et al. (2022), RePaint · Dhariwal & Nichol (2021), classifier guidance
* Satorras et al. (2021), EGNN · Gilmer et al. (2017), MPNN · Batzner et al. (2022), NequIP
* Zhong et al. (2025), host-guided inpainting for crystals · Igashov et al. (2024), DiffLinker
* Jung et al., Roadmap on machine learning glassy dynamics; GlassBench: https://zenodo.org/records/10118191
* Related work to add to the proposal: Kilgour et al. (2020), *Generating multiscale amorphous molecular structures using deep learning: a study in 2D*, J. Phys. Chem. Lett. 11, 8532. This is prior work on generative models for 2D amorphous structures (cited as ref. 13 in DM2).
