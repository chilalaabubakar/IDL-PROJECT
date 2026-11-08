# Prof. Dabo's recommendations and the new study

On 6 November 2026 we discussed the project with Prof. Ismaila Dabo (Materials Science and
Engineering, Carnegie Mellon University). He raised ten points about the midterm report and
the first version of the study. This branch, `dabo-point-defects`, holds the study we are
doing in response. The technical description is in [POINT_DEFECTS.md](POINT_DEFECTS.md).
The first results are in
[results/point_defects_20261107/](../results/point_defects_20261107/README.md).

## The first version in one paragraph

We trained a diffusion model (a periodic EGNN with DM2's training loss) to rebuild a small
region of a 2D Kob-Andersen glass with a requested "defect" at its centre. The defects were
D1− and D1+ (A atoms with at most 5 or at least 8 neighbours) and D2 (a B atom with at least
two B neighbours). On top of that we added held atoms, defect labels, classifier-free
guidance, two tasks, three architectures, several baselines, a generalisation study and a
3D silica plan.

## His points, our answers and what changes

| # | Point raised | Our answer | What changes |
|---|---|---|---|
| 1 | Which models are used at each stage, and why? | Glasses: LAMMPS molecular dynamics (Kob-Andersen potential). Defect labels: hand-written rules. Generator: the EGNN, because its output must respect the periodic box and turn with the glass. MPNN and flat MLP: a symmetry ablation only. Relaxation and checks: FIRE minimisation. 3D: NequIP. For most stages the honest reason was "DM2 did it" or "an ablation". | One model family (the EGNN) and one loss, each with a stated reason. No ablation models in the main story. |
| 2 | The sections are not connected and the text reads as machine-written. | Agreed. Experiments were added one after another and never tied back to one question; terms were explained in plain words next to dense numbers. | The final report is written by the team around one question: problem → method → test → result. |
| 3 | There are methodological and conceptual inconsistencies. | Agreed. Some of them are listed below the table. | The new study uses one dataset, one model setting, one success test and one denominator throughout. |
| 4 | The minimum viable study is already in the training section; the work is overdone. | Agreed. The training section, a network that learns to remove a perturbation from a glass, is the study. | The new study is that training section with a physical perturbation (a defect) in place of random noise. |
| 5 | What do the equations actually model? | See "What the equations model" below. The report never said. | The report explains each equation physically, with evidence. |
| 6 | Try reconstructing the original structure from its defected counterpart. | Adopted: the **repair** task. | Main task 1. |
| 7 | Or learn to identify incorrect structures. | Adopted: the **finding** task. | Main task 2. |
| 8 | The defect types look arbitrary and follow no concept in computational materials science. | Agreed. The thresholds were chosen so each defect covered 1-5% of atoms; a Voronoi definition was dropped because it was "too rare". | Standard point defects: vacancy, interstitial, Frenkel pair, antisite. For silica: the oxygen vacancy (E′ precursor), 3- and 5-fold Si, non-bridging O, 3-fold O. |
| 9 | Why use AI to generate defects when they can be made without it? | Agreed. In our 2D glass, one melt-quench takes 2.9 s and holds about 10 D2 sites, and hand insertion works 30% of the time. | We no longer generate defects. We find and repair them. A learned model can do both without an interatomic potential; the simulation supplies the ground truth to check it. |
| 10 | The method is too complex and still inconsistent. | Agreed. | Removed from the new study: defect labels, guidance, held atoms, Task A, the symmetry ladder, hand insertion and the generalisation sweep. Kept: one model, one loss, and physical baselines (minimisation, per-atom energy). |

### Inconsistencies in the first version (point 3)

- The report said "we keep DM2's training target and sampler", but we changed the noise range,
  added a noise-level input and cut the sampler from 2,900 to 100 steps.
- Different trained models served different defects (noise range 1.0 for D2, 0.5 for D1−).
- The chance baseline used 2,900 sampling steps, the model 100.
- The model rebuilt about 58 atoms (a 4σ region); hand insertion pasted a 1.5σ patch.
- Three guidance settings (w = 2, 6 and per model) fed one headline number ("1.5-2.4 times").
- The realism measure mixed successes and failures in one table and used successes only in
  another.
- Some requested sites were already defects before editing. They were counted in the main
  table and excluded in the generalisation study.
- The motivation said molecular dynamics is slow, but our own data showed a glass takes 2.9 s.

### What the equations model (point 5)

- **Input features.** These are bookkeeping: how atom type, held flag, label and noise level
  become numbers.
- **EGNN layer.** Each output arrow is a sum over neighbours of (neighbour vector × learned
  weight). That is the form of a pair force, F_i = Σ_j r_ij f(r_ij), with weights that depend
  on the local environment. The network is a learned many-body force field.
- **Training loss.** For noisy positions x = x0 + σε, the network d(x, σ) is trained to
  predict the shift x − x0. The best possible answer is d = −σ² ∇ log p_σ(x) (Tweedie's
  formula), so the correction −d points towards more probable glass structures. This is a
  learned restoring force. On our test glasses its direction agrees with the physical
  Kob-Andersen force (cosine 0.81 at σ = 0.05), more closely than the straight line back to
  the original does (0.68).
- **Sampler.** Repeated noisy steps down this learned landscape, with shrinking noise, amount
  to a simulated quench with learned forces. That is why the number of steps behaved like a
  cooling rate.
- **Classifier-free guidance.** It samples from p(x) p(label | x)^(1+w), tilting towards
  structures a hidden classifier finds more defect-like. Larger w gave more successes but
  less natural surroundings, the trade-off seen between w = 2 and w = 6.

## The new study

**Question.** Given a relaxed glass that contains a standard point defect, can a denoising
diffusion model **find** the defect and **repair** it, that is, give back the defect-free
glass it came from?

**Defects.** Each defect is made by a known operation on a known glass, so the original is
the exact target and no detection thresholds are needed:

- vacancy (remove an atom);
- interstitial (add a small atom in a gap);
- Frenkel pair (move an atom 1.2-3σ into a gap);
- antisite (swap an A atom and its nearest B atom).

The defected glass is then minimised with all atoms free.

**Models.** One EGNN, two uses:

- **Repair:** DM2's loss with a defect, not only noise, to remove.
- **Finding:** the same network with a per-atom head over five roles (intact, next to an
  empty site, extra atom, displaced atom, wrong species).

**Baselines.**

- **Repair:** energy minimisation, with and without a random shake.
- **Finding:** the per-atom energy, which needs the potential.

**First results, before any training** (the existing denoiser, never shown a defect; 200 test
defects each):

| Defect | Found by the denoiser (no potential) | Found by the energy (needs the potential) | Original glass restored |
|---|---|---|---|
| Vacancy | **44%** | 29.5% | **41-43.5%** vs 26% with the energy |
| Interstitial | 12% | **28%** | 6% vs **21%** with the energy |
| Frenkel pair | 19% | 28.5% | **8%** vs 0.5% by shaking alone |
| Antisite | 15% | **29%** | 0% vs 4% with the energy |

The untrained denoiser already finds vacancies better than the energy does. It does not know
how to undo the other defects. That is why both models were then trained on the defects.

**After the first training** (`results/point_defects_trained_20261107/`), the trained finder
locates the defect more often than the potential's energies for three of the four types:

| Defect | Found: trained finder | Found: energy | Restored with the trained models | Restored via energy |
|---|---|---|---|---|
| Vacancy | **66%** | 29.5% | **51-54%** | 26% |
| Interstitial | **46.5%** | 28% | **34.5-35%** | 21% |
| Antisite | **44.5%** | 29% | **27%** | 13.5% |
| Frenkel pair | 26% | 28.5% | 10% (region given) | 0.5% |

Almost all of the gain came from the finder. The repair model learned to remove noise but
not the defect, because the defect atoms were a tiny part of its loss. Version 2 of the
repair model fixes this (`notebooks/kaggle_point_defects_v2.ipynb`).

## Plan

| Step | What | Where | Suggested owner |
|---|---|---|---|
| 1 | Train the repair and finding models; evaluate them against the baseline above. Done: the finder works, the repair model needed a fix | `notebooks/kaggle_point_defects.ipynb` | Abubakar Chilala |
| 1b | Retrain the repair model (version 2) and evaluate it, including Frenkel pairs found and repaired with nothing given | `notebooks/kaggle_point_defects_v2.ipynb` (~5 h on one Kaggle GPU) | Abubakar Chilala |
| 2 | Check the defect operations and the success tests; add error bars over glasses | `scripts/eval_point_defects.py` | Stacy Cheruto |
| 3 | Repair procedure: number of denoising steps and starting noise | `scripts/eval_point_defects.py` | Gentille Uwera |
| 4 | Physics checks: training-data generation, minimiser agreement, stability of repaired glasses in molecular dynamics | `scripts/make_point_defects.py`, `glassdiff.md` | Nicholaus Ladislaus |
| 5 | Baselines and the final report, written by the team around the one question | `paper/` | Michel Sangwa |
| 6 | If time allows: the same study in amorphous silica with its standard defects | `docs/SILICA_3D_PLAN.md` | team |

The owners follow the roles in the midterm report; the team should confirm them.

## Questions to take back to Prof. Dabo

1. Is "the exact original energy minimum" the right measure of repair? In a glass many nearby
   minima are equally natural. The alternative is any defect-free minimum close in energy and
   structure.
2. For the main result, should we stay with the 2D model glass (cheap, exact ground truth) or
   move to amorphous silica (materials relevance, costlier)?
3. In silica, which defects matter most to him: oxygen vacancies and E′ centres,
   non-bridging oxygens, or over-coordinated Si?
4. Is a model that finds and repairs defects without an interatomic potential the right
   answer to "why AI"? Would structures from experiment (for example microscopy of 2D
   glasses) be the setting to aim for?

## Where the first version lives

The first version (placing D1−, D1+ and D2) stays on `main` with its results, the midterm
report (`paper/midterm_report.tex`) and the generalisation study (`docs/GAP2_*.md`). This
branch is for the new study.
