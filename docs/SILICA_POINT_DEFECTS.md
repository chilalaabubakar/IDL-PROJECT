# Point defects in amorphous silica

The 2D point-defect study (`docs/POINT_DEFECTS.md`) uses a model glass in which vacancies and
interstitials are defined by construction but are not the glass's native defect concept. In
amorphous SiO₂ the standard point defects are canonical: oxygen vacancies (the precursor of
the E′ centre), non-bridging oxygens, peroxy linkages, and 3- or 5-fold Si (reviews: L. Skuja,
*J. Non-Cryst. Solids* 239, 16 (1998); G. Pacchioni, L. Skuja, D. L. Griscom (eds.),
*Defects in SiO₂ and Related Dielectrics*, Kluwer (2000)). This is the same study in that
material.

## Stage 1: how far the standard tools go

`scripts/silica/point_defects_study.py`, on Kaggle with
[`notebooks/kaggle_silica_point_defects.ipynb`](../notebooks/kaggle_silica_point_defects.ipynb)
(CPU only, about 11-12 h on Kaggle's 4 cores, so it may need a second session).
[`notebooks/colab_silica_point_defects.ipynb`](../notebooks/colab_silica_point_defects.ipynb)
continues the same run on Colab. Every step is saved, including each finished repair
minimisation, so a stopped run continues where it stopped.

**Glasses.** DM2's four test glasses (`data/sio2_dm2/test.npz`: 3,000 atoms, cooled at 0.1, 1,
10 and 100 K/ps; made with the Bouhadja-Jakse-Pasturel potential). They are 300 K snapshots,
so each is first energy-minimised, which gives the defect-free originals. They already contain
natural defects (non-bridging O, 3-fold O, a few 3-fold Si; more in faster-cooled glasses),
and an introduced defect has to be found among them.

**Defects** (`glassdiff.silica.point_defects`). Each is made by a known operation on an
original and then relaxed, as in 2D:

| Defect | Operation |
|---|---|
| Oxygen vacancy | remove a bridging O; its two Si become 3-fold. First-principles calculations give a Si-Si bond (the E′ precursor); this potential cannot form one (see the results) |
| Oxygen interstitial | add an O at the largest gap within 3 Å of a random point |
| Oxygen Frenkel pair | move a bridging O to the largest gap 4-6 Å away (vacancy + interstitial, the radiation-damage pair) |
| Silicon vacancy | remove a 4-fold Si; its four O become non-bridging |

**Relaxation.** LAMMPS FIRE at fixed box, all atoms free, until the largest force is below
10⁻³ kcal/mol/Å (`glassdiff.silica.md.minimize_structure`). That takes about a minute per
3,000-atom glass on one core. Two things were checked on a DM2 glass:

- Frenkel pairs closer than about 4 Å recombine when relaxed (3 of 4 at 2-4 Å, none of 4 at
  4-6 Å), hence the 4-6 Å hop.
- Putting an atom back exactly on a site that its neighbours have partly filled can make the
  Si-O attraction run away (the known collapse of this potential form at short range). So
  repairs always place atoms in the largest gap near the target, and a minimisation that
  still fails counts as not restored and is reported.

**Finding.** The top-scored atom must lie within 2.5 Å of the defect (either site of a Frenkel
pair). For an extra or moved O, only O atoms are scored. Scores:

| Score | Needs the potential? |
|---|---|
| Coordination: the standard analysis (Si-O bonds < 2.2 Å, peroxy O-O < 1.7 Å); each atom's deviation from ideal bonding, summed over its 3 Å cluster | no |
| Coordination, rare states first: the same, with each off-ideal atom weighted by how rare its bonding state is in the defect-free originals | no |
| Energy: per-atom energy, z-score within species | yes |
| Chance: the fraction of atoms within 2.5 Å of the defect | |

**Repair.** Put the atom count right at the flagged atom and minimise:

- Vacancy: insert the missing atom in the largest gap within 2.2 Å of the flagged atom.
- Interstitial: remove the flagged O.
- Frenkel pair: move the flagged O to the gap 3-7 Å away around which Si lack the most O
  neighbours.

Restored means the original glass within 0.1 Å (identical atoms may trade places). "Defect
given" does the same at the true site, which is the ceiling a perfect finder would reach.

**What stage 1 decides.** If the coordination analysis already finds the defects, a learned
silica finder adds little and the result is the physics. If it does not, because natural
defects compete or relaxed defects hide, stage 2 trains a 3D finder (the conditional NequIP
of `glassdiff.silica.model` with a per-atom head, as `EGNNAtomClassifier` in 2D) on defects
made in the 16 training glasses, and compares it with these baselines.

## Stage 1 results

Full write-up: [`results/silica_point_defects_20261110-1342Z/README.md`](../results/silica_point_defects_20261110-1342Z/README.md).
80 defects per kind (67 Frenkel pairs; 13 recombined when relaxed); no minimisation failed.

| Defect | Found: coordination / rare states first / energy | Restored: coordination / rare states first / energy | Restored: defect given |
|---|---|---|---|
| Oxygen vacancy | 12.5 / 29 / 29% | 10 / 20 / 20% | **84%** |
| Oxygen interstitial | 17.5 / 14 / 65% | 52.5 / 35 / 65% | **96%** |
| Oxygen Frenkel pair | 15 / 12 / 34% | 4.5 / 1.5 / 7.5% | **76%** |
| Silicon vacancy | 14 / 11 / 79% | 14 / 6 / 52.5% | **76%** |

- **Finding is the bottleneck.** Given the site, minimisation restores 76-96% of glasses.
  From the coordination analysis (no potential), it restores 1.5-52.5%.
- **Why the coordination analysis misses.** The relaxed defects look like the natural defects
  the glasses already have (4 off-ideal atoms in the slowest-cooled glass, 46 in the fastest),
  and its cluster ranking often flags an ideal atom near the defect.
- **The energy** (needs the potential) is strong for silicon vacancies and oxygen
  interstitials, but restores at most 20% of oxygen vacancies and Frenkel pairs.
- **Oxygen vacancies do not form the Si-Si bond** of first-principles calculations: the two Si
  push apart (3.2 → 4.1 Å on average) or re-bond to neighbouring O. The potential's fixed
  charges only let them repel.
- **Hit radius.** 2.5 Å is just below the 2.6 Å O-O edge of a SiO₄ tetrahedron. At 3.0 Å the
  coordination analysis finds 57.5% of interstitials (its flag is usually an O on the same Si,
  and removing that O also restores the glass) and the energy finds 94% of silicon vacancies.

**Decision: stage 2 is justified.** The coordination analysis leaves 44-72 points between what
it restores and the ceiling. Stage 2 is not started. It needs DM2's silica package and GPU
training, and it is set up as follows, decided before any stage-2 result:

- **Training data.** Defects made in DM2's 16 training glasses.
- **Test.** These same 320 test defects (`defects_<kind>.npz`), so the stage-1 numbers are its
  baselines.
- **Measures.** Restored is the main measure. Found is counted at 3.0 Å (the defect atom or an
  O on the same Si), with 2.5 Å reported alongside.
- **Comparisons.** The coordination analysis (no potential, the baseline to beat) and the
  energy (needs the potential).

**Caveats.**

- Four test glasses, one per cooling rate, so the 20 defects per glass and kind share a glass
  and glass-to-glass differences are large.
- One classical potential: no Si-Si bond for the oxygen vacancy, and no charged defects such as
  the E′ centre itself.
- Frenkel pairs are oxygen pairs only.
- DM2's own pretrained silica denoiser is not yet included as a baseline. It needs DM2's
  package (`pip install -e ".[silica]"` with DM2's repository), and that was not tested here.

```bash
pip install -e ".[md]"      # LAMMPS (put MPICH's lib directory on LD_LIBRARY_PATH)
python scripts/silica/point_defects_study.py --config configs/silica/point_defects.yaml \
    out_dir=results/silica_point_defects_<date>
python scripts/silica/analyse_point_defects.py results/silica_point_defects_<date>   # analysis.json
```
