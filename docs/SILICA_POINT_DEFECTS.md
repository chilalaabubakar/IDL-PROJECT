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
(CPU only, about 5-7 h).

**Glasses.** DM2's four test glasses (`data/sio2_dm2/test.npz`: 3,000 atoms, cooled at 0.1, 1,
10 and 100 K/ps; made with the Bouhadja-Jakse-Pasturel potential). They are 300 K snapshots,
so each is first energy-minimised, which gives the defect-free originals. They already contain
natural defects (non-bridging O, 3-fold O, a few 3-fold Si; more in faster-cooled glasses),
and an introduced defect has to be found among them.

**Defects** (`glassdiff.silica.point_defects`). Each is made by a known operation on an
original and then relaxed, as in 2D:

| Defect | Operation |
|---|---|
| Oxygen vacancy | remove a bridging O; its two Si become 3-fold and may form a Si-Si bond |
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

**Caveats.**

- Four test glasses, one per cooling rate, so the 20 defects per glass and kind share a glass.
- Frenkel pairs are oxygen pairs only.
- DM2's own pretrained silica denoiser is not yet included as a baseline. It needs DM2's
  package (`pip install -e ".[silica]"` with DM2's repository), and that was not tested here.

```bash
pip install -e ".[md]"      # LAMMPS (put MPICH's lib directory on LD_LIBRARY_PATH)
python scripts/silica/point_defects_study.py --config configs/silica/point_defects.yaml \
    out_dir=results/silica_point_defects_<date>
```
