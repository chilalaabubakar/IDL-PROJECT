# Point defects in amorphous silica, stage 1: the standard tools miss most defects; repair works once the defect is found

Stage 1 of the silica study (`docs/SILICA_POINT_DEFECTS.md`). Four standard point defects are
made in DM2's four test glasses and relaxed. They are then found and repaired with the field's
standard tools, without any learned model.

**Main result.** Given the defect site, putting the atom count right and minimising restores
the original glass 76-96% of the time. The standard coordination analysis, which needs no
potential, finds only 11-29% of the defects and restores 1.5-52.5% of the glasses. The
potential's per-atom energy does much better for silicon vacancies and oxygen interstitials;
for oxygen vacancies and Frenkel pairs it too restores at most 20%. The headroom between these
tools and the ceiling is what a learned silica finder (stage 2) would have to close.

## The run

- **Code.** `scripts/silica/point_defects_study.py` with `configs/silica/point_defects.yaml`.
- **Where it ran.** On Kaggle (`notebooks/kaggle_silica_point_defects.ipynb`, 4 CPU cores,
  10-11 November 2026): the originals, all four defect sets and three of the four evaluations.
  The silicon-vacancy evaluation ran on Colab (`notebooks/colab_silica_point_defects.ipynb`,
  11 November).
- **Glasses.** DM2's four test glasses: 3,000 atoms, cooled at 0.1, 1, 10 and 100 K/ps,
  energy-minimised to give the defect-free originals.
- **Defects.** 80 per kind (20 per glass), each relaxed with LAMMPS FIRE and the
  Bouhadja-Jakse-Pasturel potential until the largest force is below 10⁻³ kcal/mol/Å. No
  minimisation failed in the whole run.
- **Frenkel pairs.** 13 of the 80 recombined when relaxed (the O went back to its site), so 67
  remain.

Files:

- `originals.npz` and `defects_<kind>.npz`: the relaxed structures, with per-atom energies,
  roles and sites.
- `eval_<kind>.json` and `summary.json`: finding and repair, from the study script.
- `study.log`: the run log.
- `analysis.json`, from `scripts/silica/analyse_point_defects.py` (no minimisation): natural
  defects, finding at three hit radii and per glass, and what each defect became when relaxed.
  It recomputes the study's finding rates and checks that they match.

## The glasses already contain defects

Off-ideal atoms in each defect-free original (Si-O bond < 2.2 Å, ideal: Si 4-fold, O 2-fold):

| Cooling rate | Off-ideal atoms | Of which |
|---|---|---|
| 0.1 K/ps | 4 | 2 non-bridging O, 2 three-fold O |
| 1 K/ps | 10 | 5 non-bridging O, 4 three-fold O, 1 three-fold Si |
| 10 K/ps | 22 | 10 non-bridging O, 11 three-fold O, 1 five-fold Si |
| 100 K/ps | 46 | 23 non-bridging O, 21 three-fold O, 2 three-fold Si |

None has a peroxy (O-O) bond. Faster cooling leaves more natural defects, as expected, and an
introduced defect has to be found among them.

## What the defects became when relaxed

- **Oxygen vacancy (removed bridging O).**
  - Its two Si stay 3-fold in 28 of 80 cases. In 33 one of them, and in 19 both, bond to a
    neighbouring O, which becomes 3-fold.
  - The two Si move apart, from 3.16 Å to 4.12 Å on average; only 4 of 80 end closer than
    2.6 Å.
  - In first-principles calculations the neutral oxygen vacancy instead forms a Si-Si bond of
    about 2.5 Å, the precursor of the E′ centre (K. Rudra, W. B. Fowler, *Phys. Rev. B* 35,
    8223 (1987); S. Mukhopadhyay et al., *Phys. Rev. B* 70, 195203 (2004)). The potential used
    here has fixed charges (Si +2.4) and no Si-Si bond term, so two Si can only repel each
    other.
  - In this study "oxygen vacancy" therefore means a removed bridging O as this potential
    relaxes it, not the E′ precursor.
- **Oxygen interstitial.**
  - The inserted O bonds to one Si in 69 of 80 cases. In 65 of these that Si still has four O
    within 2.2 Å, so one of its original O was pushed out of its bond: the defect relaxes into
    two non-bridging O on one Si.
  - 5 of the inserted O bridge two Si and 6 stay unbonded; none forms a peroxy bond.
  - In 75 of 80 cases the extra O is still the inserted atom, rather than a network O pushed
    elsewhere.
- **Oxygen Frenkel pair.** Of the 67 that did not recombine, the moved O ends as a non-bridging O
  in 57 and as a bridging O in 10.
- **Silicon vacancy.** All four of its O stay non-bridging in 54 of 80 cases, three in 20, and
  fewer in 6.

## Finding and repair

Found: the top-scored atom lies within 2.5 Å of the defect (the radius set before the run).
Restored: after the repair and minimisation, the glass is the original within 0.1 Å (identical
atoms may trade places). Brackets: 95% Wilson intervals. These treat the defects as
independent, but each set of 20 shares a glass, so the true uncertainty is larger.

| Defect | n | Found: coordination | Found: rare states first | Found: energy (needs the potential) | Chance |
|---|---|---|---|---|---|
| Oxygen vacancy | 80 | 12.5% [7, 22] | **29%** [20, 39] | **29%** [20, 39] | 0.1% |
| Oxygen interstitial | 80 | 17.5% [11, 27] | 14% [8, 23] | **65%** [54, 75] | 0.07% |
| Oxygen Frenkel pair | 67 | 15% [8, 25] | 12% [6, 22] | **34%** [24, 46] | 0.2% |
| Silicon vacancy | 80 | 14% [8, 23] | 11% [6, 20] | **79%** [69, 86] | 0.09% |

| Defect | Restored: coordination | Restored: rare states first | Restored: energy | Restored: defect given (ceiling) |
|---|---|---|---|---|
| Oxygen vacancy | 10% [5, 19] | 20% [13, 30] | 20% [13, 30] | **84%** [74, 90] |
| Oxygen interstitial | 52.5% [42, 63] | 35% [25, 46] | 65% [54, 75] | **96%** [90, 99] |
| Oxygen Frenkel pair | 4.5% [2, 12] | 1.5% [0, 8] | 7.5% [3, 16] | **76%** [65, 85] |
| Silicon vacancy | 14% [8, 23] | 6% [3, 14] | 52.5% [42, 63] | **76%** [66, 84] |

**The hit radius.** 2.5 Å is just below the 2.6 Å O-O edge of a SiO₄ tetrahedron, so an O
bonded to the same Si as the defect counts as a miss. For interstitials that is what the
coordination analysis usually flags. Removing that O also restores the glass, because the
inserted O moves into its place: restored is 52.5% although found is 17.5%. Of four such cases
repaired and minimised separately, three were restored. At 3.0 Å the finding rates become:

| Defect | Coordination | Rare states first | Energy |
|---|---|---|---|
| Oxygen vacancy | 15% | 35% | 32.5% |
| Oxygen interstitial | **57.5%** | 40% | 66% |
| Oxygen Frenkel pair | 27% | 24% | 34% |
| Silicon vacancy | 20% | 15% | **94%** |

Restored needs no radius and is the measure to compare.

**Per glass** (found at 2.5 Å, slowest → fastest cooling: 0.1, 1, 10, 100 K/ps):

| Defect | Coordination | Rare states first | Energy |
|---|---|---|---|
| Oxygen vacancy | 30, 10, 10, 0% | **80**, 15, 10, 10% | 50, 15, 45, 5% |
| Oxygen interstitial | 20, 20, 10, 20% | 15, 20, 10, 10% | 85, 60, 70, 45% |
| Oxygen Frenkel pair | 37, 0, 12, 6% | 21, 0, 12, 11% | 47, 23, 47, 17% |
| Silicon vacancy | 30, 5, 15, 5% | 20, 5, 15, 5% | 80, 90, 75, 70% |

- The coordination scores work best in the slowest-cooled glass, which has 4 natural defects
  against 10-46 in the others. The clearest case is oxygen vacancies ranked by rare states:
  80% there, against 10-15% elsewhere.
- Natural defects are not the whole story. Even in that glass, the coordination score finds
  only 30% of silicon vacancies, although every one has off-ideal atoms within 3 Å of its
  site. The score sums deviations over a 3 Å cluster, so it often flags an ideal atom among
  the off-ideal ones, 2.5-4 Å away.
- A check made after the results were seen (in `analysis.json`): allowing only off-ideal
  atoms to be flagged raises silicon vacancies to 25% and oxygen vacancies, rare states first,
  to 40%. Every coordination-based variant still finds a minority of the defects.

## What this says

1. **Repair is not the problem: finding is.** Given the site, putting the atom count right and
   minimising restores 76-96% of glasses. From the coordination analysis, it restores 1.5-52.5%.
2. **The standard coordination analysis is a weak finder in a real glass.** Without a
   potential, it finds 11-29% of the defects. The relaxed defects look like the natural defects
   the glasses already have: re-bonded Si next to 3-fold O, and non-bridging O. Faster-cooled
   glasses have dozens of these.
3. **The per-atom energy is a strong baseline for two defects, but it needs the potential.** It
   finds 79% of silicon vacancies and 65% of oxygen interstitials, restoring 52.5% and 65%. For
   oxygen vacancies and Frenkel pairs it finds 29% and 34% and restores 20% and 7.5%.
4. **Frenkel pairs fail at the second step.** Even when the moved O is flagged (by the energy,
   34%), choosing which gap 3-7 Å away to put it back in rarely works: 7.5% restored, against
   76% when the site is given. The same was true in 2D.
5. **Stage 2 is justified.** A learned silica finder would have to beat the coordination
   analysis (no potential), where the gap to the ceiling is 44-72 points. It should also be
   compared with the energy, which it can only match without a potential by learning the
   structure.

## Caveats

- **Four glasses**, one per cooling rate, so glass-to-glass variation is large (see per glass)
  and the intervals above are too narrow.
- **One classical potential.** It cannot form the Si-Si bond of the oxygen vacancy (see above),
  and with fixed charges it cannot represent charged defects such as the E′ centre itself.
- **Our implementation of the coordination analysis.** Bond cutoffs are 2.2 Å (Si-O) and
  1.7 Å (O-O), and atoms are ranked by a cluster sum. Other rankings may do somewhat better
  (see the check above).
- **A strict success test.** "Restored" asks for the exact original minimum within 0.1 Å.
- **Hit radius.** 2.5 Å was set before the run; 3.0 Å changes the finding rates for
  interstitials and silicon vacancies (see above).
