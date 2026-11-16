# Physics recheck of the 2D point defects, and the test numbers of the report

Made while checking the physics for the report on the two-dimensional study
(`paper/point_defects_report.tex`, `paper/point_defects_report.pdf`). Three things came out
of it:

1. **A quarter of the Frenkel pairs are no defects** (they annihilated during relaxation).
   They are now left out of every test.
2. **An antisite is a defect of chemical order, not of volume.** The cell-size baseline had
   this wrong, so its antisite rule was corrected and a second baseline that needs no
   potential, the number of unlike neighbours, was added.
3. **The final test numbers,** on the 750 real defects, with intervals that resample whole
   glasses and paired tests against every baseline.

## Final test results

`summary.md` (written by `paper/make_report_numbers.py` from `chunks/`). Brackets: 95%
intervals from resampling whole test glasses (10,000 bootstrap samples), because the two
defects made in one glass are not independent.

Found (the flagged atom lies within 1.5σ of the defect):

| Defect | n | Learned finder | Cell size | Unlike neighbours | Atom energy | Random atom |
|---|---|---|---|---|---|---|
| Vacancy | 200 | **69.5%** [62, 76] | 47.5% [40, 55] | 29% [22, 36] | 29.5% [23, 36] | 2.9% |
| Interstitial | 200 | **50%** [42, 57] | 15% [10, 20] | 28% [22, 34] | 28% [22, 34] | 3% |
| Antisite | 200 | **49%** [42, 56] | 16.5% [12, 22] | 31.5% [25, 38] | 29% [23, 35] | 4% |
| Frenkel pair | 150 | **45.3%** [38, 53] | 26.7% [20, 33] | 27.3% [21, 34] | 32.7% [25, 40] | 5.4% |

Restored (the original glass after correcting the atoms at the flagged atom and minimising):

| Defect | Learned finder | Cell size | Unlike neighbours | Atom energy | True location given |
|---|---|---|---|---|---|
| Vacancy | **53.5%** [46, 61] | 40.5% [34, 48] | 22% [16, 28] | 26% [20, 32] | 67% [60, 74] |
| Interstitial | **38%** [31, 46] | 8.5% [4, 13] | 21.5% [16, 27] | 21% [16, 26] | 80% [74, 86] |
| Antisite | **30%** [24, 36] | 1.5% [0, 4] | 7.5% [4, 11] | 8% [4, 12] | 68.5% [62, 74] |
| Frenkel pair | **12.7%** [8, 18] | 0% [0, 2] | 1.3% [0, 3] | 7.3% [3, 11] | 42% [34, 50] |

The learned finder and the two structural measures use no potential to flag the atom; the
atom energy does. Every route ends with energy minimisation.

What the numbers say:

- **The finder finds more of every defect type than the measures without learning that we
  tested.** Compared on the same defects (McNemar's exact test, corrected for the twelve
  comparisons of each table with Holm's method), its lead in finding is significant against
  each of the three baselines for each type: p < 0.001 everywhere (cell size, Frenkel pairs:
  0.002), except against the atom energy for Frenkel pairs (p = 0.014 before and after the
  correction). `summary.md` lists both p values.
- **For the finder, restoring follows finding** for vacancies, interstitials and antisites
  (p < 0.001 against each baseline route, after Holm's correction). For Frenkel pairs every route restores few, and the finder's lead
  over the energy route is within the noise (12.7% vs 7.3%, p = 0.17). The finder flags
  exactly the displaced atom, which the repair has to move back, in only 21% of the pairs
  (the atom energy in 7%).
- **Cell size competes only for vacancies,** where missing volume is what one looks for.
- **A count of unlike neighbours finds about what the atom energy finds** (27-32% against
  28-33%), without the potential. Section "What two simple structural measures see" below
  explains why.
- **Finding still limits the result:** with the true location given, minimisation restores
  42-80%. "True location given" means: vacancy, an atom of the removed species at the most
  open point within 1σ of the removed atom's place; interstitial, the added atom removed;
  Frenkel pair, the displaced atom moved to the most open point within 1σ of its original
  place; antisite, the two species swapped back; then minimisation. It is a reference, not a
  strict upper limit.
- **The brackets cover the test sample only.** The finder was trained once; repeated shorter
  trainings vary by less than one point (`../point_defects_seeds_20261114/`).

## Vacancies by the species removed

`defect_occupancy.json` (key `vacancy`) gives the species of every removed atom; the rates
are computed by `paper/make_report_numbers.py`.

| Removed atom | n | Found: finder | Cell size | Unlike neighbours | Atom energy | Restored: finder | Atom energy | True location given |
|---|---|---|---|---|---|---|---|---|
| A (large) | 127 | **69.3%** | 56.7% | 40.9% | 40.2% | **49.6%** | 34.6% | 58.3% |
| B (small) | 73 | **69.9%** | 31.5% | 8.2% | 11% | **60.3%** | 11% | 82.2% |

The finder finds both alike. The measures without learning find far fewer B vacancies: a
missing small atom leaves little trace on any single neighbour. The repair of a vacancy is told
the species of the missing atom (part of "the type of defect is given").

## What the Frenkel pairs are (definition C)

`defect_occupancy.json`, from `scripts/pilots/defect_occupancy.py`. A Frenkel pair is made by an
operation; this checks what the relaxed glass contains. Every atom is assigned to the nearest
atom of the original glass (`glassdiff.data.point_defects.site_occupancy`, a Wigner-Seitz
analysis with the original glass as reference) and the empty and doubly occupied sites are
counted. A Frenkel pair exists only if exactly one site is empty and one is doubly occupied,
more than one neighbour shell (1.4σ_αβ) apart.

Validation on defects whose answer is known (test split; validation split in brackets):

| Defects | Expected | Got it |
|---|---|---|
| 200 vacancies | one empty site | 93% (91.5%) |
| 200 interstitials | one doubly occupied site | 95.5% (94%) |
| 50 Frenkel pairs that undid themselves | no defect | 100% (65 of 65) |
| 200 antisites | no empty or doubly occupied site, species changed | 71% (67.5%); 22% (29%) contain an atom pushed into a neighbouring site |

The 150 real test Frenkel pairs:

| Class | n |
|---|---|
| Frenkel pair (40 of them without any species change) | 66 |
| Close pair (empty and doubly occupied site within one shell) | 27 |
| Antisite-like (every site singly occupied, two of the wrong species) | 35 |
| Complex | 19 |
| No defect | 3 |

On the 66 true pairs: found by the finder 47% [35, 59], atom energy 34.8%, cell size 34.8%,
unlike neighbours 28.8% (finder vs energy p = 0.15); restored by the finder 18.2%, the energy
route 9.1% (p = 0.18), true location given 60.6%. On the other 84: found by the finder 44%,
restored 8.3%. The finder was trained with the old labels (all of them "Frenkel pair").

## A quarter of the Frenkel pairs are no defects

`defect_physics.json`, from `scripts/pilots/point_defect_physics.py`.

**Frenkel pairs that annihilated.** In a Frenkel pair an atom is moved 1.2-3σ into a gap and
the glass is relaxed. In about a quarter of the cases, the moved atom pushed a neighbour of its
own species into the empty site and took that neighbour's place. The relaxed glass is then the
original glass with two identical atoms exchanged: the same structure and the same energy.
There is no defect left to find or repair.

| Split | Frenkel pairs | The original glass again |
|---|---|---|
| Training | 6,377 | 1,735 (27%) |
| Validation | 199 | 65 (33%) |
| Test | 200 | 50 (25%) |

The generator only checked whether the moved atom had returned to *its own* site
(`recombined_tol`), so these passed as valid. Vacancies, interstitials and antisites are not
affected (none is the original again).

Consequences:

- **Test.** Every earlier Frenkel number counted these 50 glasses as defects. Finding had
  nothing to find there, which lowered "found" (37% of 200 before, 45.3% of the 150 real
  pairs now). "Restored" was raised instead, because a glass that is already the original
  easily comes back to it: of the 50, about 17 counted as restored by the finder route and
  26 by the energy route (finder 18% before, 12.7% now; energy route 18.5% before, 7.3% now).
- **Training.** The finder was trained with 27% of its Frenkel examples labelled "displaced
  atom" and "next to an empty site" in a glass that had neither. We expected this to explain
  part of its weakness on Frenkel pairs. A short training run without these examples found
  no more defects, however (`../point_defects_ablations_20261112/README.md`), so the wrong
  labels do not seem to be what holds the finder back.

**The fix.** `glassdiff.data.point_defects.already_original` detects these cases (the relaxed
glass matches the original one to one within 0.05σ, identical atoms allowed to trade places),
and `real_defects` leaves them out. The stored dataset files are unchanged. The option
`--real-defects-only` applies the filter in `scripts/eval_point_defects.py`,
`scripts/eval_point_defect_repair.py`, `scripts/pilots/compare_finders.py` and
`scripts/pilots/baseline_routes.py`; `real_defects_only: true` does it in a training config.

## What the relaxed defects are (test glasses, real defects)

Median, with the middle half of the cases in brackets.

| | Vacancy | Interstitial | Frenkel pair | Antisite |
|---|---|---|---|---|
| Real defects | 200 | 200 | 150 | 200 |
| Other atoms moved by more than 0.1σ | 14 [3, 43] | 21.5 [11.5, 40] | 44 [21, 69] | 28.5 [15, 55] |
| Largest move of another atom (σ) | 0.33 | 0.32 | 0.64 | 0.42 |
| Energy above the original (ε) | | | 3.2 [1.5, 5.0] | 2.8 [1.4, 4.3] |
| Energy higher than the original | | | 91% | 89% |
| Energy lower than the original | | | 8% | 11% |

- **Vacancies partly close.** Right after removal, the most open point at the site is 0.87σ
  from the nearest atom. After relaxation it is 0.74σ [0.66, 0.83]. Around atoms of
  defect-free glasses the same measure is 0.63σ, and 0.70σ at the 95th percentile. So 60.5%
  of relaxed vacancies are still more open than normal glass and 39.5% are not. This agrees
  with early simulations of Lennard-Jones glasses (Bennett et al., *Philos. Mag. A* 40, 485
  (1979); Delaye and Limoge, *J. Phys. I* 3, 2079 (1993)).
- **Interstitials.** The extra B atom sits 0.85σ from its nearest neighbour (B atoms in
  defect-free glass: 0.87σ) and has a higher energy (−3.0 against −3.4).
- **An antisite does not always cost energy.** For 11% of the antisites the swapped and
  relaxed glass is lower in energy than the original. In a stable ordered crystal that cannot
  happen; a
  quickly cooled glass is not in its lowest state, and exchanging two atoms can lead it to a
  lower one (the principle of swap Monte Carlo). The original is still the target for
  repair, but in these cases it is not the more stable structure.

## What two simple structural measures see of a defect

`structural_signatures.json`, from `scripts/pilots/structural_signatures.py`. The measures are
the baselines of the report that need neither a potential nor learning
(`scripts/pilots/baseline_routes.py`): the area of an atom's Voronoi cell (free volume) and
its number of unlike nearest neighbours (chemical order), each compared with the atoms of its
species in defect-free glasses.

**An antisite is a defect of chemical order, not of volume.** After relaxation the cells of
the two exchanged atoms are close to normal, because the neighbours adjust their distances:

| Exchanged atom | Cell area (z-score, median) | Unlike neighbours | Usual for its species | Fewer than usual |
|---|---|---|---|---|
| A atom on a B site | +0.74 (larger than usual in 78%) | 1.76 | 2.37 | 82% |
| B atom on an A site | −0.13 (larger than usual in 45%) | 3.75 | 4.36 | 93% |

A first version of the cell-size baseline flagged "a B atom with too large a cell or an A
atom with too small a cell" for antisites. The table shows that this rule had the physics
backwards (an exchanged atom's cell is set by its neighbours, not by its own size), so the
rule was replaced by "the most unusual cell", and the count of unlike neighbours was added as
a second baseline.

**The atom energy is mostly a count of A-B bonds.** In defect-free glasses the correlation
between an atom's energy and its number of unlike neighbours is −0.90 for A atoms and −0.84
for B atoms: the A-B bond (ε = 1.5) is by far the strongest. A count of bonds, which needs no
potential, therefore finds about as many defects as the atom energy does.

**The rules were chosen on validation glasses,** not on the test. Defects found (the flagged
atom within 1.5σ), validation split, real defects:

| Rule | Vacancy | Interstitial | Frenkel pair | Antisite |
|---|---|---|---|---|
| Cell: largest | **42%** | 2.5% | 30.6% | 12% |
| Cell: smallest | 2.5% | **16%** | 9% | 7.5% |
| Cell: most unusual | 42% | 3.5% | **30.6%** | **12%** |
| Cell: B large or A small | 39.5% | 7.5% | 21.6% | 9.5% |
| Cell: A large or B small | 25.5% | 1.5% | 27.6% | 9.5% |
| Unlike neighbours: fewest | **25.5%** | **29%** | **31.3%** | **31%** |
| Unlike neighbours: most | 3.5% | 8% | 11.2% | 8% |

Bold: the rule each baseline uses.

## Local pressure: a fourth measure, for finding only

The one earlier study that identified inserted point defects in a glass (Laakkonen and
Nieminen, J. Phys. C 21, 3663 (1988): one species, three dimensions) recognised interstitials
by the local pressure on each atom and vacancies by the empty space. Empty space is what the
cell-size baseline measures. Local pressure was missing from our comparison, and interstitials
are where the finder leads most, so it was added as a check
(`structural_signatures.json`, key `local_pressure`).

For every atom, w = 1/2 sum over neighbours of (-r dU/dr): positive where the atom is
squeezed (`glassdiff.physics.ka_potential.ka_virial`; the sum over atoms reproduces the
pressure -dE/dA of the box). It is compared with the atoms of the same species in defect-free
glass (A: 6.5 ± 5.4, B: 2.9 ± 5.4) and one atom is flagged. The rule was chosen on the
validation glasses.

| | Vacancy | Interstitial | Frenkel pair | Antisite |
|---|---|---|---|---|
| Rule chosen on validation | lowest | most unusual | lowest | lowest |
| Found on validation | 8% | 21.5% | 16.4% | 13% |
| **Found on test** | **8.5%** [5, 12] | **19%** [14, 25] | **10.7%** [6, 16] | **16%** [11, 22] |
| Same, pressure with the atom's own cell area | 7% | 16.5% | 7.3% | 15% |
| For comparison: atom energy | 29.5% | 28% | 32.7% | 29% |
| For comparison: learned finder | 69.5% | 50% | 45.3% | 49% |

Local pressure finds fewer defects than the atom energy and the count of unlike neighbours
for every type (finder against local pressure: McNemar p < 0.001 for every type). The relaxed
interstitial has made room for itself, and in this two-species glass the pressure on an atom
of defect-free glass varies as much as its mean. It was not carried through the repair.

## The repair networks without a potential

`repair_networks.json` (`scripts/eval_point_defect_repair.py --no-minimise`): the count and
species are already right at the true site, and the network is applied to the whole glass
with no minimisation. "Defect atoms" is the RMS distance to the original (σ) over the atoms
the defect moved by more than 0.1σ; "overlap" is the share of glasses left with two atoms
closer than 0.56σ, which never occurs in the data.

| Defect | Do nothing | Repair v1, one step | Repair v2, one step | Minimise (needs the potential) | Overlap: v1 / v2, one step | Overlap: v2, 10 steps |
|---|---|---|---|---|---|---|
| Vacancy | 0.234 | 0.216 (−7%) | 0.203 (−13%) | 0.208 | 18% / 20% | 60% |
| Interstitial | 0.189 | 0.174 (−8%) | 0.160 (−16%) | 0.107 | 0% / 1% | 50% |
| Antisite | 0.211 | 0.193 (−9%) | 0.180 (−15%) | 0.149 | 0% / 2% | 60% |
| Frenkel pair (150) | 0.255 | 0.236 (−7%) | 0.216 (−15%) | 0.226 | 16% / 40% | 73% |

Version 2 learned the defect (13-16% of the error removed in one step, version 1: 7-9%), but
it leaves overlapping atoms, and more steps make that worse. The validation numbers of
`../point_defects_v2_20261109/` (12-26%) included the annihilated Frenkel pairs.

## Is a glass defective at all?

`glass_level.json`: without calibration the finder flags 68% of the 100 defect-free test
glasses; with it 2%, but then only 28.7% of the defected ones. AUC 0.75.

## Corrections to earlier write-ups

- **Frenkel pairs:** every earlier Frenkel number included the 50 non-defects (see above).
- **The antisite "energy route" (13.5%)** of `../point_defects_trained_20261107/` and later
  tables used the *finder's* flag and chose the pair by energy. Flagged and chosen by energy
  alone, the route restores 8%.
- **Intervals** were Wilson intervals over defects treated as independent. They now resample
  glasses, which changes them by a point or two.

## Files

- `summary.md`: the result tables and all paired tests.
- `chunks/`: the test evaluation in 64 pieces (one defect type, 50 test examples each), with
  the result for every defect and the glass it came from. `finder_*`: the learned finder and
  the energy routes (`scripts/eval_point_defects.py --calibrate`); `baseline_*`: cell size
  and atom energy, `bonds_*`: unlike neighbours (`scripts/pilots/baseline_routes.py`);
  `ceiling_*`: the true location given (`scripts/eval_point_defect_repair.py`).
- `recheck.log`: the log of the run.
- `defect_physics.json`, `structural_signatures.json`, `repair_networks.json`,
  `glass_level.json`: the measurements above.

To reproduce (about 2 hours on 4 CPU cores; a run that is interrupted continues where it
stopped):

```bash
bash scripts/pilots/recheck_real_defects.sh
python paper/make_report_numbers.py
```
