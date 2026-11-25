# Defects found in thermal snapshots (25 November 2026)

Item 3 of `docs/NPJ_EXTENSIONS.md`, second part. Validation glasses only; the fresh test set
is scored once, at the end.

**Snapshots** (`scripts/pilots/thermal_snapshots.py`, `data/ka2d_256_point_defects_thermal/`,
3.4 minutes on 4 cores). Each of the 734 real validation defects starts from its relaxed
defected glass, gets velocities at temperature T and runs at constant temperature for 10 time
units (Nose-Hoover, LAMMPS; `glassdiff.md.thermal`). Neighbouring atoms move together, unlike
the independent shifts of `results/displaced_inputs_20261124/`. A defect counts as surviving
when the minimised snapshot has the same empty, doubly occupied and wrong-species sites of
the original glass as the relaxed defected glass (the site-occupancy analysis of the
manuscript); only those are scored, so the labels stay exact.

| T | Defects surviving | Same energy minimum | Median RMS shift per coordinate (survivors) |
|---|---|---|---|
| 0.02 | 638 of 734 (87%) | 71% | 0.019σ |
| 0.05 | 554 (75%) | 52% | 0.030σ |
| 0.1 | 449 (61%) | 36% | 0.045σ |
| 0.2 | 314 (43%) | 21% | 0.067σ |

Vacancies survive least (31% at T = 0.2): a neighbour hops into the empty site and the vacancy
moves. Atoms elsewhere often settle into a nearby energy minimum ("same energy minimum" is the
strict test that nothing at all changed).

**Scoring** (`scripts/pilots/displaced_inputs.py --thermal ... --minimised`, 35 minutes on 4
cores): every method on the snapshots, and on the minimised snapshots as a reference that
needs the potential. Share of the surviving defects found (%), all types:

| Method | T = 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|
| Finder trained on displaced inputs (mean of 3 runs) | 45.4 | 43.4 | 40.9 | 28.7 |
| Main finder | 46.4 | 41.2 | 34.5 | 22.9 |
| Trees D trained on displaced inputs (positions only) | 42.8 | 39.5 | 35.0 | 28.0 |
| Trees E trained on displaced inputs (with potential) | 41.9 | 39.5 | 35.0 | 27.1 |
| Unlike neighbours | 26.0 | 27.1 | 22.9 | 19.8 |
| Atom energy | 28.2 | 24.9 | 18.0 | 10.5 |
| *Main finder on the minimised snapshot (needs the potential)* | *47.8* | *45.7* | *43.6* | *37.3* |
| *Finder trained on displaced inputs, minimised snapshot* | *45.7* | *45.6* | *43.4* | *35.4* |

Each column has its own set of defects (those that survived at that temperature), so compare
methods within a column. Every method and the McNemar p values are in `summary.md`; every
outcome is in `found.json`.

**What it shows.**

- Thermal vibration costs the main finder 1.4, 4.5, 9.1 and 14.4 points at T = 0.02, 0.05,
  0.1 and 0.2 (snapshot against minimised snapshot). Trained on displaced inputs, the finder
  loses 0.3, 2.2, 2.5 and 6.7 points. The three runs agree within 1.1 points.
- Against trees trained on the same displaced inputs, the finder leads by about 3 points at
  T = 0.02 (p = 0.008-0.19 over runs and tree versions), 4 points at 0.05 (p = 0.008-0.05),
  6 points at 0.1 (p < 0.001 to 0.02, significant in every run) and 1 point at 0.2 (not
  significant). With thermal motion the learned network gains on hand-made inputs more than
  with independent shifts of similar size (6 points at T = 0.1 and 0.045σ, against 2.5 points
  for independent shifts of 0.05σ).
- At T = 0.2 all methods fall, and the network and the trees are level.
- Of the single measures, the count of unlike neighbours holds best (26-27% up to T = 0.05);
  the atom energy falls with temperature.

**Limits.** Validation glasses only; one snapshot per defect, after 10 time units; the
surviving defects differ between temperatures. The glass transition of this mixture lies
above the temperatures used; at T = 0.2 defects already move within 10 time units.
