<!-- written by scripts/pilots/finder_architectures.py -->

Found on the real test defects (Vacancy 200, Interstitial 200, Frenkel pair 150, Antisite 200), calibrated. Mean over seeds, lowest and highest seed in brackets.

| Finder | Parameters | Seeds | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|---|---|
| Transformer: attention over all atoms, distances as bias | 0.58 M | 3 | 67.2 [66.5, 68.0] | 48.3 [46.5, 49.5] | 42.0 [40.0, 44.7] | 54.2 [53.0, 55.0] | 53.6 [53.3, 53.9] |
| EGNN (control): message passing, reach 10 sigma | 0.52 M | 3 | 63.2 [62.5, 63.5] | 29.7 [29.0, 30.0] | 35.1 [34.0, 36.0] | 44.0 [42.0, 46.0] | 43.5 [42.9, 43.9] |
| the finder of the report after 20,000 updates (from the denoiser) | 0.52 M | 1 | 67.0 [67.0, 67.0] | 47.5 [47.5, 47.5] | 36.7 [36.7, 36.7] | 46.5 [46.5, 46.5] | 50.3 [50.3, 50.3] |

Against the EGNN control with the same seed (all defects; defects found only by the model / only by the control, McNemar p):

- attention, seed 0: 110 / 36, p = 6.55e-10
- attention, seed 1: 116 / 34, p = 1.12e-11
- attention, seed 2: 113 / 41, p = 5.61e-09
- egnn_pretrained, seed 0: 91 / 43, p = 4.11e-05

| Finder | Same flagged atom: turn by 90 degrees | random shift | renumbering |
|---|---|---|---|
| attention | 100.0% | 100.0% | 100.0% |
| egnn_scratch | 100.0% | 100.0% | 100.0% |
| egnn_pretrained | 100.0% | 100.0% | 100.0% |

| Run | Grad. norm median (last 1,000) | 99th pct | Largest | Clipped (last 1,000) | Grad. spikes | Loss spikes | Not finite |
|---|---|---|---|---|---|---|---|
| attention seed 0 | 0.928 (0.845) | 2.355 | 4.907 | 40.4% (28.8%) | 0 | 0 | 0 |
| attention seed 1 | 0.968 (0.879) | 2.396 | 4.863 | 45.8% (32.1%) | 0 | 0 | 0 |
| attention seed 2 | 1.002 (0.887) | 2.666 | 5.532 | 50.2% (34.4%) | 0 | 1 | 0 |
| egnn_scratch seed 0 | 1.475 (1.476) | 5.335 | 55.282 | 84.7% (78.5%) | 71 | 14 | 0 |
| egnn_scratch seed 1 | 1.573 (1.378) | 6.749 | 82.026 | 88.4% (79.9%) | 131 | 22 | 0 |
| egnn_scratch seed 2 | 1.569 (1.432) | 6.805 | 61.078 | 86.9% (79.4%) | 96 | 34 | 0 |
