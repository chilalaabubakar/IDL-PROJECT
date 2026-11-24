# Defects found against the size of a random displacement

Real defects of `data/ka2d_256_point_defects/val.npz`, found (%) over all four types, when every atom is
shifted by Gaussian noise of the given standard deviation (sigma). Made by
`scripts/pilots/displaced_inputs.py`.

| Method | 0 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| finder main | 52.2 | 50.1 | 39.5 | 16.5 | 8.2 |
| trees D tuned | 50.0 | 43.2 | 38.4 | 26.7 | 16.4 |
| trees E tuned | 52.2 | 46.2 | 32.8 | 22.2 | 13.4 |
| trees D displaced | 48.8 | 46.7 | 44.7 | 40.6 | 30.0 |
| trees E displaced | 48.4 | 48.5 | 46.2 | 40.5 | 28.5 |
| cell size | 24.7 | 23.8 | 22.3 | 14.6 | 8.9 |
| unlike neighbours | 29.0 | 29.3 | 28.7 | 27.9 | 10.0 |
| atom energy | 30.5 | 28.1 | 12.7 | 3.4 | 4.9 |
| local pressure | 14.6 | 13.2 | 12.1 | 5.6 | 4.4 |

By defect type at 0.02 sigma:

| Method | vacancy | interstitial | frenkel | antisite |
|---|---|---|---|---|
| finder main | 56.5 | 27.0 | 23.1 | 46.0 |
| trees D tuned | 53.5 | 28.5 | 28.4 | 40.0 |
| trees E tuned | 53.0 | 15.5 | 23.9 | 36.0 |
| trees D displaced | 53.5 | 35.5 | 44.0 | 45.5 |
| trees E displaced | 57.0 | 36.0 | 46.3 | 45.5 |
| cell size | 44.5 | 10.0 | 23.9 | 11.5 |
| unlike neighbours | 25.0 | 28.0 | 31.3 | 31.5 |
| atom energy | 6.0 | 22.0 | 14.2 | 9.0 |
| local pressure | 10.0 | 14.0 | 14.2 | 11.0 |

McNemar p over all defects, by displacement in sigma (exact, not corrected for multiple tests):

- finder main vs trees D tuned: at 0.0: p = 0.133, at 0.01: p < 0.001, at 0.02: p = 0.56, at 0.05: p < 0.001, at 0.1: p < 0.001
- finder main vs trees E tuned: at 0.0: p = 1, at 0.01: p = 0.0081, at 0.02: p < 0.001, at 0.05: p < 0.001, at 0.1: p < 0.001
- finder main vs trees D displaced: at 0.0: p = 0.0385, at 0.01: p = 0.0414, at 0.02: p = 0.0035, at 0.05: p < 0.001, at 0.1: p < 0.001
- finder main vs trees E displaced: at 0.0: p = 0.0167, at 0.01: p = 0.356, at 0.02: p < 0.001, at 0.05: p < 0.001, at 0.1: p < 0.001
