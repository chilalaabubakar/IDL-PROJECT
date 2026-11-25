# Defects found against the size of a random displacement

Real defects of `data/ka2d_256_point_defects/val.npz`, found (%) over all four types, when every atom is
shifted by Gaussian noise of the given standard deviation (sigma). Made by
`scripts/pilots/displaced_inputs.py`.

| Method | 0 | 0.02 |
|---|---|---|
| finder run1 | 52.2 | 39.5 |
| finder run2 | 52.6 | 40.3 |
| finder run3 | 53.1 | 38.4 |
| cell size | 24.7 | 22.3 |
| unlike neighbours | 29.0 | 28.7 |
| atom energy | 30.5 | 12.7 |
| local pressure | 14.6 | 12.1 |

By defect type at 0.02 sigma:

| Method | vacancy | interstitial | frenkel | antisite |
|---|---|---|---|---|
| finder run1 | 56.5 | 27.0 | 23.1 | 46.0 |
| finder run2 | 59.5 | 29.5 | 22.4 | 44.0 |
| finder run3 | 55.5 | 29.0 | 21.6 | 42.0 |
| cell size | 44.5 | 10.0 | 23.9 | 11.5 |
| unlike neighbours | 25.0 | 28.0 | 31.3 | 31.5 |
| atom energy | 6.0 | 22.0 | 14.2 | 9.0 |
| local pressure | 10.0 | 14.0 | 14.2 | 11.0 |
