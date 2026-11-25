# Defects found in thermal snapshots

Real defects of `data/ka2d_256_point_defects/val.npz` that survived the molecular dynamics (same defect at
the same sites after minimising the snapshot), found (%) over all four types, at each
temperature T; "minimised": the minimised snapshot (a reference that needs the
potential). Made by `scripts/pilots/displaced_inputs.py --thermal`.

| Method | T0.02 | T0.02 minimised | T0.05 | T0.05 minimised | T0.1 | T0.1 minimised | T0.2 | T0.2 minimised |
|---|---|---|---|---|---|---|---|---|
| finder main | 46.4 | 47.8 | 41.2 | 45.7 | 34.5 | 43.6 | 22.9 | 37.3 |
| finder displaced_s0 | 45.1 | 45.8 | 43.3 | 45.5 | 39.6 | 43.4 | 28.0 | 35.7 |
| finder displaced_s1 | 46.1 | 45.1 | 43.0 | 45.0 | 41.6 | 42.8 | 28.7 | 34.4 |
| finder displaced_s2 | 45.0 | 46.2 | 44.0 | 46.2 | 41.4 | 44.1 | 29.3 | 36.0 |
| trees D tuned | 43.3 | 46.1 | 37.0 | 44.4 | 31.9 | 41.2 | 22.3 | 35.0 |
| trees E tuned | 44.8 | 48.4 | 38.5 | 46.2 | 33.4 | 42.5 | 20.1 | 38.2 |
| trees D displaced | 42.8 | 44.2 | 39.5 | 43.3 | 35.0 | 38.8 | 28.0 | 32.8 |
| trees E displaced | 41.9 | 43.1 | 39.5 | 43.1 | 35.0 | 39.2 | 27.1 | 32.2 |
| cell size | 21.9 | 21.8 | 20.9 | 21.1 | 16.0 | 18.0 | 8.9 | 15.0 |
| unlike neighbours | 26.0 | 26.3 | 27.1 | 27.8 | 22.9 | 24.3 | 19.8 | 18.8 |
| atom energy | 28.2 | 27.3 | 24.9 | 27.1 | 18.0 | 23.6 | 10.5 | 20.7 |
| local pressure | 12.7 | 13.8 | 11.6 | 14.3 | 9.8 | 14.9 | 5.1 | 12.1 |

Defects scored and median per-coordinate RMS displacement of the snapshot:

| Condition | vacancy | interstitial | frenkel | antisite |
|---|---|---|---|---|
| T0.02 | 168/200, 0.019σ | 180/200, 0.019σ | 117/134, 0.019σ | 173/200, 0.018σ |
| T0.05 | 136/200, 0.031σ | 163/200, 0.029σ | 96/134, 0.031σ | 159/200, 0.031σ |
| T0.1 | 106/200, 0.046σ | 131/200, 0.043σ | 78/134, 0.045σ | 134/200, 0.047σ |
| T0.2 | 61/200, 0.070σ | 97/200, 0.064σ | 49/134, 0.064σ | 107/200, 0.070σ |

By defect type at T0.05:

| Method | vacancy | interstitial | frenkel | antisite |
|---|---|---|---|---|
| finder main | 47.8 | 39.3 | 31.2 | 43.4 |
| finder displaced_s0 | 49.3 | 39.9 | 41.7 | 42.8 |
| finder displaced_s1 | 48.5 | 40.5 | 42.7 | 40.9 |
| finder displaced_s2 | 49.3 | 41.1 | 45.8 | 41.5 |
| trees D tuned | 44.9 | 31.9 | 29.2 | 40.2 |
| trees E tuned | 47.8 | 34.4 | 31.2 | 39.0 |
| trees D displaced | 44.9 | 33.1 | 45.8 | 37.7 |
| trees E displaced | 44.1 | 31.3 | 44.8 | 40.9 |
| cell size | 37.5 | 15.3 | 25.0 | 10.1 |
| unlike neighbours | 19.9 | 27.0 | 33.3 | 29.6 |
| atom energy | 18.4 | 26.4 | 28.1 | 27.0 |
| local pressure | 6.6 | 11.0 | 15.6 | 13.8 |

McNemar p over all defects, by condition (exact, not corrected for multiple tests):

- finder main vs trees D tuned: at T0.02: p = 0.0619, at T0.02 minimised: p = 0.284, at T0.05: p = 0.014, at T0.05 minimised: p = 0.47, at T0.1: p = 0.207, at T0.1 minimised: p = 0.207, at T0.2: p = 0.888, at T0.2 minimised: p = 0.349
- finder main vs trees E tuned: at T0.02: p = 0.378, at T0.02 minimised: p = 0.762, at T0.05: p = 0.124, at T0.05 minimised: p = 0.826, at T0.1: p = 0.609, at T0.1 minimised: p = 0.644, at T0.2: p = 0.243, at T0.2 minimised: p = 0.771
- finder main vs trees D displaced: at T0.02: p = 0.0487, at T0.02 minimised: p = 0.0346, at T0.05: p = 0.464, at T0.05 minimised: p = 0.246, at T0.1: p = 0.912, at T0.1 minimised: p = 0.0183, at T0.2: p = 0.0519, at T0.2 minimised: p = 0.109
- finder main vs trees E displaced: at T0.02: p = 0.0087, at T0.02 minimised: p = 0.0054, at T0.05: p = 0.439, at T0.05 minimised: p = 0.202, at T0.1: p = 0.914, at T0.1 minimised: p = 0.0352, at T0.2: p = 0.117, at T0.2 minimised: p = 0.0599
- finder displaced_s0 vs trees D tuned: at T0.02: p = 0.285, at T0.02 minimised: p = 0.919, at T0.05: p < 0.001, at T0.05 minimised: p = 0.576, at T0.1: p < 0.001, at T0.1 minimised: p = 0.268, at T0.2: p = 0.0222, at T0.2 minimised: p = 0.892
- finder displaced_s0 vs trees E tuned: at T0.02: p = 0.923, at T0.02 minimised: p = 0.125, at T0.05: p = 0.0116, at T0.05 minimised: p = 0.755, at T0.1: p = 0.0034, at T0.1 minimised: p = 0.72, at T0.2: p < 0.001, at T0.2 minimised: p = 0.358
- finder displaced_s0 vs trees D displaced: at T0.02: p = 0.167, at T0.02 minimised: p = 0.358, at T0.05: p = 0.0375, at T0.05 minimised: p = 0.235, at T0.1: p = 0.0203, at T0.1 minimised: p = 0.0125, at T0.2: p = 1, at T0.2 minimised: p = 0.289
- finder displaced_s0 vs trees E displaced: at T0.02: p = 0.0375, at T0.02 minimised: p = 0.1, at T0.05: p = 0.0334, at T0.05 minimised: p = 0.182, at T0.1: p = 0.0186, at T0.1 minimised: p = 0.0248, at T0.2: p = 0.749, at T0.2 minimised: p = 0.185
- finder displaced_s1 vs trees D tuned: at T0.02: p = 0.105, at T0.02 minimised: p = 0.617, at T0.05: p = 0.0025, at T0.05 minimised: p = 0.822, at T0.1: p < 0.001, at T0.1 minimised: p = 0.483, at T0.2: p = 0.0119, at T0.2 minimised: p = 0.894
- finder displaced_s1 vs trees E tuned: at T0.02: p = 0.505, at T0.02 minimised: p = 0.0572, at T0.05: p = 0.0287, at T0.05 minimised: p = 0.53, at T0.1: p < 0.001, at T0.1 minimised: p = 1, at T0.2: p < 0.001, at T0.2 minimised: p = 0.162
- finder displaced_s1 vs trees D displaced: at T0.02: p = 0.0482, at T0.02 minimised: p = 0.61, at T0.05: p = 0.0503, at T0.05 minimised: p = 0.386, at T0.1: p < 0.001, at T0.1 minimised: p = 0.0356, at T0.2: p = 0.88, at T0.2 minimised: p = 0.597
- finder displaced_s1 vs trees E displaced: at T0.02: p = 0.008, at T0.02 minimised: p = 0.218, at T0.05: p = 0.053, at T0.05 minimised: p = 0.314, at T0.1: p < 0.001, at T0.1 minimised: p = 0.0681, at T0.2: p = 0.533, at T0.2 minimised: p = 0.427
- finder displaced_s2 vs trees D tuned: at T0.02: p = 0.338, at T0.02 minimised: p = 1, at T0.05: p < 0.001, at T0.05 minimised: p = 0.314, at T0.1: p < 0.001, at T0.1 minimised: p = 0.148, at T0.2: p = 0.0062, at T0.2 minimised: p = 0.788
- finder displaced_s2 vs trees E tuned: at T0.02: p = 1, at T0.02 minimised: p = 0.202, at T0.05: p = 0.0049, at T0.05 minimised: p = 1, at T0.1: p < 0.001, at T0.1 minimised: p = 0.464, at T0.2: p < 0.001, at T0.2 minimised: p = 0.435
- finder displaced_s2 vs trees D displaced: at T0.02: p = 0.193, at T0.02 minimised: p = 0.218, at T0.05: p = 0.008, at T0.05 minimised: p = 0.105, at T0.1: p < 0.001, at T0.1 minimised: p = 0.0043, at T0.2: p = 0.636, at T0.2 minimised: p = 0.245
- finder displaced_s2 vs trees E displaced: at T0.02: p = 0.0495, at T0.02 minimised: p = 0.0495, at T0.05: p = 0.008, at T0.05 minimised: p = 0.0748, at T0.1: p < 0.001, at T0.1 minimised: p = 0.0081, at T0.2: p = 0.324, at T0.2 minimised: p = 0.155
