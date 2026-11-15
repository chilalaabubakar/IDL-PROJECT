<!-- written by scripts/pilots/finder_architectures.py -->

Found on the real test defects (Vacancy 200, Interstitial 200, Frenkel pair 150, Antisite 200), calibrated. Mean over seeds, lowest and highest seed in brackets.

| Finder | Parameters | Seeds | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|---|---|
| Transformer: attention over all atoms, distances as bias | 0.58 M | 3 | 67.2 [66.5, 68.0] | 48.3 [46.5, 49.5] | 42.0 [40.0, 44.7] | 54.2 [53.0, 55.0] | 53.6 [53.3, 53.9] |
| CNN: periodic U-Net on the glass drawn as an image | 0.50 M | 3 | 69.7 [69.0, 70.0] | 51.2 [50.5, 52.0] | 38.4 [35.3, 41.3] | 52.8 [52.0, 53.5] | 54.0 [53.5, 54.4] |
| EGNN (control): message passing, reach 10 sigma | 0.52 M | 3 | 63.2 [62.5, 63.5] | 29.7 [29.0, 30.0] | 35.1 [34.0, 36.0] | 44.0 [42.0, 46.0] | 43.5 [42.9, 43.9] |
| the finder of the report after 20,000 updates (from the denoiser) | 0.52 M | 1 | 67.0 [67.0, 67.0] | 47.5 [47.5, 47.5] | 36.7 [36.7, 36.7] | 46.5 [46.5, 46.5] | 50.3 [50.3, 50.3] |
| the finder of the report, 60,000 updates (from the denoiser) | 0.52 M | 1 | 69.5 [69.5, 69.5] | 50.0 [50.0, 50.0] | 45.3 [45.3, 45.3] | 49.0 [49.0, 49.0] | 54.0 [54.0, 54.0] |

Against the EGNN control with the same seed (all defects; defects found only by the model / only by the control, McNemar p):

- attention, seed 0: 110 / 36, p = 6.55e-10
- attention, seed 1: 116 / 34, p = 1.12e-11
- attention, seed 2: 113 / 41, p = 5.61e-09
- cnn, seed 0: 114 / 42, p = 7.14e-09
- cnn, seed 1: 122 / 38, p = 1.8e-11
- cnn, seed 2: 122 / 42, p = 3.05e-10
- egnn_pretrained, seed 0: 91 / 43, p = 4.11e-05
- egnn_pretrained_60k, seed 0: 122 / 46, p = 3.93e-09

Finder against finder with the same seed (defects found only by the first / only by the second, McNemar p; all defects, then each type):

- attention vs cnn, seed 0: 41 / 39, p = 0.911; Vacancy 6 / 10 (p = 0.45), Interstitial 6 / 9 (p = 0.61), Frenkel pair 17 / 10 (p = 0.25), Antisite 12 / 10 (p = 0.83)
- attention vs cnn, seed 1: 39 / 41, p = 0.911; Vacancy 4 / 11 (p = 0.12), Interstitial 5 / 10 (p = 0.3), Frenkel pair 14 / 10 (p = 0.54), Antisite 16 / 10 (p = 0.33)
- attention vs cnn, seed 2: 48 / 56, p = 0.493; Vacancy 11 / 15 (p = 0.56), Interstitial 3 / 12 (p = 0.035), Frenkel pair 17 / 12 (p = 0.46), Antisite 17 / 17 (p = 1)
- attention vs egnn_scratch, seed 0: 110 / 36, p = 6.55e-10; Vacancy 11 / 2 (p = 0.022), Interstitial 42 / 4 (p = 5.1e-09), Frenkel pair 25 / 19 (p = 0.45), Antisite 32 / 11 (p = 0.0019)
- attention vs egnn_scratch, seed 1: 116 / 34, p = 1.12e-11; Vacancy 10 / 4 (p = 0.18), Interstitial 46 / 5 (p = 2.3e-09), Frenkel pair 23 / 14 (p = 0.19), Antisite 37 / 11 (p = 0.00022)
- attention vs egnn_scratch, seed 2: 113 / 41, p = 5.61e-09; Vacancy 15 / 6 (p = 0.078), Interstitial 41 / 8 (p = 2e-06), Frenkel pair 31 / 15 (p = 0.026), Antisite 26 / 12 (p = 0.034)
- cnn vs egnn_scratch, seed 0: 114 / 42, p = 7.14e-09; Vacancy 18 / 5 (p = 0.011), Interstitial 43 / 2 (p = 5.9e-11), Frenkel pair 23 / 24 (p = 1), Antisite 30 / 11 (p = 0.0043)
- cnn vs egnn_scratch, seed 1: 122 / 38, p = 1.8e-11; Vacancy 17 / 4 (p = 0.0072), Interstitial 48 / 2 (p = 2.3e-12), Frenkel pair 22 / 17 (p = 0.52), Antisite 35 / 15 (p = 0.0066)
- cnn vs egnn_scratch, seed 2: 122 / 42, p = 3.05e-10; Vacancy 22 / 9 (p = 0.029), Interstitial 47 / 5 (p = 1.3e-09), Frenkel pair 25 / 14 (p = 0.11), Antisite 28 / 14 (p = 0.044)

Do the finders find the same defects (same seed)?

| Seed | Found by all | By none | By at least one |
|---|---|---|---|
| 0 | 36.7% | 37.9% | 62.1% |
| 1 | 36.4% | 37.6% | 62.4% |
| 2 | 35.9% | 36.0% | 64.0% |

Finders used together (the mean of their calibrated probabilities; 95% interval over glasses; against its best single member: found only together / only by the member, McNemar p):

| Together | Vacancy | Interstitial | Frenkel pair | Antisite | All | Against best member |
|---|---|---|---|---|---|---|
| transformer, 3 seeds | 68.0 | 48.0 | 42.7 | 56.5 | 54.5 [50.6, 58.3] | 16 / 11, p = 0.44 (attention_s1) |
| CNN, 3 seeds | 70.5 | 51.5 | 38.7 | 54.0 | 54.7 [50.8, 58.5] | 17 / 15, p = 0.86 (cnn_s2) |
| EGNN control, 3 seeds | 65.0 | 30.0 | 32.7 | 45.5 | 44.0 [40.1, 48.0] | 15 / 14, p = 1 (egnn_scratch_s0) |
| transformer + CNN, seed 0 | 69.0 | 51.5 | 41.3 | 55.5 | 55.2 [51.5, 59.0] | 23 / 12, p = 0.09 (attention_s0) |
| transformer + CNN + EGNN control, seed 0 | 68.0 | 50.0 | 40.7 | 54.5 | 54.1 [50.2, 58.0] | 25 / 22, p = 0.77 (attention_s0) |
| transformer + CNN, 3 seeds each | 70.0 | 49.5 | 41.3 | 56.0 | 55.1 [51.2, 58.9] | 29 / 24, p = 0.58 (cnn_s2) |
| transformer + CNN (seed 0) + the finder of the report | 70.0 | 51.0 | 41.3 | 55.0 | 55.2 [51.3, 59.0] | 34 / 25, p = 0.3 (egnn_pretrained_60k_s0) |

| Finder | Same flagged atom: turn by 90 degrees | random shift | renumbering |
|---|---|---|---|
| attention | 100.0% | 100.0% | 100.0% |
| cnn | 100.0% | 88.3% | 100.0% |
| egnn_scratch | 100.0% | 100.0% | 100.0% |
| egnn_pretrained | 100.0% | 100.0% | 100.0% |
| egnn_pretrained_60k | 100.0% | 100.0% | 100.0% |

| Run | Grad. norm median (last 1,000) | 99th pct | Largest | Clipped (last 1,000) | Grad. spikes | Loss spikes | Not finite |
|---|---|---|---|---|---|---|---|
| attention seed 0 | 0.928 (0.845) | 2.355 | 4.907 | 40.4% (28.8%) | 0 | 0 | 0 |
| attention seed 1 | 0.968 (0.879) | 2.396 | 4.863 | 45.8% (32.1%) | 0 | 0 | 0 |
| attention seed 2 | 1.002 (0.887) | 2.666 | 5.532 | 50.2% (34.4%) | 0 | 1 | 0 |
| cnn seed 0 | 1.784 (1.605) | 3.513 | 7.236 | 84.7% (70.3%) | 0 | 0 | 0 |
| cnn seed 1 | 1.65 (1.55) | 3.351 | 8.077 | 85.1% (72.9%) | 0 | 0 | 0 |
| cnn seed 2 | 1.67 (1.47) | 3.421 | 6.279 | 85.2% (71.7%) | 0 | 0 | 0 |
| egnn_scratch seed 0 | 1.475 (1.476) | 5.335 | 55.282 | 84.7% (78.5%) | 71 | 14 | 0 |
| egnn_scratch seed 1 | 1.573 (1.378) | 6.749 | 82.026 | 88.4% (79.9%) | 131 | 22 | 0 |
| egnn_scratch seed 2 | 1.569 (1.432) | 6.805 | 61.078 | 86.9% (79.4%) | 96 | 34 | 0 |
