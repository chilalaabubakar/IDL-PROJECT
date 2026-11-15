<!-- written by scripts/pilots/finder_seeds.py -->

## Training stability

| Run | Updates | Grad. norm: median (first 100 / last 1,000) | 99th pct | Largest (update) | Clipped: all / last 1,000 | Grad. spikes | Loss spikes | Not finite |
|---|---|---|---|---|---|---|---|---|
| layers1 seed 1 | 2000 | 1.407 (1.533 / 1.364) | 2.914 | 3.621 (38, frenkel) | 93.4% / 92.5% | 0 | 0 | 0 |
| layers1 seed 2 | 2000 | 1.413 (1.517 / 1.333) | 2.924 | 3.848 (1259, frenkel) | 91.5% / 90.4% | 0 | 0 | 0 |
| layers2 seed 1 | 2000 | 1.462 (1.552 / 1.423) | 3.134 | 3.856 (1755, frenkel) | 92.6% / 92.2% | 0 | 0 | 0 |
| layers2 seed 2 | 2000 | 1.365 (1.463 / 1.309) | 3.188 | 4.125 (1259, frenkel) | 87.4% / 85.1% | 0 | 0 | 0 |
| no_weights seed 1 | 2000 | 0.311 (0.422 / 0.266) | 1.282 | 6.218 (1, vacancy) | 1.3% / 0.1% | 3 | 5 | 0 |
| no_weights seed 2 | 2000 | 0.343 (0.503 / 0.292) | 1.546 | 6.696 (5, defect-free) | 2.9% / 0.6% | 4 | 11 | 0 |
| real_only seed 1 | 2000 | 1.944 (2.531 / 1.709) | 6.853 | 25.812 (494, defect-free) | 93.1% / 90.8% | 7 | 4 | 0 |
| real_only seed 2 | 2000 | 2.519 (2.725 / 2.221) | 12.855 | 71.809 (650, antisite) | 96.1% / 94.0% | 21 | 17 | 0 |
| reference seed 0 | 2000 | 2.173 (2.237 / 1.948) | 12.927 | 44.104 (1456, defect-free) | 95.4% / 92.7% | 24 | 11 | 0 |
| reference seed 1 | 2000 | 1.99 (2.462 / 1.749) | 7.127 | 37.623 (813, antisite) | 94.6% / 92.0% | 9 | 1 | 0 |
| reference seed 2 | 2000 | 2.456 (2.782 / 2.247) | 16.791 | 85.972 (651, vacancy) | 96.4% / 93.7% | 32 | 13 | 0 |
| scratch seed 0 | 2000 | 1.388 (1.57 / 1.376) | 3.188 | 4.101 (1862, frenkel) | 92.2% / 91.2% | 0 | 0 | 0 |
| scratch seed 1 | 2000 | 1.445 (1.71 / 1.407) | 3.2 | 5.177 (1747, frenkel) | 92.5% / 91.3% | 0 | 0 | 0 |
| scratch seed 2 | 2000 | 1.53 (1.697 / 1.443) | 3.549 | 4.874 (1259, frenkel) | 93.2% / 92.9% | 0 | 0 | 0 |

## Found, per variant (mean over seeds, with the lowest and highest seed)

| Variant | Seeds | Vacancy | Interstitial | Frenkel pair | Antisite | All | Spread of All (sd) | Difference from the reference, same seed |
|---|---|---|---|---|---|---|---|---|
| reference: starts from the denoiser, 4 layers, class weights | 0, 1, 2 | 63.0 [62.5, 64.0] | 30.7 [29.5, 31.5] | 38.7 [34.7, 42.0] | 41.7 [40.5, 42.5] | 43.8 [43.2, 44.7] | 0.8 |  |
| scratch: random starting weights | 0, 1, 2 | 14.3 [11.5, 17.0] | 26.8 [26.0, 27.5] | 22.2 [19.3, 24.0] | 24.2 [18.0, 28.0] | 21.9 [18.7, 23.7] | 2.8 | -22.0 (lower in 3 of 3 seeds) |
| layers2: random starting weights, 2 layers | 0, 1, 2 | 13.8 [12.5, 15.5] | 26.5 [25.5, 27.5] | 21.3 [18.7, 24.7] | 20.8 [16.5, 25.0] | 20.6 [18.3, 23.1] | 2.4 | -23.2 (lower in 3 of 3 seeds) |
| layers1: random starting weights, 1 layer | 0, 1, 2 | 12.3 [11.5, 13.5] | 21.8 [21.0, 23.0] | 13.8 [12.0, 17.3] | 14.2 [13.0, 15.5] | 15.6 [14.8, 17.3] | 1.5 | -28.2 (lower in 3 of 3 seeds) |
| no_weights: no class weights | 0, 1, 2 | 63.2 [61.0, 65.0] | 31.0 [30.5, 31.5] | 35.1 [30.7, 38.0] | 39.5 [38.5, 41.0] | 42.7 [41.6, 43.5] | 1.0 | -1.2 (lower in 3 of 3 seeds) |
| real_only: trained without the annihilated Frenkel pairs | 0, 1, 2 | 64.7 [63.0, 67.0] | 30.8 [29.5, 31.5] | 38.0 [35.3, 41.3] | 40.2 [38.0, 43.5] | 43.8 [42.1, 46.1] | 2.1 | -0.0 (lower in 2 of 3 seeds) |

## Every run

| Run | Source | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|---|
| layers1 seed 0 | ablation run | 13.5 [10, 18] | 23.0 [17, 30] | 17.3 [11, 24] | 15.5 [10, 20] | 17.3 [14, 20] |
| layers1 seed 1 | seed study | 12.0 [8, 16] | 21.5 [16, 28] | 12.0 [7, 17] | 13.0 [8, 18] | 14.8 [12, 18] |
| layers1 seed 2 | seed study | 11.5 [8, 16] | 21.0 [15, 28] | 12.0 [7, 18] | 14.0 [9, 19] | 14.8 [12, 17] |
| layers2 seed 0 | ablation run | 15.5 [11, 20] | 27.5 [22, 34] | 24.7 [17, 33] | 25.0 [19, 31] | 23.1 [20, 27] |
| layers2 seed 1 | seed study | 13.5 [10, 18] | 26.5 [20, 33] | 20.7 [14, 28] | 21.0 [16, 26] | 20.4 [17, 24] |
| layers2 seed 2 | seed study | 12.5 [8, 17] | 25.5 [19, 32] | 18.7 [12, 26] | 16.5 [12, 22] | 18.3 [15, 21] |
| no_weights seed 0 | ablation run | 61.0 [54, 68] | 31.5 [25, 38] | 36.7 [28, 45] | 41.0 [34, 48] | 42.9 [39, 47] |
| no_weights seed 1 | seed study | 63.5 [56, 70] | 30.5 [24, 37] | 30.7 [23, 39] | 39.0 [32, 46] | 41.6 [37, 46] |
| no_weights seed 2 | seed study | 65.0 [58, 72] | 31.0 [24, 38] | 38.0 [30, 46] | 38.5 [32, 46] | 43.5 [39, 48] |
| real_only seed 0 | ablation run | 64.0 [57, 71] | 29.5 [24, 36] | 35.3 [28, 43] | 38.0 [32, 44] | 42.1 [38, 46] |
| real_only seed 1 | seed study | 63.0 [56, 70] | 31.5 [25, 38] | 37.3 [30, 45] | 39.0 [32, 46] | 43.1 [39, 47] |
| real_only seed 2 | seed study | 67.0 [60, 74] | 31.5 [25, 38] | 41.3 [34, 49] | 43.5 [36, 50] | 46.1 [42, 50] |
| reference seed 0 | seed study | 64.0 [56, 72] | 29.5 [24, 36] | 42.0 [34, 50] | 42.5 [36, 50] | 44.7 [40, 49] |
| reference seed 1 | seed study | 62.5 [55, 70] | 31.5 [25, 38] | 34.7 [27, 43] | 42.0 [35, 49] | 43.2 [39, 48] |
| reference seed 2 | seed study | 62.5 [55, 70] | 31.0 [25, 38] | 39.3 [32, 47] | 40.5 [34, 48] | 43.6 [39, 48] |
| scratch seed 0 | seed study | 17.0 [12, 22] | 27.5 [21, 34] | 24.0 [16, 32] | 26.5 [20, 33] | 23.7 [20, 27] |
| scratch seed 1 | seed study | 14.5 [10, 20] | 27.0 [21, 34] | 23.3 [16, 31] | 28.0 [22, 34] | 23.2 [20, 27] |
| scratch seed 2 | seed study | 11.5 [8, 16] | 26.0 [20, 32] | 19.3 [13, 26] | 18.0 [13, 24] | 18.7 [16, 22] |

Reference, seeds [0, 1, 2]: found by all 37.7%, by some 12.1%, by none 50.1% of the defects. The seeds averaged (ensemble) find 45.1% [41, 49].

Seed-0 reruns with the same weights as the published run: reference yes, scratch yes.
