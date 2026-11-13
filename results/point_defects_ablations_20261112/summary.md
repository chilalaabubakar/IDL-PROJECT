<!-- written by scripts/pilots/finder_ablations.py -->

Found on the real test defects (Vacancy 200, Interstitial 200, Frenkel pair 150, Antisite 200). Brackets: 95% intervals from resampling test glasses.

The finders of the report:

| Finder | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|
| 20,000 updates | 67.0% [60, 74] | 47.5% [40, 55] | 36.7% [29, 45] | 46.5% [40, 54] | 50.3% [46, 54] |
| 60,000 updates | 69.5% [62, 76] | 50.0% [42, 57] | 45.3% [38, 53] | 49.0% [42, 56] | 54.0% [50, 58] |
| 60,000 updates, not calibrated | 68.5% [62, 75] | 47.5% [40, 55] | 34.7% [28, 42] | 48.0% [41, 56] | 50.7% [47, 54] |

Ablation runs (2,000 updates each, trained weights):

| Finder | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|
| reference: starts from the denoiser, 4 layers, class weights | 64.0% [56, 72] | 29.5% [24, 36] | 42.0% [34, 50] | 42.5% [36, 50] | 44.7% [40, 49] |
| scratch: random starting weights | 17.0% [12, 22] | 27.5% [22, 34] | 24.0% [16, 32] | 26.5% [20, 33] | 23.7% [20, 27] |
| layers2: random starting weights, 2 layers | 15.5% [11, 20] | 27.5% [21, 34] | 24.7% [17, 33] | 25.0% [19, 31] | 23.1% [20, 27] |
| layers1: random starting weights, 1 layer | 13.5% [10, 18] | 23.0% [17, 30] | 17.3% [11, 24] | 15.5% [10, 21] | 17.3% [14, 20] |
| no_weights: no class weights | 61.0% [54, 68] | 31.5% [25, 38] | 36.7% [28, 45] | 41.0% [34, 48] | 42.9% [39, 47] |
| real_only: trained without the annihilated Frenkel pairs | 64.0% [57, 71] | 29.5% [24, 36] | 35.3% [28, 43] | 38.0% [32, 44] | 42.1% [38, 46] |

Paired comparisons on the same defects (McNemar's exact test; counts over all defects):

| Comparison | Only the first | Only the second | p (all) | p per type |
|---|---|---|---|---|
| 60,000 against 20,000 updates | 49 (60,000) | 21 (20,000 updates) | 0.0011 | Vacancy 0.227, Interstitial 0.302, Frenkel pair 0.015, Antisite 0.359 |
| calibrated against not calibrated | 38 (calibrated) | 13 (not calibrated) | 0.0006 | Vacancy 0.727, Interstitial 0.180, Frenkel pair 0.002, Antisite 0.754 |
| scratch against reference | 26 (scratch) | 183 (reference) | 0.0000 | Vacancy 0.000, Interstitial 0.454, Frenkel pair 0.000, Antisite 0.000 |
| layers2 against scratch | 12 (layers2) | 17 (scratch) | 0.4583 | Vacancy 0.581, Interstitial 1.000, Frenkel pair 1.000, Antisite 0.250 |
| layers1 against scratch | 20 (layers1) | 68 (scratch) | 0.0000 | Vacancy 0.143, Interstitial 0.122, Frenkel pair 0.031, Antisite 0.000 |
| no_weights against reference | 23 (no_weights) | 36 (reference) | 0.1175 | Vacancy 0.146, Interstitial 0.289, Frenkel pair 0.096, Antisite 0.664 |
| real_only against reference | 24 (real_only) | 43 (reference) | 0.0271 | Vacancy 1.000, Interstitial 1.000, Frenkel pair 0.064, Antisite 0.093 |
