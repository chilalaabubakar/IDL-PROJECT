<!-- written by scripts/pilots/finder_ablations.py -->

Found on the real test defects (Vacancy 200, Interstitial 200, Frenkel pair 150, Antisite 200). Brackets: 95% intervals from resampling test glasses.

The finders of the report:

| Finder | Vacancy | Interstitial | Frenkel pair | Antisite | All |
|---|---|---|---|---|---|
| 20,000 updates | 67.0% [60, 74] | 47.5% [40, 55] | 36.7% [29, 45] | 46.5% [40, 54] | 50.3% [46, 54] |
| 60,000 updates | 69.5% [62, 76] | 50.0% [42, 57] | 45.3% [38, 53] | 49.0% [42, 56] | 54.0% [50, 58] |
| 60,000 updates, not calibrated | 68.5% [62, 75] | 47.5% [40, 55] | 34.7% [28, 42] | 48.0% [41, 56] | 50.7% [47, 54] |

Paired comparisons on the same defects (McNemar's exact test; counts over all defects):

| Comparison | Only the first | Only the second | p (all) | p per type |
|---|---|---|---|---|
| 60,000 against 20,000 updates | 49 (60,000) | 21 (20,000 updates) | 0.0011 | Vacancy 0.227, Interstitial 0.302, Frenkel pair 0.015, Antisite 0.359 |
| calibrated against not calibrated | 38 (calibrated) | 13 (not calibrated) | 0.0006 | Vacancy 0.727, Interstitial 0.180, Frenkel pair 0.002, Antisite 0.754 |
