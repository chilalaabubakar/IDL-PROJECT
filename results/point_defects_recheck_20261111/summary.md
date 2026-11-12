<!-- written by paper/make_report_numbers.py -->

Found (the flagged atom within 1.5σ of the defect):

| Defect | n | Learned finder | Cell size | Unlike neighbours | Atom energy | Random atom |
|---|---|---|---|---|---|---|
| Vacancy | 200 | 69.5% [62, 76] | 47.5% [40, 55] | 29% [22, 36] | 29.5% [23, 36] | 2.9% |
| Interstitial | 200 | 50% [42, 57] | 15% [10, 20] | 28% [22, 34] | 28% [22, 34] | 3% |
| Antisite | 200 | 49% [42, 56] | 16.5% [12, 22] | 31.5% [25, 38] | 29% [23, 35] | 4% |
| Frenkel pair | 150 | 45.3% [38, 53] | 26.7% [20, 33] | 27.3% [21, 34] | 32.7% [25, 40] | 5.4% |

Restored (the original glass after correcting the atoms and minimising):

| Defect | Learned finder | Cell size | Unlike neighbours | Atom energy | True location given |
|---|---|---|---|---|---|
| Vacancy | 53.5% [46, 61] | 40.5% [34, 48] | 22% [16, 28] | 26% [20, 32] | 67% [60, 74] |
| Interstitial | 38% [31, 46] | 8.5% [4, 13] | 21.5% [16, 27] | 21% [16, 26] | 80% [74, 86] |
| Antisite | 30% [24, 36] | 1.5% [0, 4] | 7.5% [4, 11] | 8% [4, 12] | 68.5% [62, 74] |
| Frenkel pair | 12.7% [8, 18] | 0% [0, 2] | 1.3% [0, 3] | 7.3% [3, 11] | 42% [34, 50] |

Learned finder against each baseline on the same defects (McNemar's exact test; the counts are defects that only one of the two gets right):

| Defect | Comparison | Only the finder | Only the baseline | p |
|---|---|---|---|---|
| Vacancy | found: finder vs cell size | 49 | 5 | < 0.001 |
| Vacancy | restored: finder vs cell size | 31 | 5 | < 0.001 |
| Vacancy | found: finder vs unlike neighbours | 87 | 6 | < 0.001 |
| Vacancy | restored: finder vs unlike neighbours | 65 | 2 | < 0.001 |
| Vacancy | found: finder vs atom energy | 85 | 5 | < 0.001 |
| Vacancy | restored: finder vs atom energy | 58 | 3 | < 0.001 |
| Interstitial | found: finder vs cell size | 79 | 9 | < 0.001 |
| Interstitial | restored: finder vs cell size | 62 | 3 | < 0.001 |
| Interstitial | found: finder vs unlike neighbours | 50 | 6 | < 0.001 |
| Interstitial | restored: finder vs unlike neighbours | 37 | 4 | < 0.001 |
| Interstitial | found: finder vs atom energy | 48 | 4 | < 0.001 |
| Interstitial | restored: finder vs atom energy | 36 | 2 | < 0.001 |
| Antisite | found: finder vs cell size | 74 | 9 | < 0.001 |
| Antisite | restored: finder vs cell size | 57 | 0 | < 0.001 |
| Antisite | found: finder vs unlike neighbours | 49 | 14 | < 0.001 |
| Antisite | restored: finder vs unlike neighbours | 47 | 2 | < 0.001 |
| Antisite | found: finder vs atom energy | 55 | 15 | < 0.001 |
| Antisite | restored: finder vs atom energy | 47 | 3 | < 0.001 |
| Frenkel pair | found: finder vs cell size | 47 | 19 | < 0.001 |
| Frenkel pair | restored: finder vs cell size | 19 | 0 | < 0.001 |
| Frenkel pair | found: finder vs unlike neighbours | 38 | 11 | < 0.001 |
| Frenkel pair | restored: finder vs unlike neighbours | 18 | 1 | < 0.001 |
| Frenkel pair | found: finder vs atom energy | 37 | 18 | 0.014 |
| Frenkel pair | restored: finder vs atom energy | 17 | 9 | 0.169 |
