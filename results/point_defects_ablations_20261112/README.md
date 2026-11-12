# Experiments on the point-defect finder

For the report on the two-dimensional study (`paper/point_defects_report.pdf`, Sections
"Experimental iterations" and "Ablations of the finder"). Every finder is scored on the same
750 real test defects: a defect is found when the atom the finder flags lies within 1.5σ of
it. Finders are compared in pairs on the same defects (McNemar's exact test), and intervals
resample whole test glasses.

- `scripts/pilots/finder_ablations.py` writes `ablations.json` and `summary.md`.
- `scripts/pilots/finder_ablations.sh` trains the six ablation runs.

**Status, 12 November 2026:** the comparisons between the finders of the report (below) are
final. The six ablation runs are training (about 4 hours on 4 CPU cores); their table will be
added here.

## The finders of the report

| Finder | Vacancy | Interstitial | Frenkel pair | Antisite | All 750 |
|---|---|---|---|---|---|
| After 20,000 updates | 67% | 47.5% | 36.7% | 46.5% | 50.3% [46, 54] |
| After 60,000 updates (the finder of the report) | 69.5% | 50% | 45.3% | 49% | **54.0%** [50, 58] |
| The same, not calibrated | 68.5% | 47.5% | 34.7% | 48% | 50.7% [47, 54] |

- **Training length.** 40,000 more updates at half the learning rate find 3.7 points more
  defects (49 defects found only by the longer run, 21 only by the shorter; p = 0.001). The
  gain is clear for Frenkel pairs (p = 0.015) and within the noise for each other type.
- **Calibration.** Subtracting the logarithm of the class weights from the finder's scores
  (logit adjustment) finds 3.3 points more defects (38 against 13; p < 0.001), nearly all of
  it in Frenkel pairs (34.7% to 45.3%, p = 0.002). It also cuts the defect-free glasses that
  are flagged from 68% to 2% (`../point_defects_recheck_20261111/glass_level.json`).

These repeat the comparisons of `../point_defects_finder_20261109/` and
`../point_defects_finder_v2_20261110/` on the real defects only (the earlier ones included the
50 Frenkel pairs that are no defects).
