# Analyses for the final course report

Written by `scripts/pilots/final_report_analysis.py` from trained weights only (no training);
used by `paper/make_final_numbers.py`, `paper/make_final_figures.py` and `paper/final_report.tex`.

- `shapes.json`: output shape and parameters of every block of the EGNN finder and denoiser,
  the transformer and the CNN, for 16 glasses of 256 atoms (finder 523,021, denoiser 530,696,
  transformer 576,677, CNN 498,309 parameters).
- `splits.json`: the finder of the main results on training (200 random defects per type),
  validation and test defects (real defects only). Found, all defects: training
  50.5%, validation 52.2%, test 54.0%; the test numbers
  equal those of the report, so the finder does not overfit.
- `features.json`, `features.npz`: the 128 features of 4,159 atoms of test glasses after the
  last layer, for the denoiser and the finder, embedded with t-SNE; a 5-nearest-neighbour vote
  tells the five roles apart with balanced accuracy 31% (denoiser) and
  43% (finder); chance is 20%.
- `sensitivity.json`: defects found (%, all 750 test defects) when every atom is shifted at random by a
  Gaussian of standard deviation 0, 0.02, 0.05 and 0.1 sigma:

| Method | 0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|
| EGNN finder (main) | 54.0 | 38.1 | 18.8 | 11.3 |
| EGNN, random start (seed 0) | 43.9 | 42.4 | 34.8 | 12.7 |
| Transformer (seed 0) | 53.7 | 31.1 | 7.1 | 3.5 |
| CNN (seed 0) | 53.5 | 35.1 | 17.6 | 10.1 |
| Cell size | 26.4 | 22.8 | 15.7 | 8.9 |
| Unlike neighbours | 28.9 | 29.2 | 26.3 | 8.0 |
| Atom energy | 29.6 | 10.4 | 3.5 | 3.6 |

Every method loses most of its advantage when the atoms are shaken; the count of unlike
neighbours is the most robust up to 0.05 sigma, the atom energy the least. The finders were
trained only on perfectly relaxed glasses.
