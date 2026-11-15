# Three kinds of network as finder: transformer, CNN and EGNN

Two questions from Prof. Dabo's review are answered here together with the seed study
(`../point_defects_seeds_20261114/`): is the EGNN stable in training, and would another kind
of network find more defects?

Three finders were trained in exactly the same way: from random starting weights, the same
data, loss, class weights, optimiser and batch size, 20,000 updates, seeds 0, 1 and 2
(`configs/train/detect_attention.yaml`, `detect_cnn.yaml`, `detect_egnn_scratch.yaml`). The
EGNN trained this way is the control. The runs were made on Kaggle (T4 x2) with
`notebooks/kaggle_attention_finder.ipynb` and `notebooks/kaggle_cnn_finder.ipynb`; their
weights and per-update logs are in `../kaggle_arch_20261115-*/training/<model>_s<seed>/`.
This folder is the evaluation of all of them at once, on the 750 real test defects, made on
CPU with

```bash
python scripts/pilots/finder_architectures.py evaluate \
    --out results/point_defects_architectures_20261115 --device cpu --threads 4   # ~35 min
```

It gives, for every run, exactly the found rates of the evaluation made on the GPU
(`../kaggle_arch_20261115-1254Z/`), and adds the comparisons between finders.

## Found (mean of three seeds, spread between seeds)

| Finder | Start | Updates | Vacancy | Interstitial | Antisite | Frenkel pair | All |
|---|---|---|---|---|---|---|---|
| Transformer (0.58 M parameters) | random | 20,000 | 67.2% | 48.3% | 54.2% | 42.0% | **53.6 ± 0.3%** |
| CNN (0.50 M) | random | 20,000 | 69.7% | 51.2% | 52.8% | 38.4% | **54.0 ± 0.5%** |
| EGNN, the control (0.52 M) | random | 20,000 | 63.2% | 29.7% | 44.0% | 35.1% | 43.5 ± 0.5% |
| EGNN (one run) | denoiser | 20,000 | 67.0% | 47.5% | 46.5% | 36.7% | 50.3% |
| EGNN, the finder of the report (one run) | denoiser | 60,000 | 69.5% | 50.0% | 49.0% | 45.3% | 54.0% |

- **Both find about 10 points more than the EGNN trained the same way**, in every seed
  (McNemar p < 1e-8 each: 110-122 defects found only by the new finder, 34-42 only by the
  control). Most of the lead is in interstitials (+19 to +22 points) and antisites (+9 to +10).
- **The transformer and the CNN are equal** (p = 0.49-0.91 between them, per seed).
- **They match the finder of the report** (54.0%), which starts from the denoiser and has
  three times the training. Starting from the denoiser still helps the EGNN at 20,000 updates
  (50.3% against 43.5%, p < 0.001), but the kind of network matters more.
- **The finders agree on which defects are hard.** With the same seed, 36-37% of the defects
  are found by all three and 36-38% by none. Averaging the probabilities of the six
  transformer and CNN runs finds 55.1% [51, 59], not significantly more than the best single
  run (p = 0.58); neither does averaging the three seeds of one finder (54.5%, 54.7%, 44.0%)
  or adding the finder of the report (55.2%).

## Symmetry

How often the flagged atom stays the same when the glass is changed in a way that should not
matter:

| Finder | Turned by 90 degrees | Shifted by a random vector | Atoms renumbered |
|---|---|---|---|
| Transformer | 100% | 100% | 100% |
| CNN | 100% | **88.3%** | 100% |
| EGNN | 100% | 100% | 100% |

The CNN works on pixels: turns and reflections are averaged over at test time, but a shift
that is not a whole number of its coarsest pixels (1.8σ) changes its answer a little, and in
12% of the glasses the flagged atom.

## Training stability (20,000 updates, gradient norm before clipping at 1)

| Finder | Median | Largest | Updates clipped | Gradient spikes per run | Loss spikes per run | Hours per run (T4) |
|---|---|---|---|---|---|---|
| Transformer | 0.9-1.0 | 4.9-5.5 | 40-50% | 0, 0, 0 | 0, 0, 1 | 0.58 |
| CNN | 1.7-1.8 | 6.3-8.1 | 85% | 0, 0, 0 | 0, 0, 0 | 0.62 |
| EGNN | 1.5-1.6 | 55-82 | 85-88% | 71, 131, 96 | 14, 22, 34 | 0.77 |

A gradient spike is a gradient norm above 5 times the median of the previous 100 updates; a
loss spike a loss above 3 times the median of the previous 50 updates of the same batch type.
The EGNN's first spike comes at update 7,818, 8,641 and 9,704 in the three seeds. No run had a
non-finite loss or gradient. The spikes are specific to the EGNN: the two other networks see
the same data, loss, optimiser and clipping and have none.

## Caveats

- One training length (20,000 updates), from random weights; the transformer and the CNN were
  not started from a denoiser.
- Only finding was compared. The repair (correct, then minimise) was not rerun with the new
  finders.
- The transformer has 10% more parameters than the control and a learning-rate warm-up over
  the first 1,000 updates; the CNN has neither and does equally well.
- The cause of the EGNN's gradient spikes was not identified.

## Files

- `architectures.json`: every number above, per seed (`models`), finder against finder
  (`between`), agreement (`agreement`), finders used together (`together`).
- `summary.md`: the same as tables.
- The report (`paper/point_defects_report.pdf`, Sections 5.5 and 5.6, Tables 10 and 11,
  Figure 5) reads its numbers from this folder and from `../point_defects_seeds_20261114/`.
