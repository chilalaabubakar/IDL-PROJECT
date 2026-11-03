# Gap 2: placement on new host glasses (4 of 6 conditions)

Analysis of the four completed Gap 2 conditions, made with `scripts/analyze_gap2_shift.py` on
2026-11-03 from `results/kaggle_gap2_kaggle_remaining_20261101-220126Z/study` (Kaggle task
`kaggle_remaining`, complete 2026-11-03 01:04 UTC). Supersedes `../gap2_shift_interim_20261102/`.

Same frozen models (seeds 0-2), thresholds and sampling setting (w = 2, R_out = 6) everywhere;
100 new host glasses and 256 matched requests per condition. Primary denominator: requests whose
site was not already D2. Intervals: 95%, whole-host bootstrap (1,000 draws); conditions have
independent hosts and are resampled separately.

**Data check.** In every condition, all five methods have 256 outputs on the same hosts and
sites; all are finite and relaxed to the force tolerance (1e-6); no host glass was rejected.

Still missing: 1,024 atoms at standard and at fast cooling (assigned to Colab,
`docs/GAP2_COLAB.md`).

## The host glasses differ as intended

| Per host | 256 standard | 256 slow | 256 fast | 1,024 slow |
|---|---|---|---|---|
| Cooling rate | 0.01 | 0.001 | 0.1 | 0.001 |
| Energy per atom | -3.681 | -3.707 | -3.639 | -3.702 |
| Natural D2 per 256 atoms | 10.1 [9.4, 10.8] | 7.9 [7.3, 8.6] | 12.4 [11.7, 13.1] | 8.2 [7.9, 8.5] |
| Natural D1- per 256 atoms | 1.6 [1.3, 1.9] | 1.0 [0.8, 1.2] | 2.6 [2.2, 2.9] | 1.1 [1.0, 1.2] |

Slower cooling gives lower energy and fewer natural defects, faster cooling the reverse. The
1,024-atom slow glasses match the 256-atom slow glasses, so size changes the box, not the glass.
(`hosts.csv` also lists D1+ and global hexatic order; the latter shrinks with N and is not
comparable across sizes.)

## Placement success (initially non-D2 requests)

| | 256 standard | 256 slow | 256 fast | 1,024 slow |
|---|---|---|---|---|
| Seed 0 | 0.508 | 0.549 | 0.545 | 0.508 |
| Seed 1 | 0.361 | 0.392 | 0.400 | 0.320 |
| Seed 2 | 0.424 | 0.404 | 0.413 | 0.389 |
| **Mean of seeds** | **0.431** [0.397, 0.465] | **0.448** [0.404, 0.491] | **0.453** [0.420, 0.485] | **0.406** [0.370, 0.444] |
| Hand insertion | 0.206 [0.156, 0.257] | 0.234 [0.181, 0.291] | 0.221 [0.165, 0.281] | 0.279 [0.223, 0.333] |
| Label removed (sham) | 0.097 | 0.081 | 0.115 | 0.078 |
| **Mean advantage over hand insertion** | **+0.226** [0.165, 0.292] | **+0.214** [0.146, 0.282] | **+0.231** [0.165, 0.289] | **+0.127** [0.061, 0.192] |

Change from the 256-standard control (`shift.csv`):

| Change | 256 slow | 256 fast | 1,024 slow |
|---|---|---|---|
| Mean of seeds | +0.017 [-0.038, +0.072] | +0.021 [-0.024, +0.070] | -0.026 [-0.074, +0.026] |
| Hand insertion | +0.028 [-0.053, +0.106] | +0.015 [-0.062, +0.099] | +0.073 [-0.001, +0.152] |
| Mean advantage | -0.011 [-0.108, +0.085] | +0.006 [-0.088, +0.099] | -0.098 [-0.196, -0.009] |

Size alone, at the same slow cooling (`size_at_slow_cooling/`, reference 256 slow):

| Change from 256 slow to 1,024 slow | |
|---|---|
| Mean of seeds | -0.043 [-0.099, +0.018] |
| Hand insertion | +0.045 [-0.036, +0.118] |
| Mean advantage | -0.087 [-0.184, +0.016] |

## What this says

1. **Cooling history does not change placement.** With glasses cooled 10 times more slowly or
   10 times faster than the training glasses, every seed's success and the advantage over hand
   insertion stay within a few points of the control (seed-mean changes +0.02, intervals
   within about ±0.07). This holds although the hosts differ clearly in energy and in how many
   defects they contain naturally.
2. **The main result transfers.** On 1,024-atom slow glasses the seed mean is 0.41, against
   0.43 on the control (change -0.03 [-0.07, +0.03]), with all three seeds far above the
   label-removed model.
3. **The margin over hand insertion narrows only in the larger glass**, from about +0.22 to
   +0.13: the model loses a little and hand insertion gains a little. Cooling history alone
   does not do this (256 slow keeps +0.21), so the narrowing points to system size; on its own
   the size effect at slow cooling (-0.09 [-0.18, +0.02]) is not yet significant. Seed 1's
   advantage at 1,024 slow is +0.04 [-0.04, +0.12]; seeds 0 and 2 stay clearly ahead.
4. **The two missing conditions settle point 3.** 1,024 atoms at standard cooling tests size at
   the training cooling rate; 1,024 fast completes the size x history grid.

Placement only: structure of successful placements, host displacement and survival in MD on
these conditions are not covered here.

Reproduce:

```bash
python scripts/analyze_gap2_shift.py --study results/kaggle_gap2_kaggle_remaining_20261101-220126Z/study \
    --out results/gap2_shift_20261103
python scripts/analyze_gap2_shift.py --study results/kaggle_gap2_kaggle_remaining_20261101-220126Z/study \
    --out results/gap2_shift_20261103/size_at_slow_cooling --reference ka2d_256_slow
```
