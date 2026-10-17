# Successful-only defect analysis

Analysis completed on 18 October 2026. This report covers 12 retained runs: seven variable-radius-training evaluations, four fixed-radius-training D2 radius evaluations, and one D2 hand-insertion baseline. These are the saved inputs available through Kaggle push `9ef2df9`; subsequent notebook results are not automatically included.

The main finding is that averaging realism across successful and failed requests can conceal a larger energy-distribution mismatch among successful placements. Larger regeneration regions remain promising for D2, but window-level energy agreement does not imply natural structure in every radial shell.

## How to reproduce

From the repository root, with the project installed:

```bash
python scripts/analyze_successful_defects.py \
  --runs 'results/*/experiments/rr_*' 'results/*/experiments/seam_*' \
  --data data/ka2d_256 \
  --out runs/gap1_repeat \
  --bootstrap 1000
```

Script revision: `2e88502`. Local window radius is 3, radial shells have width 1 up to radius 7, and the natural reference cap is 512 windows. A future run of these globs may include additional Kaggle outputs. `manifest.json` records the exact input files and their SHA-256 hashes for this report.

## Results

Success is measured on all 256 requests per run after relaxation. Local energy W1 is lower when the pooled per-atom energy distribution is closer to naturally formed defects of the same class. “Successful” and “failed” refer to placement outcomes, not separate training runs. These are single-training-seed comparisons.

### D2

| Run | Successes | Success | All-sample W1 | Successful-only W1 [95% CI] | Failed-only W1 |
|---|---:|---:|---:|---:|---:|
| [seam_cond_smax1_w2_rout3_host_D2](../kaggle_20261017-1702Z/experiments/seam_cond_smax1_w2_rout3_host_D2) | 137/256 | 53.5% | 0.0728 | 0.0864 [0.0742, 0.0985] | 0.0573 |
| [seam_cond_smax1_w2_rout4_host_D2](../kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout4_host_D2) | 143/256 | 55.9% | 0.0564 | 0.0632 [0.0504, 0.0751] | 0.0483 |
| [seam_cond_smax1_w2_rout5_host_D2](../kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout5_host_D2) | 122/256 | 47.7% | 0.0257 | 0.0341 [0.0226, 0.0460] | 0.0193 |
| [seam_cond_smax1_w2_rout6_host_D2](../kaggle_20261017-1739Z/experiments/seam_cond_smax1_w2_rout6_host_D2) | 131/256 | 51.2% | 0.0112 | 0.0174 [0.0129, 0.0241] | 0.0083 |
| [seam_hand_insert_D2](../kaggle_20261017-1739Z/experiments/seam_hand_insert_D2) | 76/256 | 29.7% | 0.0866 | 0.0847 [0.0694, 0.1012] | 0.0874 |
| [rr_cond_smax1_w2_rout4_host_D2](../kaggle_20261017-2129Z/experiments/rr_cond_smax1_w2_rout4_host_D2) | 82/256 | 32.0% | 0.0408 | 0.0502 [0.0362, 0.0637] | 0.0364 |
| [rr_cond_smax1_w2_rout3_host_D2](../kaggle_20261017-2217Z/experiments/rr_cond_smax1_w2_rout3_host_D2) | 108/256 | 42.2% | 0.0641 | 0.0731 [0.0615, 0.0864] | 0.0576 |
| [rr_cond_smax1_w2_rout5_host_D2](../kaggle_20261017-2217Z/experiments/rr_cond_smax1_w2_rout5_host_D2) | 77/256 | 30.1% | 0.0156 | 0.0172 [0.0132, 0.0285] | 0.0152 |
| [rr_cond_smax1_w6_rout4_host_D2](../kaggle_20261017-2217Z/experiments/rr_cond_smax1_w6_rout4_host_D2) | 139/256 | 54.3% | 0.0509 | 0.0659 [0.0548, 0.0762] | 0.0341 |

Natural train-versus-test energy W1 floor: **0.0130**. The test reference uses 512 defect windows from 53 host glasses.

### D1_MINUS

| Run | Successes | Success | All-sample W1 | Successful-only W1 [95% CI] | Failed-only W1 |
|---|---:|---:|---:|---:|---:|
| [rr_cond_w2_rout3_host_D1_MINUS](../kaggle_20261017-2217Z/experiments/rr_cond_w2_rout3_host_D1_MINUS) | 49/256 | 19.1% | 0.0221 | 0.0378 [0.0294, 0.0555] | 0.0186 |
| [rr_cond_w2_rout4_host_D1_MINUS](../kaggle_20261017-2217Z/experiments/rr_cond_w2_rout4_host_D1_MINUS) | 56/256 | 21.9% | 0.0207 | 0.0449 [0.0290, 0.0629] | 0.0152 |
| [rr_cond_w6_rout4_host_D1_MINUS](../kaggle_20261017-2217Z/experiments/rr_cond_w6_rout4_host_D1_MINUS) | 40/256 | 15.6% | 0.0429 | 0.0646 [0.0435, 0.0880] | 0.0388 |

Natural train-versus-test energy W1 floor: **0.0160**. The test reference uses 155 defect windows from 80 host glasses.

## Interpretation

1. **Successful-only scoring changes the realism assessment.** For the new variable-radius D1− model at w=2 and radius 4, energy W1 rises from 0.0207 across all requests to 0.0449 among the 56 successful requests. Failed requests score 0.0152. Thus its apparently near-floor all-sample score is not representative of the successful placements. This report does not reanalyse the older fixed-radius D1− model, whose retained samples are not part of these inputs.
2. **The strongest retained D2 energy result remains promising.** The fixed-radius-trained model sampled at radius 6 succeeds on 131/256 requests (51.2%), with successful-only local energy W1 0.0174 [0.0129, 0.0241], compared with 0.0847 [0.0694, 0.1012] for successful hand insertions (76/256). This supports a better success/local-energy tradeoff in this tested setting, pending repeated training seeds and broader physical checks.
3. **Natural-looking pooled energy can conceal shell-level differences.** For that radius-6 model, successful placements have energy W1 0.1591 in the innermost 0–1 shell, versus a natural floor 0.0179; the 4–5 shell has W1 0.0414 versus floor 0.0075. Do not describe the entire defect environment as natural solely because the radius-3 pooled metric is near its floor.
4. **Variable-radius training has not demonstrated overall dominance.** At D2 radius 5 and w=2 it gives successful-only W1 0.0172, but only 77/256 successes (30.1%). The fixed-trained radius-6 setting gives approximately the same local energy mismatch with higher success, while editing a larger region. At radius 4, stronger guidance raises the new model’s success from 32.0% to 54.3% and increases successful-only W1 from 0.0502 to 0.0659. It is a tradeoff, not an unqualified improvement.
5. **Energy and structural metrics should remain separate.** `summary.csv` includes successful-only bond-angle and A–A, A–B, B–B distance W1. They do not uniformly favour one method: for example, successful radius-6 D2 placements improve energy W1 over hand insertion but have larger bond-angle and A–B-distance W1 point estimates. An energy advantage is not proof of overall structural equivalence.

## Outputs and validation

- [Summary table](summary.csv): 48 group rows, including all, successful, failed, and natural-floor groups.
- [Sample outcomes](sample_outcomes.csv): 3,072 requests with before/after success and convergence indicators.
- [Radial measurements](shells.csv): 336 shell rows.
- [Success versus successful-only energy realism](success_vs_realism.png): comparison figure; compare each defect class with its own references.
- [Metadata](metadata.json) and [input manifest](manifest.json).

All 12 recomputed success rates and all-sample local energy W1 point estimates match their original `metrics.json` values (tolerances 1e-12 and 1e-10 respectively). Successful plus failed counts sum to the request count in every run. All 12 runs report full relaxation convergence. Four statistical tests pass, including equivalence of the optimized bootstrap to expanded-sample NumPy quantiles; repository lint and formatting checks pass.

### Radial figure index

- [seam_cond_smax1_w2_rout3_host_D2](radial_000.png)
- [seam_cond_smax1_w2_rout4_host_D2](radial_001.png)
- [seam_cond_smax1_w2_rout5_host_D2](radial_002.png)
- [seam_cond_smax1_w2_rout6_host_D2](radial_003.png)
- [seam_hand_insert_D2](radial_004.png)
- [rr_cond_smax1_w2_rout4_host_D2](radial_005.png)
- [rr_cond_smax1_w2_rout3_host_D2](radial_006.png)
- [rr_cond_smax1_w2_rout5_host_D2](radial_007.png)
- [rr_cond_smax1_w6_rout4_host_D2](radial_008.png)
- [rr_cond_w2_rout3_host_D1_MINUS](radial_009.png)
- [rr_cond_w2_rout4_host_D1_MINUS](radial_010.png)
- [rr_cond_w6_rout4_host_D1_MINUS](radial_011.png)

## Limits and next steps

- Confidence intervals resample requests/windows with a fixed natural reference. They do not account for repeated host glasses, uncertainty in the reference pool, or variation across trained models. The natural window cap follows the repository’s ordered-site selection.
- The natural floor uses a different number of windows from the successful-only groups. Sample-count-matched natural comparisons and host-clustered uncertainty would strengthen a publication analysis.
- Successful-only groups differ in which requests they contain. These conditional comparisons describe the outputs each method succeeds on; they do not control for site difficulty. A shared-success-site comparison would be a useful additional check.
- Windows are centred on the requested site; natural windows are centred on their defect atoms. Report this convention alongside position-tolerance sensitivity.
- Pinned-atom displacement is reported where available. It is not a measurement of full original-host displacement; the saved hand-insertion run has no pinned atoms.
- No finite-temperature MD was performed, and there are no retained D1− hand-insertion or melt-quench baseline samples in this selected set. Gap 1 is therefore advanced, not fully closed.

Recommended next steps: inspect the successful-only radial and structural differences, add sample-count-matched natural comparisons and repeated-host uncertainty, repeat the promising D2 setting across training seeds, and perform matched finite-temperature defect-retention tests.
