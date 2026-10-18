# Gap 2 host-generation feasibility benchmark

One independently generated host per condition; no trained-model evaluation has been run. These are not generalization performance estimates. Benchmark seeds are excluded from the planned evaluation draws. Wall times include generation and quality checks under concurrent CPU load.

| Atoms | Cooling rate | Seconds | Quality pass | Energy/atom |
|---:|---:|---:|---|---:|
| 256 | 0.001 | 35.7 | True | -3.723759 |
| 256 | 0.01 | 5.6 | True | -3.675483 |
| 256 | 0.1 | 6.9 | True | -3.613247 |
| 1024 | 0.001 | 156.1 | True | -3.701706 |
| 1024 | 0.01 | 42.2 | True | -3.669144 |
| 1024 | 0.1 | 11.9 | True | -3.635960 |

See `docs/GAP2_GENERALIZATION.md` and `configs/eval/gap2_generalization.json` for the fixed study design. Per-condition raw hosts, force residuals, order and pair-distance diagnostics are retained here. One host per cell cannot establish a quench-history trend or support a reliable full-study runtime estimate.
