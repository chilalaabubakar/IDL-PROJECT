# Run the successful-defect analysis

Run from the IDL-PROJECT checkout root with the project dependencies installed:

```bash
python scripts/analyze_successful_defects.py \
  --runs 'results/*/experiments/rr_*' 'results/*/experiments/seam_*' \
  --data data/ka2d_256 \
  --out runs/gap1 \
  --bootstrap 1000
```

On Kaggle, first change into `/kaggle/working/IDL-PROJECT`, then run the command above. The shell quotes around patterns are intentional. Add other run-directory patterns to `--runs` for D1− or older baselines. Each run needs `samples.npz` and `relaxed.npz`; incomplete or metrics-only folders are reported and skipped. Analyse one dataset per invocation. Use a new output directory for each analysis you want to retain.

For a quick check, use `--bootstrap 100 --reference-cap 64`; for final analysis retain the default 512 natural windows and use at least 1000 bootstrap resamples. Quick-check reference caps change the estimates, not only runtime. This is CPU analysis; it does not train or run MD, and does not change input run files.

Outputs:
- `summary.csv`: group counts, all-request relaxed success with CIs, successful-only / failed / all-sample structural W1 metrics, and natural train-vs-reference floor.
- `sample_outcomes.csv`: pre/post success and convergence per sample.
- `shells.csv`: radial energy W1 with CIs for each group.
- `success_vs_realism.png`: all-request success versus successful-only local energy W1.
- `radial_*.png`: radial group comparisons, indexed in sorted run-path order.
- `metadata.json`: arguments, uncertainty scope, threshold provenance, and skipped folders.

Interpretation:
- Small groups (<30 samples/windows) are flagged; a one-sample group has no CI.
- CIs resample requests or natural windows with a fixed natural reference, not independent training seeds or host-glass clusters. Natural windows may share glasses. Counts of reference glasses are reported.
- Natural windows follow the repository's ordered-site selection, capped at `--reference-cap`; examine sensitivity to that cap before publication.
- Radii are measured from the requested target, matching the existing evaluation. A natural window is centred on its labelled defect atom.
- `pinned_displacement_mean` measures final movement of saved pinned atoms relative to `pin_pos`. It is not full original-host displacement; baseline files with no pinned atoms have no value.
- Convergence is reported but not used to silently exclude samples.
- The script does not evaluate finite-temperature stability or replace host-clustered / across-seed uncertainty analysis.
