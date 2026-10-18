# Gap 1 completion workflow

The defensible current claim is controlled geometric defect placement with reduced local-energy mismatch in several tested settings. Successful placements remain structurally distinguishable from natural defects; measured host displacement and thermal retention are separate limitations. The manuscript now states these limits. No retraining or structural-equivalence claim is implied.

## Full matched GPU evaluation

Import `notebooks/kaggle_gap1_completion.ipynb` into Kaggle. Select GPU T4 ×2 (one GPU also works), enable Internet, and grant access to the existing `GITHUB_TOKEN` secret with write access to this repository. Run All. The notebook checks GitHub write access and LAMMPS initialization before starting generation.

The notebook runs all 256 retained requests for each of seed0, seed1, seed2 and a null-label seed0 sham, at guidance 2 and radius 6. Hosts, sites, pinning and initialization rule are matched. Checkpoint weights are hashed; batch size 64 and the CUDA device setting are part of the resume configuration. The GPU batch noise assignment differs from the earlier CPU batch-size-2 pilot, so numerical pilot/full results need not coincide. The four full runs share the same full-run noise rule.

After generation, successful placements undergo freely evolving NVT MD to time 250 at T=0.01 with three velocity seeds. Failed placements count as zero in the joint creation-and-retention yield over **all 256 requests**. A second denominator excludes sites already D2 in the unedited host. The retained 256-request insertion baseline is included. The requested-site detector is the original predicate: the nearest B atom within 0.5 must carry the frozen D2 label. This differs from following the initially targeted atom regardless of its current location.

Valid completed batches and trajectories are pushed every five minutes to `results/kaggle_gap1_full_<UTC timestamp>/`. `progress.json` says `complete` only after generation and joint MD both finish; `running` or `incomplete` means partial evidence. Re-running the notebook restores the most recent incomplete run from GitHub. It stops at an 11-hour budget and saves available checkpoints. GPU runtime and peak memory remain unbenchmarked until the first Kaggle execution. Logs remain in `/kaggle/working/gap1-completion/`; save Kaggle output files if diagnosing a failure.

`joint_yield.csv` reports request-weighted estimates with whole-host bootstrap intervals; velocity realizations are averaged within requests. A zero-event bootstrap interval can be degenerate and is not proof of zero probability. Pointwise occupancy is not a continuous survival curve or an energy-barrier measurement.

## Longer local stability evaluation

```bash
export LD_LIBRARY_PATH="$PWD/.venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv/bin/python scripts/complete_gap1.py local \
  --out results/gap1_extended_20261018 --workers 4
```

The old 32-host cohort and 24 additional distinct hosts per method are followed to time 250 (five times the earlier endpoint). The additional cohort excludes each method's old hosts; this is not a new independently generated dataset. Four groups, two temperatures and three velocities give 1,344 trajectories, each with five independently minimized observations. The unedited-host group is paired with the generated group's hosts/sites. Both same-atom D2 occupancy and the requested-site predicate are reported. Cohort files and per-trajectory observations permit separate replication checks. Scripts resume only when protocol and input fingerprints match. All atoms evolve freely; probing quenches never alter the MD trajectory.

`extended_retention.csv` summarizes 56 distinct hosts per method. Intervals resample hosts, and a separate Wilson interval covers the proportion of hosts retaining D2 in all three velocity realizations. Report the estimand with its interval: a host-level interval is not an interval treating 168 trajectories as independent.

## What remains before declaring closure

- Inspect and interpret the **completed**, rather than partial, full Kaggle runs and joint outcomes.
- Inspect the extended local results and the additional cohort separately; do not infer natural equivalence from persistence alone.
- Retain the structural mismatch and locality caveats. This thermal evaluation covers D2; D1-minus and other generation settings need their own evidence before extending the claim.

The workflow can strengthen or qualify physical credibility. It cannot guarantee that the model passes the proposed checks.
