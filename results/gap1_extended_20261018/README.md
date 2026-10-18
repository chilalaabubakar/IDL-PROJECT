# Extended D2 physical credibility study

Completed 1,344 freely evolving whole-box trajectories: 56 distinct hosts per group, two temperatures and three velocity seeds. Cohort A repeats the original 32 hosts at a longer horizon; cohort B adds 24 hosts disjoint from A within each method. The new endpoint, time 250, is five times the earlier time 50.

All 6,720 independently minimized snapshots meet the 1e-6 residual-force limit. Probe minimizations do not change MD trajectories.

## Endpoint counts

Counts below are trajectory occupancies, not independent sample counts. Intervals in `extended_retention.csv` resample the 56 hosts; separate Wilson intervals describe hosts retaining the label in all three velocities. `cohort_retention.csv` permits a disjoint-cohort check.

| Group | T | Same initial atom D2 | Requested-site D2 |
|---|---:|---:|---:|
| fixed_r6 | 0.01 | 146/168 | 143/168 |
| fixed_r6 | 0.1 | 125/168 | 115/168 |
| hand_insert | 0.01 | 153/168 | 153/168 |
| hand_insert | 0.1 | 121/168 | 118/168 |
| natural | 0.01 | 160/168 | 160/168 |
| natural | 0.1 | 145/168 | 145/168 |
| unedited_host | 0.01 | 6/168 | 6/168 |
| unedited_host | 0.1 | 6/168 | 6/168 |

These are pointwise, frozen-threshold occupancy measurements. Loss followed by recovery is possible; this is not a continuous survival curve, an energy-barrier estimate or proof of natural structural equivalence. Same-atom and requested-site predicates answer different questions. The unedited-host group estimates background appearance.

This successful-only study does not estimate successful creation and later retention per request. The full 256-request-per-model Kaggle workflow evaluates that outcome, with initial failures counting as zero. Do not mark that evaluation complete without its final results. D1-minus and other model/sampling settings remain outside this thermal study.
