# Seed study of the point-defect finder

**Status, 14 November 2026: running** (`scripts/pilots/finder_seeds.sh`, about 10 hours on
4 CPU cores). Results and their write-up will be added here.

The question, raised by Prof. Dabo: is the EGNN finder stable? Two meanings are tested.

1. **Does training misbehave?** Every update is logged (`steps.csv` in each run): the loss
   and the gradient norm *before* clipping (training clips at 1.0, which would otherwise hide
   a large gradient). Spikes are judged against recent updates of the same defect type,
   because each batch holds one defect type and one batch in five holds no defect, so the
   loss jumps between batches by design.
2. **Do the results depend on the seed?** The six 2,000-update runs of the ablation table
   (`../point_defects_ablations_20261112/`, seed 0) are repeated with seeds 1 and 2, and
   all are scored on the same 750 real test defects. This shows whether the differences in
   the ablation table are larger than the spread between seeds.

The current finder (reference) is also rerun with seed 0, so that all three of its seeds
have the per-update log. Training on CPU is reproducible (two 30-update runs with the same
seed gave identical weights, one with and one without the log), so the rerun should match
the published seed-0 run exactly.
