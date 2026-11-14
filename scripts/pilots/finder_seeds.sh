#!/usr/bin/env bash
# Seed study of the point-defect finder: the 2,000-update runs of the ablation table
# (scripts/pilots/finder_ablations.sh) again with seeds 1 and 2, every update logged.
#
#   bash scripts/pilots/finder_seeds.sh [output directory]
#   python scripts/pilots/finder_seeds.py        # test evaluation, spread and stability
#
# Seed 0 of every variant is the ablation run itself (results/point_defects_ablations_20261112).
# Those runs have no per-update log, so the current finder (reference) is also run again with
# seed 0, and so, last, is the run from random starting weights (scratch). Training on CPU is
# reproducible, so these reruns should give the seed-0 weights exactly, now with the log.
#
# What a seed changes: the order of the training batches and the starting weights of
# everything that is not taken from the denoiser. For the runs that start from the denoiser
# that is only the output layer; for scratch and the runs with fewer layers it is all of it.
#
# Every update writes one row to steps.csv in its run folder (scripts/train_point_defects.py):
# the loss and the gradient norm before clipping. Each run is the same as in the ablation
# table apart from the seed. A run that is interrupted continues from its last checkpoint
# when the command is given again. About 10 hours on 4 CPU cores in all.
set -euo pipefail
OUT=${1:-results/point_defects_seeds_20261114}
N=${N_UPDATES:-2000}
CONFIG=configs/train/detect_point_defects.yaml
# CPU only: keep freed memory in the process, which halves the time per update
export MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=100000000000 MALLOC_TOP_PAD_=1073741824
mkdir -p "$OUT/runs"

train() { # train <variant> <seed> [overrides ...]
  local name="$1_s$2" seed=$2
  shift 2
  local dir
  dir=$(ls -d "$OUT"/runs/*_"$name" 2>/dev/null | tail -1 || true)
  if [ -n "$dir" ] && [ -f "$dir/ckpt.pt" ]; then
    if python -c "import sys, torch; c = torch.load(sys.argv[1], map_location='cpu', weights_only=False); sys.exit(c['step'] < $N)" "$dir/ckpt.pt"; then
      return 0 # finished already
    fi
    echo "$(date -u +%H:%M:%S) $name (resumed)" >> "$OUT/seeds.log"
    python scripts/train_point_defects.py --config $CONFIG resume="$dir" threads=4 \
      >> "$OUT/train_$name.log" 2>&1
  else
    if [ -n "$dir" ]; then rm -rf "$dir"; fi # started, but stopped before its first checkpoint
    echo "$(date -u +%H:%M:%S) $name" >> "$OUT/seeds.log"
    # k_max 32: no atom has more than 26 neighbours, so the result is that of the default 64
    python scripts/train_point_defects.py --config $CONFIG name="$name" runs_root="$OUT/runs" \
      seed="$seed" optim.n_updates="$N" val_every=500 ckpt_every=250 log_every=50 \
      model.k_max=32 threads=4 log_steps=true "$@" >> "$OUT/train_$name.log" 2>&1
  fi
}

# in order of priority: the current finder first
for seed in 1 2; do train reference $seed; done
train reference 0
for seed in 1 2; do train scratch $seed init_from=null; done
for seed in 1 2; do train no_weights $seed class_weight_power=0; done
for seed in 1 2; do train real_only $seed real_defects_only=true; done
for seed in 1 2; do train layers2 $seed init_from=null model.n_layers=2; done
for seed in 1 2; do train layers1 $seed init_from=null model.n_layers=1; done
train scratch 0 init_from=null
echo "$(date -u +%H:%M:%S) all runs done" >> "$OUT/seeds.log"
