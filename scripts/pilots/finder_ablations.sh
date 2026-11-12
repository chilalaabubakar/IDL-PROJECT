#!/usr/bin/env bash
# Ablations of the point-defect finder: one change at a time, the same small training budget.
#
#   bash scripts/pilots/finder_ablations.sh [output directory]
#   python scripts/pilots/finder_ablations.py        # test evaluation and paired tests
#
# Every variant is configs/train/detect_point_defects.yaml with one setting changed, trained
# for N_UPDATES updates (default 2,000; the finder of the report was trained for 60,000):
#
#   reference    starts from the denoiser, 4 layers, class-weighted loss
#   scratch      random initial weights instead of the denoiser
#   real_only    training without the Frenkel pairs that are the original glass again
#   layers2      2 message-passing layers (reach 5 sigma), from scratch
#   layers1      1 message-passing layer (reach 2.5 sigma), from scratch
#   no_weights   plain cross-entropy, no class weights
#
# A run that is interrupted continues from its last checkpoint when the command is given
# again. About 4 hours on 4 CPU cores (20 minutes on one T4 GPU).
set -euo pipefail
OUT=${1:-results/point_defects_ablations_20261112}
N=${N_UPDATES:-2000}
CONFIG=configs/train/detect_point_defects.yaml
# CPU only: keep freed memory in the process, which halves the time per update
export MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=100000000000 MALLOC_TOP_PAD_=1073741824
mkdir -p "$OUT/runs"

train() { # train <name> [overrides ...]
  local name=$1
  shift
  local dir
  dir=$(ls -d "$OUT"/runs/*_"$name" 2>/dev/null | tail -1 || true)
  echo "$(date -u +%H:%M:%S) $name" >> "$OUT/ablations.log"
  if [ -n "$dir" ] && [ -f "$dir/ckpt.pt" ]; then
    python scripts/train_point_defects.py --config $CONFIG resume="$dir" threads=4 \
      >> "$OUT/train_$name.log" 2>&1
  else
    if [ -n "$dir" ]; then rm -rf "$dir"; fi # started, but stopped before its first checkpoint
    # k_max 32: no atom has more than 26 neighbours, so the result is that of the default 64
    python scripts/train_point_defects.py --config $CONFIG name="$name" runs_root="$OUT/runs" \
      optim.n_updates="$N" val_every=500 ckpt_every=250 log_every=50 model.k_max=32 threads=4 \
      "$@" >> "$OUT/train_$name.log" 2>&1
  fi
}

train reference
train scratch init_from=null
train real_only real_defects_only=true
train layers2 init_from=null model.n_layers=2
train layers1 init_from=null model.n_layers=1
train no_weights class_weight_power=0
echo "$(date -u +%H:%M:%S) all runs done" >> "$OUT/ablations.log"
