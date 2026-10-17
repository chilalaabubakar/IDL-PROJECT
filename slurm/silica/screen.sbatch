#!/bin/bash
# Screening funnel, stage 1: the 64 variants of configs/silica/screen.yaml, 5,000 updates each
# (~133 epochs of the 600-glass set). Re-submitting continues unfinished variants.
#   sbatch -A <allocation> slurm/silica/screen.sbatch
#   python scripts/silica/rank_runs.py runs/silica_screen/v* --top 6 --out runs/silica_screen/top6.txt
# Stage 2 and 3 (extend the best): N=15000 then N=50000 with the list of survivors:
#   sbatch -A <allocation> --array=0-5 --export=ALL,LIST=runs/silica_screen/top6.txt,N=15000 slurm/silica/screen.sbatch
#SBATCH -J silica-screen
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 12:00:00
#SBATCH --array=0-63%8
#SBATCH -o slurm/logs/%x-%A_%a.out
CONFIG=${CONFIG:-configs/silica/train_cond.yaml}
N=${N:-5000}
ARGS=${ARGS:-}
source slurm/silica/env.sh
mkdir -p slurm/logs
K=$SLURM_ARRAY_TASK_ID
if [[ -n "${LIST:-}" ]]; then
  RUN=$(sed -n "$((K + 1))p" "$LIST")
  python scripts/silica/train.py --config "$CONFIG" resume="$RUN" n_updates="$N" \
    max_minutes=700 threads=5
else
  RUN=runs/silica_screen/v$(printf %02d "$K")
  python scripts/silica/train.py --config "$CONFIG" variant="$K" run_dir="$RUN" \
    n_updates="$N" max_minutes=700 threads=5 $ARGS
fi
