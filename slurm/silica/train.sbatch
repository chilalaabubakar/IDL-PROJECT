#!/bin/bash
# One training run on a Bridges-2 V100-32GB (GPU-shared); resubmits itself until done.
#   sbatch -A <allocation> slurm/silica/train.sbatch                        # conditional model
#   sbatch -A <allocation> --export=ALL,CONFIG=configs/silica/train_uncond.yaml,RUN=runs/silica/uncond_s0 slurm/silica/train.sbatch
#   sbatch -A <allocation> --export=ALL,RUN=runs/silica/cond_s1,ARGS="seed=1" slurm/silica/train.sbatch
#SBATCH -J silica-train
#SBATCH -p GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --ntasks-per-node=5
#SBATCH -t 48:00:00
#SBATCH -o slurm/logs/%x-%j.out
CONFIG=${CONFIG:-configs/silica/train_cond.yaml}
RUN=${RUN:-runs/silica/cond_s0}
ARGS=${ARGS:-}
source slurm/silica/env.sh
mkdir -p slurm/logs
# stop 20 minutes before the limit so the last checkpoint is written in time
python scripts/silica/train.py --config "$CONFIG" run_dir="$RUN" max_minutes=2860 threads=5 $ARGS \
  || exit 1
# not finished yet -> queue the next 48 h (the same command continues the run)
if python - "$RUN" <<'PY'
import sys, torch
c = torch.load(f"{sys.argv[1]}/ckpt.pt", map_location="cpu", weights_only=False)
sys.exit(0 if c["step"] < c["cfg"]["optim"]["n_updates"] else 1)
PY
then
  sbatch --export=ALL,CONFIG="$CONFIG",RUN="$RUN",ARGS="$ARGS" slurm/silica/train.sbatch
fi
