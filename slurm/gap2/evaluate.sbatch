#!/usr/bin/env bash
#SBATCH --job-name=gap2-eval
#SBATCH --partition=GPU-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=v100-32:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
cd "${GAP2_REPO:?Export GAP2_REPO}"
source slurm/gap2/env.sh
conditions=(ka2d_1024_standard ka2d_1024_fast)
condition=${conditions[${SLURM_ARRAY_TASK_ID:?Array index required}]}
exec python -u scripts/run_gap2_psc.py all --condition "$condition" \
    --out "${GAP2_OUT:?Export a dedicated PSC result path}" --workers 8
