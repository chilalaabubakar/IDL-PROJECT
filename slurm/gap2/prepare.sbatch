#!/usr/bin/env bash
#SBATCH --job-name=gap2-hosts
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
cd "${GAP2_REPO:?Export GAP2_REPO}"
source slurm/gap2/env.sh
conditions=(ka2d_1024_standard ka2d_1024_fast)
condition=${conditions[${SLURM_ARRAY_TASK_ID:?Array index required}]}
# exec forwards scheduler signals to the supervising Python process.
exec python -u scripts/run_gap2_psc.py prepare --condition "$condition" \
    --out "${GAP2_OUT:?Export a dedicated PSC result path}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
