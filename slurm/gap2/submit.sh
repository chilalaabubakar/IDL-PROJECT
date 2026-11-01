#!/usr/bin/env bash
# Run on a Bridges-2 login node: bash slurm/gap2/submit.sh ALLOCATION
set -euo pipefail
account=${1:?Usage: bash slurm/gap2/submit.sh ALLOCATION}
: "${GAP2_REPO:?Export an absolute checkout path}"
: "${GAP2_ENV:?Export an absolute Python environment path}"
: "${GAP2_OUT:?Export a dedicated PSC result path}"
cd "$GAP2_REPO"
mkdir -p slurm/logs "$GAP2_OUT"
dependency=()
if [[ -n "${GAP2_CPU_ACCOUNT:-}" ]]; then
    prep_id=$(sbatch --parsable -A "$GAP2_CPU_ACCOUNT" --export=ALL slurm/gap2/prepare.sbatch)
    prep_id=${prep_id%%;*}
    dependency=("--dependency=afterok:$prep_id")
    printf 'Optional CPU preparation array: %s\n' "$prep_id"
fi
eval_id=$(sbatch --parsable -A "$account" --export=ALL "${dependency[@]}" slurm/gap2/evaluate.sbatch)
printf 'GPU evaluation array (both conditions): %s\n' "$eval_id"
