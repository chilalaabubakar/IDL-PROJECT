# Shared setup for the Bridges-2 jobs (source it; edit the two paths once).
#   one-time install:  bash slurm/silica/env.sh install
set -eo pipefail
REPO=${REPO:-$PROJECT/IDL-PROJECT}         # clone of this repository on $PROJECT
ENV=${ENV:-$PROJECT/envs/glassdiff}        # conda environment
module load anaconda3
eval "$(conda shell.bash hook)"
cd "$REPO"
if [[ "${1:-}" == "install" ]]; then
  conda create -y -p "$ENV" python=3.11
  conda activate "$ENV"
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -e ".[silica,md,dev]"
  python -c "import glassdiff.silica.model as m; print(sum(p.numel() for p in m.ConditionalNequIP().parameters()), 'parameters')"
  exit 0
fi
conda activate "$ENV"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS_PER_NODE:-1}}
