#!/usr/bin/env bash
# Source for jobs; execute with "install" only for a new dedicated environment.
set -eo pipefail
: "${GAP2_REPO:?Set GAP2_REPO to an absolute checkout path on Bridges-2}"
: "${GAP2_ENV:?Set GAP2_ENV to an absolute Python environment path}"
cd "$GAP2_REPO"
if [[ "${1:-}" == install ]]; then
    if [[ -x "$GAP2_ENV/bin/python" ]]; then
        echo 'Environment already exists. Use it, or choose a new GAP2_ENV path for installation.'
        exit 1
    fi
    module load anaconda3
    eval "$(conda shell.bash hook)"
    conda create -y -p "$GAP2_ENV" python=3.11
    conda activate "$GAP2_ENV"
    # CUDA 12.1 / PyTorch 2.5.1 retains support for Bridges-2 V100 (sm_70).
    python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -e '.[md,dev]'
    exit 0
fi
if [[ -f "$GAP2_ENV/bin/activate" ]]; then
    source "$GAP2_ENV/bin/activate"
else
    module load anaconda3
    eval "$(conda shell.bash hook)"
    conda activate "$GAP2_ENV"
fi
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Locate the wheel's MPI library, which need not live in sys.prefix/lib.
GAP2_MPI_DIR=$(python - <<'PY'
from importlib.metadata import distribution
p = distribution('mpich')
folders = sorted({str(p.locate_file(f).resolve().parent) for f in p.files or []
                  if f.name == 'libmpi.so.12' and p.locate_file(f).exists()})
if not folders:
    raise SystemExit('mpich did not install libmpi.so.12; install .[md] in GAP2_ENV')
print(':'.join(folders))
PY
)
export LD_LIBRARY_PATH="$GAP2_MPI_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
