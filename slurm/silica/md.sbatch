#!/bin/bash
# Make our 576-atom glasses on a Bridges-2 RM-shared allocation (one LAMMPS per core).
# Re-submit until it reports every split assembled: finished glasses are never redone.
#   sbatch -A <allocation> slurm/silica/md.sbatch
#   sbatch -A <allocation> --export=ALL,ARGS="only=test" slurm/silica/md.sbatch
# Check the potential first (10 min):  python scripts/silica/check_potential.py --config configs/silica/md.yaml
#SBATCH -J silica-md
#SBATCH -p RM-shared
#SBATCH --ntasks-per-node=64
#SBATCH -t 48:00:00
#SBATCH -o slurm/logs/%x-%j.out
ARGS=${ARGS:-}
source slurm/silica/env.sh
export OMP_NUM_THREADS=1
mkdir -p slurm/logs
# ~565 core-hours for the 1,000 glasses (0.24 h each at 100 K/ps, 0.58 h at 10, 4.0 h at 1):
# about 9-10 h on 64 cores. A glass cut off by the time limit is simply redone next time.
python scripts/silica/make_glasses.py --config configs/silica/md.yaml workers=64 $ARGS
