"""Relax samples with batched FIRE (all atoms free) and save displacements.  [Ticket P-3]

python scripts/relax.py --config configs/eval/default.yaml samples=runs/<run>/samples.npz
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket P-3")
