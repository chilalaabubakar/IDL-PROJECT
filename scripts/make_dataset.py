"""Generate a glass dataset with LAMMPS.  [Ticket P-1]

    python scripts/make_dataset.py --config configs/data/ka2d_256.yaml

For each split and seed: make_glass -> passes_quality_gates (reject, log, use the next
unused seed) -> save_split to <out_dir>/<split>.npz, plus <out_dir>/meta.json (protocol,
LAMMPS version, git state, rejection log). Run glasses in parallel with a process pool.
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket P-1")
