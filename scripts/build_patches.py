"""Build the reference-defect patch library from the TRAIN split.  [Ticket E-3]

python scripts/build_patches.py --config configs/defects/v0.yaml data=data/ka2d_256
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket E-3")
