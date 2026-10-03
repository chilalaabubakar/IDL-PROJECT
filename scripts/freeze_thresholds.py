"""Fit/check defect thresholds on the TRAIN split, freeze them, and label every split.  [Ticket E-2]

    python scripts/freeze_thresholds.py --config configs/defects/v0.yaml data=data/ka2d_256

Writes thresholds (+ their hash) and per-split defect rates into <data>/meta.json and the
``labels`` array into each split. Refuses to overwrite frozen thresholds unless force=true.
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket E-2")
