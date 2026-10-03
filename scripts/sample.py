"""Generate samples with a trained model and a conditioning strategy.  [Tickets S-1..S-4, B-2]

    python scripts/sample.py --config configs/sampler/repaint.yaml ckpt=runs/<run>/ema.pt defect=D2

Saves raw samples, the requests, and wall-clock timings to the run directory.
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket S-1")
