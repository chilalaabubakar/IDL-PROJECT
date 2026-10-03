"""Compute all metrics with bootstrap CIs for one sampling run.  [Ticket E-4]

    python scripts/evaluate.py --config configs/eval/default.yaml run=runs/<run>

Writes <run>/metrics.json (one entry per metric: estimate, lower, upper, n_samples).
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket E-4")
