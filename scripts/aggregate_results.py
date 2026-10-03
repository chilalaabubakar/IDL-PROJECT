"""Collect metrics.json from runs/ into results/summary.csv and the report tables.  [Ticket B-3]

python scripts/aggregate_results.py --config configs/eval/default.yaml runs_dir=runs
"""

from glassdiff.utils.config import cli_config

if __name__ == "__main__":
    cfg = cli_config(__doc__)
    raise NotImplementedError("Ticket B-3")
