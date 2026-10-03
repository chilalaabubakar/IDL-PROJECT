"""Plot a training run's validation curves (val loss / sigma^2 per noise level).

python scripts/plot_training.py --config configs/eval/default.yaml run=runs/<train-run>
"""

from __future__ import annotations

from pathlib import Path

from glassdiff.utils.config import cli_config
from glassdiff.viz import plot_training


def main() -> None:
    cfg = cli_config(__doc__)
    run = Path(cfg["run"])
    print(f"wrote {plot_training(run / 'log.csv', cfg.get('out') or run / 'training.png')}")


if __name__ == "__main__":
    main()
