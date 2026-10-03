"""Collect metrics.json from run directories into one CSV table.  [Ticket B-3]

    python scripts/aggregate_results.py --config configs/eval/default.yaml runs_dir=runs \\
        out=results/summary.csv

One row per run: method (from the run directory name), defect, request mode, sample count,
success before/after relaxation with CIs, cost, and the local and global realism metrics.
Floor runs (natural vs natural) are included with method "floor".
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from glassdiff.utils.config import cli_config


def flatten(d: dict, prefix: str = "") -> dict:
    """Nested metrics -> flat columns; [estimate, lo, hi] triples become three columns."""
    row: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            row.update(flatten(v, key + "."))
        elif isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
            row[key], row[key + "_lo"], row[key + "_hi"] = v
        else:
            row[key] = v
    return row


def main() -> None:
    cfg = cli_config(__doc__)
    rows = []
    for path in sorted(Path(cfg.get("runs_dir", "runs")).glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        name = path.parent.name.split("_", 1)[-1]  # strip the timestamp
        row = {"run": path.parent.name, "method": "floor" if "floor" in name else name}
        row.update(flatten(metrics))
        rows.append(row)
    if not rows:
        raise SystemExit("no metrics.json found")
    columns = list(dict.fromkeys(k for r in rows for k in r))
    out = Path(cfg.get("out", "results/summary.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} runs, {len(columns)} columns -> {out}")


if __name__ == "__main__":
    main()
