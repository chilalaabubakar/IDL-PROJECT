"""Rank silica training runs for the screening funnel (configs/silica/screen.yaml).

    python scripts/silica/rank_runs.py runs/*silica_cond_v* [--top 6] [--out top6.txt]

Reads the last row of each run's log.csv and sorts by the funnel's ``rank_by`` column
(validation loss of the EMA model as a fraction of the injected noise; lower is better).
Prints a table with the variant's levels, steps and seconds per update; with ``--out``
writes the top run directories, one per line, for the next stage:

    xargs -I{} python scripts/silica/train.py --config configs/silica/train_cond.yaml \\
        resume={} n_updates=15000 < top6.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from glassdiff.utils.config import load_config


def last_row(log: Path) -> dict | None:
    with log.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--screen", default="configs/silica/screen.yaml")
    ap.add_argument("--by", default=None, help="log.csv column (default: screen.yaml rank_by)")
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the top run directories here")
    args = ap.parse_args()
    by = args.by or load_config(args.screen).get("rank_by", "valn_0.2")

    rows = []
    for run in map(Path, args.runs):
        log = run / "log.csv"
        if not log.exists():
            continue
        row = last_row(log)
        if row is None or row.get(by) in (None, ""):
            continue
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        levels = " ".join(cfg.get("variant_levels", [])) or "-"
        rows.append((float(row[by]), int(row["step"]), float(row["s_per_update"]), levels, run))
    rows.sort(key=lambda r: r[0])
    print(f"{'rank':>4} {by:>10} {'step':>7} {'s/upd':>6}  levels  run")
    for i, (score, step, spu, levels, run) in enumerate(rows, 1):
        print(f"{i:>4} {score:>10.4f} {step:>7} {spu:>6.2f}  {levels}  {run}")
    if args.out:
        top = rows[: args.top] if args.top else rows
        Path(args.out).write_text("".join(f"{r[4]}\n" for r in top))
        print(f"top {len(top)} -> {args.out}")


if __name__ == "__main__":
    main()
