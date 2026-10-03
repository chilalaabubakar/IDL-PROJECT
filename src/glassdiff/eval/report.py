"""Results tables for the reports.  [Ticket E-5]

Reads results/summary.csv (scripts/aggregate_results.py) and renders markdown tables:
one row per run (method, defect, mode, n), each metric as "estimate [lower, upper]".
Floor rows (natural vs natural) are listed first so every number has its reference.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

DEFAULT_METRICS = (
    "success_after_relax",
    "success_before_relax",
    "local.w1_pe_atom",
    "local.w1_bond_angle",
    "local.overlap_rate",
    "seconds_per_success",
)


def _fmt(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value in ("", None):
        return "—"
    v = float(value)
    if math.isinf(v):
        return "∞"
    lo, hi = row.get(f"{key}_lo"), row.get(f"{key}_hi")
    digits = 3 if abs(v) < 10 else 1
    if lo not in (None, "") and hi not in (None, ""):
        return f"{v:.{digits}f} [{float(lo):.{digits}f}, {float(hi):.{digits}f}]"
    return f"{v:.{digits}f}"


def results_table(summary_csv: str | Path, metrics: tuple[str, ...] = DEFAULT_METRICS) -> str:
    """Markdown table of the given metrics across all runs in the summary."""
    with open(summary_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: (r.get("method") != "floor", r.get("defect", ""), r.get("method", "")))
    header = ["method", "defect", "mode", "n", *metrics]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        n = r.get("n_samples") or r.get("n_windows") or ""
        cells = [r.get("method", ""), r.get("defect", ""), r.get("mode", "") or "—", n]
        cells += [_fmt(r, m) for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
