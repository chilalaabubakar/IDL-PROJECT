"""Results tables and figures for the reports.  [Ticket E-5]

Reads results/summary.csv (scripts/aggregate_results.py) and produces the report tables:
one row per (task, method, defect class, model), each with estimate [lower, upper], plus
the floor (natural vs natural) and ceiling (hand insertion before relaxation) rows.
"""

from __future__ import annotations

from pathlib import Path


def results_table(summary_csv: str | Path, metric: str) -> str:
    """Markdown table of one metric across methods and defect classes."""
    raise NotImplementedError("Ticket E-5")
