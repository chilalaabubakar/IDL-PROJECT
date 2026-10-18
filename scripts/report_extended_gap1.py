"""Validate and summarize a complete 1,344-trajectory extended D2 stability study."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glassdiff-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPEC = importlib.util.spec_from_file_location(
    "physical", Path(__file__).with_name("check_physical_credibility.py")
)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
GROUPS = ["fixed_r6", "hand_insert", "natural", "unedited_host"]
COLORS = ["#2766a8", "#c44e52", "#339966", "#888888"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = args.out
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["n_trajectories"] == 1344
    assert protocol["all_snapshot_residuals_within_tolerance"]
    rows = []
    raw = []
    for cohort, n in [("A", 32), ("B", 24)]:
        files = sorted((out / cohort / "md_trajectories").glob("*.json"))
        assert len(files) == n * 4 * 2 * 3, "Incomplete cohort"
        data = [json.loads(f.read_text()) for f in files]
        raw.extend(data)
        for group in GROUPS:
            for temperature in [0.01, 0.1]:
                selected = [
                    r for r in data if r["group"] == group and r["temperature"] == temperature
                ]
                hosts = sorted({r["host"] for r in selected})
                assert len(hosts) == n
                assert all(sum(r["host"] == h for r in selected) == 3 for h in hosts)
                for k, step in enumerate(protocol["options"]["steps"]):
                    for metric in ["quenched_same_atom", "quenched_requested_site"]:
                        means = [
                            np.mean(
                                [r["observations"][k][metric] for r in selected if r["host"] == h]
                            )
                            for h in hosts
                        ]
                        estimate = p.analysis.ci(means, np.mean, 1000, 91)
                        rows.append(
                            dict(
                                cohort=cohort,
                                group=group,
                                temperature=temperature,
                                time=step * 0.005,
                                metric=metric,
                                n_hosts=n,
                                retained_count=int(round(sum(means) * 3)),
                                n_trajectories=n * 3,
                                fraction=estimate[0],
                                lo=estimate[1],
                                hi=estimate[2],
                            )
                        )
    observations = [o for r in raw for o in r["observations"]]
    assert len(observations) == 6720
    assert max(o["max_force"] for o in observations) <= 1e-6
    p.write_csv(out / "cohort_retention.csv", rows)
    with (out / "extended_retention.csv").open() as f:
        pooled = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for column, metric in enumerate(["quenched_same_atom", "quenched_requested_site"]):
        for row, temperature in enumerate([0.01, 0.1]):
            ax = axes[row, column]
            for group, color in zip(GROUPS, COLORS):
                selected = [
                    r
                    for r in pooled
                    if r["group"] == group
                    and r["metric"] == metric
                    and float(r["temperature"]) == temperature
                ]
                times = np.array([float(r["time"]) for r in selected])
                ax.plot(
                    times, [float(r["fraction"]) for r in selected], "o-", color=color, label=group
                )
                ax.fill_between(
                    times,
                    [float(r["lo"]) for r in selected],
                    [float(r["hi"]) for r in selected],
                    color=color,
                    alpha=0.12,
                )
            ax.set(
                title=f"{metric.replace('quenched_', '').replace('_', ' ')}, T={temperature}",
                ylim=(-0.03, 1.03),
            )
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel("Pointwise D2 fraction")
            if row == 1:
                ax.set_xlabel("Time (LJ units)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("56 distinct hosts per group; three velocities per host; host-bootstrap intervals")
    fig.tight_layout()
    fig.savefig(out / "extended_retention.png", dpi=200)
    fig.savefig(out / "extended_retention.pdf")
    plt.close(fig)
    text = [
        "# Extended D2 physical credibility study\n",
        "Completed 1,344 freely evolving whole-box trajectories: 56 distinct hosts per group, two temperatures and three velocity seeds. Cohort A repeats the original 32 hosts at a longer horizon; cohort B adds 24 hosts disjoint from A within each method. The new endpoint, time 250, is five times the earlier time 50.\n",
        "All 6,720 independently minimized snapshots meet the 1e-6 residual-force limit. Probe minimizations do not change MD trajectories.\n",
        "## Endpoint counts\n",
        "Counts below are trajectory occupancies, not independent sample counts. Intervals in `extended_retention.csv` resample the 56 hosts; separate Wilson intervals describe hosts retaining the label in all three velocities. `cohort_retention.csv` permits a disjoint-cohort check.\n",
        "| Group | T | Same initial atom D2 | Requested-site D2 |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        for temperature in [0.01, 0.1]:
            values = {}
            for metric in ["quenched_same_atom", "quenched_requested_site"]:
                r = next(
                    r
                    for r in pooled
                    if r["group"] == group
                    and float(r["temperature"]) == temperature
                    and float(r["time"]) == 250
                    and r["metric"] == metric
                )
                values[metric] = f"{round(float(r['fraction']) * 168)}/168"
            text.append(
                f"| {group} | {temperature} | {values['quenched_same_atom']} | {values['quenched_requested_site']} |"
            )
    text.extend(
        [
            "\nThese are pointwise, frozen-threshold occupancy measurements. Loss followed by recovery is possible; this is not a continuous survival curve, an energy-barrier estimate or proof of natural structural equivalence. Same-atom and requested-site predicates answer different questions. The unedited-host group estimates background appearance.\n",
            "This successful-only study does not estimate successful creation and later retention per request. The full 256-request-per-model Kaggle workflow evaluates that outcome, with initial failures counting as zero. Do not mark that evaluation complete without its final results. D1-minus and other model/sampling settings remain outside this thermal study.\n",
        ]
    )
    (out / "README.md").write_text("\n".join(text))
    diagnostics = dict(
        n_trajectories=len(raw),
        n_observations=len(observations),
        max_residual_force=max(o["max_force"] for o in observations),
        polished_observations=sum(o["minimizer"] != "cg" for o in observations),
        max_initial_energy_error=max(r["initial_energy_error"] for r in raw),
    )
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
