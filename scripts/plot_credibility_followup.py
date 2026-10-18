"""Plot saved credibility follow-up outputs and summarize matched repeat batches."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glassdiff-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from glassdiff.geometry import minimum_image
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import Structures

_spec = importlib.util.spec_from_file_location(
    "physical", Path(__file__).with_name("check_physical_credibility.py")
)
physical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(physical)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out
    torch.set_num_threads(1)
    rows = list(csv.DictReader((out / "host_aware_structure.csv").open()))
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    methods = list(physical.RUNS)
    for ax, metric in zip(axs, ["energy", "angle", "pair_AA", "pair_AB", "pair_BB"]):
        for j, method in enumerate(methods):
            r = next(r for r in rows if r["method"] == method and r["metric"] == metric)
            x = float(r["w1"])
            lo = float(r["lo"])
            hi = float(r["hi"])
            ax.errorbar(
                x,
                j - 0.12,
                xerr=[[max(0, x - lo)], [max(0, hi - x)]],
                fmt="o",
                color="tab:blue",
                capsize=3,
            )
            x = float(r["matched_natural_median"])
            lo = float(r["matched_natural_p025"])
            hi = float(r["matched_natural_p975"])
            ax.errorbar(
                x,
                j + 0.12,
                xerr=[[max(0, x - lo)], [max(0, hi - x)]],
                fmt="s",
                color="tab:orange",
                capsize=3,
            )
        ax.set(
            yticks=range(len(methods)),
            yticklabels=methods,
            title=metric,
            xlabel="W1 (lower is closer)",
        )
        ax.invert_yaxis()
    fig.suptitle(
        "Blue: successful placements, host-bootstrap CI; orange: count-matched natural subsampling band"
    )
    fig.tight_layout()
    fig.savefig(out / "host_aware_structure.png", dpi=160)
    plt.close(fig)
    rows = list(csv.DictReader((out / "host_disturbance.csv").open()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in methods:
        rs = [r for r in rows if r["method"] == method and r["group"] == "successful"]
        x = [float(r["r_lo"]) + 0.5 for r in rs]
        ax.plot(x, [float(r["host_balanced_mean_displacement"]) for r in rs], ".-", label=method)
        ax.fill_between(x, [float(r["lo"]) for r in rs], [float(r["hi"]) for r in rs], alpha=0.1)
    ax.set(
        xlabel="Distance from original target (shell midpoint)",
        ylabel="Mean displacement from original host",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "host_disturbance.png", dpi=160)
    plt.close(fig)
    # Select fixed percentiles of local mean energy, without judging them visually.
    picks = []
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for row, method in enumerate(["fixed_r6", "hand_insert"]):
        s, req, _ = physical.load_run(physical.RUNS[method])
        th = physical.DefectThresholds.from_dict(
            json.loads((physical.DATA / "meta.json").read_text())["defects"]["thresholds"]
        )
        valid = np.flatnonzero(physical.success(s, req, th).numpy())
        energies = []
        for i in valid:
            sub = Structures(s.pos[i : i + 1], s.types[i : i + 1], s.box[i : i + 1])
            radius = minimum_image(sub.pos - req.target[i : i + 1, None], sub.box).norm(dim=-1)[0]
            energies.append(float(ka_energy(sub, per_atom=True)[0][radius <= 3].mean()))
        order = np.argsort(energies)
        for col, q in enumerate([0.1, 0.5, 0.9]):
            k = int(order[round(q * (len(order) - 1))])
            i = int(valid[k])
            ax = axs[row, col]
            d = minimum_image(s.pos[i : i + 1] - req.target[i : i + 1, None], s.box[i : i + 1])[
                0
            ].numpy()
            types = s.types[i].numpy()
            for species, color in [(0, "lightgray"), (1, "tab:blue")]:
                z = d[types == species]
                ax.scatter(
                    z[:, 0],
                    z[:, 1],
                    s=18 if species == 0 else 12,
                    c=color,
                    edgecolors="black",
                    linewidths=0.2,
                )
            ax.add_patch(plt.Circle((0, 0), 3, fill=False, color="green", linestyle="--"))
            ax.add_patch(plt.Circle((0, 0), 6, fill=False, color="orange", linestyle=":"))
            ax.plot(0, 0, "r+", markersize=12)
            ax.set(
                xlim=(-7, 7),
                ylim=(-7, 7),
                aspect="equal",
                title=f"{method}, sample {i}, q={q:g}\nlocal mean energy={energies[k]:.3f}",
            )
            picks.append(dict(method=method, sample=i, quantile=q, local_mean_energy=energies[k]))
    fig.suptitle("Predefined energy percentiles among successful samples; green r=3, orange r=6")
    fig.tight_layout()
    fig.savefig(out / "representative_structures.png", dpi=160)
    plt.close(fig)
    physical.write_csv(out / "representative_selection.csv", picks)
    # Distinguish creation at initially non-D2 sites from preserving pre-existing D2.
    test = physical.load_split(physical.DATA / "test.npz").structures
    th = physical.DefectThresholds.from_dict(
        json.loads((physical.DATA / "meta.json").read_text())["defects"]["thresholds"]
    )
    creation, outcomes = [], []
    for method, path in physical.RUNS.items():
        final, request, saved = physical.load_run(path)
        ids = physical.recover_hosts(saved, test)
        original = Structures(test.pos[ids], test.types[ids], test.box[ids])
        prior = physical.success(original, request, th).numpy()
        after = physical.success(final, request, th).numpy()
        clean = ~prior
        creation.append(
            dict(
                method=method,
                n=len(prior),
                originally_D2=int(prior.sum()),
                finally_D2=int(after.sum()),
                newly_D2=int((clean & after).sum()),
                initially_nonD2=int(clean.sum()),
                creation_fraction_on_initially_nonD2=float(after[clean].mean()),
            )
        )
        for i in range(len(prior)):
            outcomes.append(
                dict(
                    method=method,
                    sample=i,
                    host_id=int(ids[i]),
                    originally_D2=int(prior[i]),
                    finally_D2=int(after[i]),
                )
            )
    physical.write_csv(out / "creation_summary.csv", creation)
    physical.write_csv(out / "host_creation_outcomes.csv", outcomes)
    repeat = []
    for path in sorted((out / "repeats").glob("*/samples.npz")):
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        n = len(d["success"])
        k = int(d["success"].sum())
        lo, hi = physical.wilson_interval(k, n)
        delta = minimum_image(
            torch.from_numpy(d["pos"]) - torch.from_numpy(d["host_pos"]), torch.from_numpy(d["box"])
        ).norm(dim=-1)
        radius = minimum_image(
            torch.from_numpy(d["host_pos"]) - torch.from_numpy(d["target"])[:, None],
            torch.from_numpy(d["box"]),
        ).norm(dim=-1)
        outside = [float(delta[i][radius[i] > 6].mean()) for i in range(n)]
        repeat.append(
            dict(
                model=path.parent.name,
                n=n,
                n_success=k,
                success=k / n,
                lo=lo,
                hi=hi,
                n_converged=int(d["converged"].sum()),
                outside_r6_mean=float(np.mean(outside)),
                max_force=float(d["max_force"].max()),
            )
        )
    if repeat:
        physical.write_csv(out / "repeat_summary.csv", repeat)
    print("Figures and available repeat summaries saved", flush=True)


if __name__ == "__main__":
    main()
