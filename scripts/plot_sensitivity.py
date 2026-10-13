"""Figure for the sensitivity checks (results/sensitivity/README.md).

    python scripts/plot_sensitivity.py --config configs/eval/default.yaml \\
        out=results/sensitivity/sensitivity.png

Reads the sens_* experiments pushed by notebooks/kaggle_sensitivity.ipynb (their
metrics.json and sensitivity.json under results/*/experiments/) and the floors in
results/sensitivity/. Four panels: Task B success against the defect cutoff (D2, D1-),
local W1 against the window radius r_loc, and local W1 against the regenerated-disk
radius R_out (D2). Error bars are the 95% bootstrap CIs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glassdiff.utils.config import cli_config  # noqa: E402

# reference categorical slots 1-4, with a marker each so identity never rests on colour alone
STYLE = {
    "model, w = 2": dict(color="#2a78d6", marker="o"),
    "model, w = 6": dict(color="#eb6834", marker="s"),
    "hand insertion": dict(color="#1baf7a", marker="^"),
    "local melt-quench": dict(color="#eda100", marker="D"),
}
RUNS = {
    "D2": {
        "model, w = 2": "sens_cond_smax1_w2_rout{}_host_D2",
        "model, w = 6": "sens_cond_smax1_w6_rout{}_host_D2",
        "hand insertion": "sens_hand_insert_D2",
        "local melt-quench": "sens_lmq_rout{}_D2",
    },
    "D1_MINUS": {
        "model, w = 2": "sens_cond_w2_rout{}_host_D1_MINUS",
        "model, w = 6": "sens_cond_w6_rout{}_host_D1_MINUS",
        "hand insertion": "sens_hand_insert_D1_MINUS",
        "local melt-quench": "sens_lmq_rout{}_D1_MINUS",
    },
}


def load(name: str, fname: str) -> dict:
    paths = sorted(glob.glob(f"results/*/experiments/{name}/{fname}"))
    if not paths:
        raise FileNotFoundError(f"{name}/{fname}: run notebooks/kaggle_sensitivity.ipynb first")
    return json.loads(Path(paths[-1]).read_text())


def errorbar(ax, x, rows, label, offset):
    est = [r[0] for r in rows]
    err = [[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]]
    xs = [v + offset for v in x]
    ax.errorbar(xs, est, yerr=err, lw=2, ms=8, capsize=3, label=label, **STYLE[label])


def floor_band(ax, x, rows):
    ax.fill_between(x, [r[1] for r in rows], [r[2] for r in rows], color="0.85", lw=0)
    ax.plot(x, [r[0] for r in rows], color="0.55", lw=1, ls="--", label="natural floor")


def main() -> None:
    cfg = cli_config(__doc__)
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    # (a, b) success against the defect cutoff
    for k, (defect, key, values) in enumerate(
        (("D2", "d2_rc", (1.15, 1.2, 1.25)), ("D1_MINUS", "cn_factor", (1.35, 1.4, 1.45)))
    ):
        for j, (label, run) in enumerate(RUNS[defect].items()):
            sens = load(run.format(4), "sensitivity.json")["variants"]
            names = [f"{key}={v:g}" if v != values[1] else "default" for v in values]
            rows = [sens[n]["success_after_relax"] for n in names]
            errorbar(ax[k], values, rows, label, (j - 1.5) * 0.004)
        title = (
            "D2 (B atom with ≥ 2 B neighbours)" if defect == "D2" else "D1− (A atom with CN ≤ 5)"
        )
        ax[k].set_title(f"Task B success, {title}", fontsize=10)
        ax[k].set_xlabel("B–B cutoff d2_rc (σ)" if defect == "D2" else "coordination cutoff factor")
        ax[k].set_xticks(values)
        ax[k].axvline(values[1], color="0.8", lw=0.8, zorder=0)
    ax[0].set_ylabel("success after relaxation")
    # (c) D2 local W1 against the window radius
    radii = (2.5, 3.0, 3.5)
    floor = load_floor("D2")
    floor_band(ax[2], radii, [floor[f"r_loc={r:g}" if r != 3.0 else "default"] for r in radii])
    for j, (label, run) in enumerate(RUNS["D2"].items()):
        sens = load(run.format(4), "sensitivity.json")["variants"]
        rows = [sens[f"r_loc={r:g}" if r != 3.0 else "default"]["local_w1_pe_atom"] for r in radii]
        errorbar(ax[2], radii, rows, label, (j - 1.5) * 0.02)
    ax[2].set_title("D2 surroundings vs window radius (R_out = 4)", fontsize=10)
    ax[2].set_xlabel("realism window r_loc (σ)")
    ax[2].set_ylabel("local W1 of per-atom energy")
    ax[2].set_xticks(radii)
    # (d) D2 local W1 against R_out (hand insertion has no R_out: drawn at its one value)
    r_outs = (3, 4, 5)
    floor_band(ax[3], r_outs, [floor["default"]] * 3)
    for j, (label, run) in enumerate(RUNS["D2"].items()):
        if "{}" in run:
            rows = [load(run.format(r), "metrics.json")["local"]["w1_pe_atom"] for r in r_outs]
            errorbar(ax[3], r_outs, rows, label, (j - 1.5) * 0.04)
        else:
            w = load(run, "metrics.json")["local"]["w1_pe_atom"]
            ax[3].axhline(
                w[0], color=STYLE[label]["color"], lw=1.5, ls=":", label=f"{label} (no R_out)"
            )
    ax[3].set_title("D2 surroundings vs regenerated disk (r_loc = 3)", fontsize=10)
    ax[3].set_xlabel("regenerated disk radius R_out (σ)")
    ax[3].set_xticks(r_outs)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="0.92")
        a.set_ylim(bottom=0)
    ax[0].legend(fontsize=8, frameon=False)
    ax[2].legend(fontsize=8, frameon=False)
    ax[3].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(cfg.get("out", "results/sensitivity/sensitivity.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


def load_floor(defect: str) -> dict:
    d = json.loads(Path(f"results/sensitivity/floor_{defect}.json").read_text())
    return {k: v["local_w1_pe_atom"] for k, v in d["variants"].items()}


if __name__ == "__main__":
    main()
