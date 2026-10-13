"""Figure for the training-seed study (results/seeds/README.md).

    python scripts/plot_seeds.py --config configs/eval/default.yaml out=results/seeds/seeds.png

Reads the headline rows of seed 0 (rounds 1/3) and the seed1_* / seed2_* rows pushed by
notebooks/kaggle_seeds.ipynb, plus the baselines and the natural floors. One panel per
headline task: success after relaxation against local W1 (lower = more natural), one point
per training seed with 95% bootstrap CIs; colour = method, marker = seed.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from glassdiff.utils.config import cli_config  # noqa: E402

COLOR = {
    "w = 2": "#2a78d6",
    "w = 6": "#eb6834",
    "hand insertion": "#1baf7a",
    "local melt-quench": "#eda100",
}
MARKER = {0: "o", 1: "s", 2: "^"}
PANELS = [
    (
        "Task B, D2 (retrained recipe)",
        "D2",
        {"w = 2": "cond_smax1_w2_s100_host_D2", "w = 6": "cond_smax1_w6_s100_host_D2"},
        {"hand insertion": "hand_insert_D2", "local melt-quench": "local_melt_quench_D2"},
    ),
    (
        "Task B, D1− (round 1 recipe)",
        "D1_MINUS",
        {"w = 2": "cond_w2_s100_host_D1_MINUS", "w = 6": "cond_w6_s100_host_D1_MINUS"},
        {
            "hand insertion": "hand_insert_D1_MINUS",
            "local melt-quench": "local_melt_quench_D1_MINUS",
        },
    ),
    ("Task A, D2 (round 1 recipe)", "D2", {"w = 6": "cond_w6_s100_centre_D2"}, {}),
]


def metrics(name: str) -> dict:
    paths = sorted(glob.glob(f"results/*/experiments/{name}/metrics.json"))
    if not paths:
        raise FileNotFoundError(f"{name}: run notebooks/kaggle_seeds.ipynb first")
    return json.loads(Path(paths[-1]).read_text())


def point(ax, m, color, marker, label=None):
    s, w = m["success_after_relax"], m["local"]["w1_pe_atom"]
    ax.errorbar(
        w[0],
        s[0],
        xerr=[[w[0] - w[1]], [w[2] - w[0]]],
        yerr=[[s[0] - s[1]], [s[2] - s[0]]],
        color=color,
        marker=marker,
        ms=8,
        lw=1.2,
        capsize=2,
        ls="none",
        label=label,
    )


def main() -> None:
    cfg = cli_config(__doc__)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (title, defect, model_rows, baselines) in zip(ax, PANELS):
        floor = json.loads(Path(f"results/sensitivity/floor_{defect}.json").read_text())
        lo, hi = floor["variants"]["default"]["local_w1_pe_atom"][1:]
        a.axvspan(lo, hi, color="0.88", lw=0, label="natural floor (95% CI)")
        for label, row in model_rows.items():
            for seed in (0, 1, 2):
                name = row if seed == 0 else f"seed{seed}_{row}"
                point(
                    a,
                    metrics(name),
                    COLOR[label],
                    MARKER[seed],
                    f"model {label}, seed {seed}" if a is ax[0] or len(model_rows) == 1 else None,
                )
        for label, row in baselines.items():
            point(a, metrics(row), COLOR[label], "D", label if a is ax[0] else None)
        a.set_title(title, fontsize=10)
        a.set_xlabel("local W1 of per-atom energy (lower = more natural)")
        a.set_xlim(left=0)
        a.set_ylim(bottom=0)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="0.93")
    ax[0].set_ylabel("success after relaxation")
    ax[0].legend(fontsize=7, frameon=False, loc="upper right")
    ax[2].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    out = Path(cfg.get("out", "results/seeds/seeds.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
