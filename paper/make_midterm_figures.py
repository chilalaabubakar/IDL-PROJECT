"""Figure 6 of the midterm report: the follow-up checks (run from the repository root).

    python paper/make_midterm_figures.py

  (a) choosing the guidance weight w for each trained model on validation glasses
      (notebooks/kaggle_extras.ipynb, E1): test success per training seed at fixed w = 2,
      fixed w = 6 and at the w each model's validation rows chose;
  (b) the seam check (E2): local energy W1 in 1-sigma shells around the site for regions of
      radius R_out = 3-6 (seed 0, w = 2), hand insertion and the natural floor;
  (c) training with R_out drawn from 3-6 (notebooks/kaggle_rout_range.ipynb): success
      against local W1 for D2, fixed-R_out model -> R_out-range model at the same setting.

Data: results/*/summary/*.json and results/*/experiments/<row>/metrics.json (newest push).
Style: paper/figstyle.py.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import PALETTE as P  # noqa: E402

OUT = Path("paper/figures")


def newest(pattern: str) -> str:
    paths = sorted(
        glob.glob(pattern), key=lambda p: p.split("/results/")[-1] if "/results/" in p else p
    )
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[-1]


def summary(name: str) -> dict | list:
    return json.load(open(newest(f"results/*/summary/{name}")))


def metrics(row: str) -> dict:
    return json.load(open(newest(f"results/*/experiments/{row}/metrics.json")))


def panel_w(ax) -> None:
    rows = {r["setting"].split(" {")[0]: r for r in summary("w_selection_test.json")}
    order = [
        ("fixed w = 2", "fixed\nw = 2"),
        ("fixed w = 6", "fixed\nw = 6"),
        ("w per model, success rule", "w chosen\nper model"),
    ]
    seed_marker = {0: "o", 1: "s", 2: "^"}
    hand = metrics("hand_insert_D2")["success_after_relax"]
    ax.axhspan(hand[1], hand[2], color=P["red_1"], lw=0, zorder=0)
    ax.axhline(hand[0], color=P["red_strong"], lw=0.8, ls="--", zorder=1)
    ax.text(
        2.45,
        hand[0] + 0.02,
        "hand\ninsertion",
        color=P["red_strong"],
        fontsize=5.8,
        ha="right",
        va="bottom",
    )
    for i, (key, label) in enumerate(order):
        r = rows[key]
        for s, name in enumerate(r["rows"]):
            m = metrics(name)["success_after_relax"]
            x = i + (s - 1) * 0.16
            ax.plot([x, x], [m[1], m[2]], color=P["blue_secondary"], lw=0.6, alpha=0.5)
            ax.plot(
                x,
                m[0],
                marker=seed_marker[s],
                ms=3.6,
                mfc="white",
                mec=P["blue_main"],
                mew=0.8,
                ls="none",
                zorder=3,
            )
        mean, sd = r["success_mean"], r["success_sd"]
        ax.plot([i - 0.3, i + 0.3], [mean, mean], color=P["blue_main"], lw=1.4, zorder=4)
        top = max(metrics(n)["success_after_relax"][2] for n in r["rows"])
        ax.text(
            i,
            top + 0.03,
            f"{mean:.2f}\n± {sd:.2f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=P["blue_main"],
            linespacing=0.95,
        )
    ax.set_xticks(range(3), [lab for _, lab in order])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("D2 success after relaxation (test)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker=seed_marker[s],
            ms=3.6,
            mfc="white",
            mec=P["blue_main"],
            ls="none",
            label=f"seed {s}",
        )
        for s in range(3)
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=5.6,
        handletextpad=0.2,
        borderaxespad=0.1,
        labelspacing=0.2,
    )


def panel_seam(ax) -> None:
    d = summary("seam_D2.json")
    floor = np.array(d["floor"]["D2"]["shells"])
    mid = (floor[:, 0] + floor[:, 1]) / 2
    ax.fill_between(mid, floor[:, 3], floor[:, 4], color=P["neutral"], alpha=0.7, lw=0, zorder=0)
    ax.plot(mid, floor[:, 2], color=P["grey"], lw=0.8, ls="--", zorder=1)
    blues = {3.0: "#9DC0E5", 4.0: "#5C93CF", 5.0: P["blue_secondary"], 6.0: P["blue_main"]}
    for name, r in d["runs"].items():
        sh = np.array(r["shells"])
        x = (sh[:, 0] + sh[:, 1]) / 2
        if r["r_out"] is None:  # hand insertion
            ax.plot(
                x,
                sh[:, 2],
                color=P["red_strong"],
                lw=1.0,
                marker="D",
                ms=2.6,
                zorder=3,
                label="hand insertion",
            )
            continue
        c = blues[r["r_out"]]
        ax.plot(
            x,
            sh[:, 2],
            color=c,
            lw=1.0,
            marker="o",
            ms=2.6,
            zorder=3,
            label=f"$R_{{out}}$ = {r['r_out']:g}",
        )
        ax.plot([r["r_out"]] * 2, [0.0, 0.012], color=c, lw=1.6, zorder=2, solid_capstyle="butt")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 0.21)
    ax.set_xlabel("distance from the site (σ), 1σ shells")
    ax.set_ylabel("local energy W1 in the shell")
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.4, labelspacing=0.2)
    ax.text(0.1, 0.004, "ticks: $R_{out}$", fontsize=5.4, color=P["grey_dark"], va="bottom")


def panel_range(ax) -> None:
    d = summary("rrange_vs_fixed.json")
    ax.axvspan(0.0, 0.013, color=P["neutral"], alpha=0.7, lw=0, zorder=0)
    hand = metrics("hand_insert_D2")
    s, w = hand["success_after_relax"], hand["local"]["w1_pe_atom"]
    ax.plot(w[0], s[0], marker="D", ms=4.0, color=P["red_strong"], ls="none", zorder=4)
    ax.text(
        w[0],
        s[0] - 0.05,
        "hand\ninsertion",
        color=P["red_strong"],
        fontsize=5.8,
        ha="center",
        va="top",
    )
    for r in d:
        if not r["range"].endswith("_D2"):
            continue
        a, b = r["fixed_metrics"], r["range_metrics"]
        w6 = "_w6_" in r["range"]
        tag = r["range"].split("_rout")[1].split("_")[0]
        col = P["blue_main"] if w6 else P["blue_secondary"]
        ax.annotate(
            "",
            xy=(b["w1"][0], b["success"][0]),
            xytext=(a["w1"][0], a["success"][0]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=P["grey"],
                lw=0.7,
                shrinkA=2.5,
                shrinkB=2.5,
                mutation_scale=6,
            ),
        )
        ax.plot(
            a["w1"][0], a["success"][0], marker="o", ms=3.8, mfc=col, mec=col, ls="none", zorder=4
        )
        ax.plot(
            b["w1"][0],
            b["success"][0],
            marker="o",
            ms=3.8,
            mfc="white",
            mec=col,
            mew=0.9,
            ls="none",
            zorder=4,
        )
        ax.text(
            a["w1"][0] + 0.003,
            a["success"][0] + 0.012,
            f"{tag}σ" + (", w=6" if w6 else ""),
            fontsize=5.4,
            color=col,
        )
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("local energy W1 (3σ window)")
    ax.set_ylabel("D2 success after relaxation")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=3.8,
            mfc=P["blue_secondary"],
            mec=P["blue_secondary"],
            ls="none",
            label="trained at $R_{out}$ = 4",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=3.8,
            mfc="white",
            mec=P["blue_secondary"],
            ls="none",
            label="trained on 3–6",
        ),
        plt.Line2D([], [], color=P["neutral"], lw=4, label="natural floor"),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.14, 1.02),
        fontsize=5.6,
        handletextpad=0.3,
        labelspacing=0.2,
    )


def main() -> None:
    fs.apply_publication_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(5.5, 1.95), gridspec_kw={"width_ratios": [1.0, 1.15, 1.0], "wspace": 0.42}
    )
    panel_w(axes[0])
    panel_seam(axes[1])
    panel_range(axes[2])
    for ax, letter in zip(axes, "abc"):
        bb = ax.get_position()
        fs.panel_letter(fig, (bb.x0 - 0.055, bb.y1 + 0.02), letter)
    paths = fs.finalize_figure(fig, OUT / "fig6_followups")
    print("wrote", *paths)


if __name__ == "__main__":
    main()
