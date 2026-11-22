"""Figures of the final course report (final_report.tex) that the long report does not have.

python paper/make_final_figures.py            # all
python paper/make_final_figures.py training   # or: baselines, features

fr_fig_training   training and validation curves of the denoiser and of the finder
fr_fig_baselines  what each method sees of one test vacancy: the score of every atom
                  under the three measures without learning, local pressure and the
                  finder, with the atom each one flags
fr_fig_features   the finder's learned features (t-SNE), before and after training on
                  defects, and how the found rate changes when the atoms are shaken
                  (results/final_report_20261117/, scripts/pilots/final_report_analysis.py)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_report_figures as rf  # noqa: E402
from figstyle import apply_publication_style, finalize_figure, panel_letter  # noqa: E402

plt = rf.plt
ROOT = rf.ROOT
FIG = rf.FIG
RES = ROOT / "results/final_report_20261117"
INK, MUTED, GRID = rf.INK, rf.MUTED, rf.GRID
DENOISER_LOG = ROOT / "results/kaggle_20261006-0454Z/training/egnn_uncond_smax1/log.csv"
FINDER_LOGS = (  # the two stages of the finder of the main results, and their first update
    (ROOT / "results/kaggle_pd_20261107-2024Z/training/pd_detect/log.jsonl", 0),
    (ROOT / "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/log.jsonl", 20000),
)
KIND = {  # fixed order and colours for the four defects (validation curves)
    "vacancy": ("#2A78D6", "Vacancy"),
    "interstitial": ("#EB6834", "Interstitial"),
    "antisite": ("#1BAF7A", "Antisite"),
    "frenkel": ("#6250D6", "Frenkel pair"),
}


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, pad=2, labelsize=6)


def title(ax, text: str) -> None:
    ax.set_title(text, fontsize=7.2, loc="left", pad=4)


# --------------------------------------------------------------------------- training


def fig_training() -> None:
    den = np.genfromtxt(DENOISER_LOG, delimiter=",", names=True)
    rows = []
    for path, offset in FINDER_LOGS:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r["step"] += offset
            rows.append(r)
    step = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.21, top=0.83, wspace=0.42)
    # (a) denoiser: training loss (all noise levels) and validation loss at fixed noise levels
    ax = axes[0]
    ax.plot(den["step"] / 1000, den["train_loss"], color=INK, lw=1.1, label="training")
    for col, colour in (("val_05", "#2A78D6"), ("val_03", "#86B6EF"), ("val_01", "#B7D3F6")):
        ax.plot(den["step"] / 1000, den[col], color=colour, lw=1.1,
                label=f"validation, $\\sigma_n$={col[-2]}.{col[-1]}")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("updates (thousands)", fontsize=6.3, labelpad=1)
    ax.set_ylabel("denoising loss", fontsize=6.3, labelpad=1)
    title(ax, "Denoiser")
    ax.legend(fontsize=5.2, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.1),
              handlelength=1.2, borderaxespad=0.1, labelspacing=0.25)  # fmt: skip
    style(ax)
    # (b) finder: training loss
    ax = axes[1]
    ax.plot(step / 1000, [r["train_loss"] for r in rows], color=INK, lw=1.1, marker="o", ms=1.8)
    ax.axvline(20, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.text(20.8, 0.97, "learning rate\nhalved", transform=ax.get_xaxis_transform(),
            fontsize=5.2, color=MUTED, va="top")  # fmt: skip
    ax.set_xlabel("updates (thousands)", fontsize=6.3, labelpad=1)
    ax.set_ylabel("weighted cross-entropy", fontsize=6.3, labelpad=1)
    title(ax, "Finder: training loss")
    style(ax)
    # (c) finder: validation, defects found per type
    ax = axes[2]
    for kind, (colour, label) in KIND.items():
        ax.plot(step / 1000, [100 * r[f"{kind}_hit"] for r in rows], color=colour, lw=1.1,
                label=label)  # fmt: skip
    ax.axvline(20, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_ylim(0, 70)
    ax.set_xlabel("updates (thousands)", fontsize=6.3, labelpad=1)
    ax.set_ylabel("% found (validation)", fontsize=6.3, labelpad=1)
    title(ax, "Finder: validation")
    ax.legend(fontsize=5.2, frameon=False, loc="lower right", ncol=2, handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.1, labelspacing=0.25)  # fmt: skip
    style(ax)
    for ax, letter in zip(axes, "abc"):
        panel_letter(fig, (ax.get_position().x0 - 0.06, ax.get_position().y1 + 0.075), letter)
    finalize_figure(fig, FIG / "fr_fig_training")


# --------------------------------------------------------------------------- baselines


def fig_baselines() -> None:
    sys.path.insert(0, str(ROOT / "scripts/pilots"))
    spec = importlib.util.spec_from_file_location("br", ROOT / "scripts/pilots/baseline_routes.py")
    br = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(br)
    import torch

    from glassdiff.data.point_defects import load_point_defects
    from glassdiff.physics.ka_potential import ka_energy, ka_virial
    from glassdiff.types import Structures

    z = np.load(rf.EXAMPLE)
    pos, types, box = z["pos"].astype(float), z["types"], z["box"].astype(float)
    site = np.mod(z["site"], box)
    # reference statistics: the defect-free test glasses, as in the report
    ref = next(iter(load_point_defects(ROOT / "data/ka2d_256_point_defects/test.npz").values()))
    _, first = np.unique(ref.glass.numpy(), return_index=True)
    r = ref.subset(torch.from_numpy(first))
    rp, rt, rb = r.orig_pos.numpy(), r.orig_types.numpy(), r.box.numpy()
    s_ref = Structures(r.orig_pos, r.orig_types, r.box)

    def stats(values):
        return br.species_stats(values, rt)

    mu_a, sd_a = stats(np.stack([br.voronoi_area(p, b) for p, b in zip(rp, rb)]))
    mu_u, sd_u = stats(np.stack([br.unlike_neighbours(p, t, b) for p, t, b in zip(rp, rt, rb)]))
    mu_e, sd_e = stats(ka_energy(s_ref, per_atom=True).numpy())
    mu_w, sd_w = stats(ka_virial(s_ref).numpy())
    s = Structures(torch.tensor(pos)[None], torch.tensor(types).long()[None], torch.tensor(box)[None])  # fmt: skip
    scores = {  # higher = more suspicious, as each method ranks the atoms for a vacancy
        "Cell size\n(largest cell)": (br.voronoi_area(pos, box) - mu_a[types]) / sd_a[types],
        "Unlike neighbours\n(fewest)": -(br.unlike_neighbours(pos, types, box) - mu_u[types])
        / sd_u[types],
        "Atom energy\n(highest)": (ka_energy(s, per_atom=True)[0].numpy() - mu_e[types])
        / sd_e[types],
        "Local pressure\n(lowest)": -(ka_virial(s)[0].numpy() - mu_w[types]) / sd_w[types],
        "Learned finder\n(probability)": z["prob"],
    }
    half = 3.1
    fig = plt.figure(figsize=(5.5, 1.62))
    width, gap, left, bottom, height = 0.178, 0.0215, 0.006, 0.05, 0.66
    for k, (name, sc) in enumerate(scores.items()):
        ax = fig.add_axes([left + k * (width + gap), bottom, width, height])
        rank = np.argsort(np.argsort(sc)) / (len(sc) - 1)  # 0 = least, 1 = most suspicious
        colours = [rf.BLUES(v**3) for v in rank]
        rf.draw(ax, pos, types, box, site, half, face=colours, lw=0.3)
        rf.ring(ax, site, rf.DIAMETER[z["otypes"][z["atom"]]], style=(0, (1.2, 1.2)), lw=0.9)
        flag = int(np.argmax(sc))
        p = rf.near(pos[flag], box, site)
        dist = float(np.linalg.norm(rf.wrap(pos[flag] - site, box)))
        hit = dist < 1.5
        rf.ring(
            ax, p, rf.DIAMETER[types[flag]] + 0.28, colour="#E34948" if not hit else INK, lw=1.2
        )
        inside = abs(p - site).max() < half
        verdict = ("found" if hit else "missed") + ("" if inside else ", flag outside view")
        x0, y1 = ax.get_position().x0, ax.get_position().y1
        fig.text(x0 + 0.002, y1 + 0.03, name, fontsize=6.0, ha="left", va="bottom",
                 linespacing=1.05)  # fmt: skip
        fig.text(x0 + width / 2, bottom - 0.035, f"{verdict} ({dist:.1f}$\\sigma$)", fontsize=5.6,
                 ha="center", va="top", color=INK if hit else "#B02A2A")  # fmt: skip
    finalize_figure(fig, FIG / "fr_fig_baselines")


# --------------------------------------------------------------------------- features


ROLE = {  # fixed order and colours for the five roles
    0: ("#D6D5CE", "intact"),
    1: ("#2A78D6", "next to an empty site"),
    2: ("#EB6834", "extra atom"),
    3: ("#6250D6", "displaced atom"),
    4: ("#1BAF7A", "wrong species"),
}
METHOD = {  # finder of the report, the two other networks, the three measures
    "finder": ("#2A78D6", "EGNN finder", "-"),
    "egnn_scratch": ("#86B6EF", "EGNN, random start", "-"),
    "transformer": ("#E87BA4", "Transformer", "-"),
    "cnn": ("#008300", "CNN", "-"),
    "cell size": ("#1BAF7A", "Cell size", (0, (3, 1.5))),
    "unlike neighbours": ("#6250D6", "Unlike neighb.", (0, (3, 1.5))),
    "atom energy": ("#EB6834", "Atom energy", (0, (3, 1.5))),
}


def fig_features() -> None:
    f = np.load(RES / "features.npz")
    info = json.loads((RES / "features.json").read_text())
    sens = json.loads((RES / "sensitivity.json").read_text())
    fig = plt.figure(figsize=(5.5, 1.8))
    ax_d = fig.add_axes([0.0, 0.06, 0.27, 0.72])
    ax_f = fig.add_axes([0.285, 0.06, 0.27, 0.72])
    ax_s = fig.add_axes([0.66, 0.2, 0.335, 0.62])
    role = f["role"]
    order = np.argsort(role != 0)  # intact atoms first, under the others
    for ax, key, name in ((ax_d, "denoiser", "Denoiser features"), (ax_f, "finder", "Finder features")):  # fmt: skip
        xy = f[key][order]
        ax.scatter(xy[:, 0], xy[:, 1], s=0.6, c=[ROLE[int(r)][0] for r in role[order]],
                   linewidths=0, rasterized=True)  # fmt: skip
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(MUTED)
            sp.set_linewidth(0.5)
        acc = info[f"{key}_knn_balanced_accuracy"]
        title(ax, f"{name}\n(roles told apart: {100 * acc:.0f}%)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=3, color=c) for c, _ in ROLE.values()]
    fig.legend(handles, [n for _, n in ROLE.values()], loc="lower center", ncol=5, fontsize=5.6,
               frameon=False, handletextpad=0.2, columnspacing=0.9,
               bbox_to_anchor=(0.28, -0.07))  # fmt: skip
    levels = sens["levels"]
    for key, (colour, name, ls) in METHOD.items():
        y = [100 * sens["found"][str(lv)][key]["all"] for lv in levels]
        ax_s.plot(levels, y, color=colour, lw=1.2, ls=ls, marker="o", ms=2.2, label=name)
    leg = ax_s.legend(fontsize=5.0, frameon=True, loc="upper right", handlelength=1.8,
                      borderaxespad=0.1, labelspacing=0.2, framealpha=1.0)  # fmt: skip
    leg.get_frame().set_edgecolor("none")
    ax_s.set_xlim(-0.004, levels[-1] + 0.004)
    ax_s.set_xticks(levels)
    ax_s.set_ylim(0, 80)
    ax_s.set_yticks([0, 20, 40, 60])
    ax_s.set_xlabel("random shift of every atom ($\\sigma$)", fontsize=6.3, labelpad=1)
    ax_s.set_ylabel("% of test defects found", fontsize=6.3, labelpad=1)
    title(ax_s, "Shaken inputs")
    style(ax_s)
    for ax, letter in zip((ax_d, ax_f, ax_s), "abc"):
        panel_letter(fig, (ax.get_position().x0 + (0.005 if ax is not ax_s else -0.06), 0.97), letter)  # fmt: skip
    finalize_figure(fig, FIG / "fr_fig_features")


def main() -> None:
    apply_publication_style()
    which = set(sys.argv[1:]) or {"training", "baselines", "features"}
    for name in ("training", "baselines", "features"):
        if name in which:
            globals()[f"fig_{name}"]()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
