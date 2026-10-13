"""Build the paper's figures (run from the repository root).

    python paper/make_examples.py part=model; python paper/make_examples.py part=hand
    python paper/make_examples.py part=lmq          # once: example samples (CPU)
    python paper/make_figures.py

Style: paper/figstyle.py (figures4papers conventions), drawn at printed size.
  Figure 1  overview: (a) workflow, (b) natural defects in a test glass, (c) the Task B
            request on a real host, (d) a real sampling trajectory (examples_D2_model.npz).
  Figure 2  one Task B request answered by the model, hand insertion and local melt-quench.
  Figure 3  training seeds (data of scripts/plot_seeds.py).
  Figure 4  sensitivity checks (data of scripts/plot_sensitivity.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import PALETTE as P  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    ConnectionPatch,
    FancyArrow,
    FancyBboxPatch,
    Patch,
)

OUT = Path("paper/figures")
DATA = Path("paper/data")
EXAMPLE = 2  # request index (of make_examples.py) shown in Figures 1d and 2
SNAP_STEPS = (0, 25, 100)  # sampler steps shown in Figure 1d, then the relaxed sample
NATURAL = (1, 154, 144)  # Figure 1b: test glass, a D1- atom, a nearby D1+ atom


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inches(fig, x, y, w, h):
    W, H = fig.get_size_inches()
    return fig.add_axes((x / W, y / H, w / W, h / H))


def _min_image(d, box):
    return d - box * np.round(d / box)


# --------------------------------------------------------------------------- Figure 1


def _box(ax, x, y, w, h, num, title, lines, fc, tc, ec=None, lw=0.8, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.06",
            fc=fc,
            ec=ec or P["ink"],
            lw=lw,
            ls=ls,
            zorder=1,
        )
    )
    ty = y + h - 0.105
    if num:
        ax.add_patch(Circle((x + 0.095, ty), 0.06, fc=P["ink"], ec="none", zorder=3))
        ax.text(
            x + 0.095,
            ty - 0.003,
            num,
            color="white",
            fontsize=5.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=4,
        )
    ax.text(
        x + (0.185 if num else 0.07),
        ty,
        title,
        color=tc,
        fontsize=7,
        fontweight="bold",
        va="center",
        ha="left",
        zorder=4,
    )
    for k, line in enumerate(lines):
        ax.text(
            x + 0.07, ty - 0.165 - 0.107 * k, line, fontsize=6, va="center", ha="left", zorder=4
        )


def _block_arrow(ax, x0, y0, x1, y1, fc=None):
    ax.add_patch(
        FancyArrow(
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            width=0.045,
            head_width=0.11,
            head_length=0.065,
            length_includes_head=True,
            fc=fc or P["neutral"],
            ec=P["ink"],
            lw=0.6,
            zorder=2,
        )
    )


def _workflow(ax):
    """Panel (a): numbered workflow in inch coordinates."""
    bw, gap, x0 = 1.215, 0.2, 0.02
    cx = [x0 + k * (bw + gap) for k in range(4)]
    h1, h2, y2, y1 = 0.6, 0.72, 0.0, 1.14
    green, blue, grey = (
        (P["green_1"], P["green_dark"]),
        (P["blue_1"], P["blue_main"]),
        (P["neutral_1"], P["grey_dark"]),
    )
    _box(
        ax,
        cx[0],
        y1,
        bw,
        h1,
        "1",
        "Glasses",
        ["2D Kob–Andersen, N = 256", "LAMMPS melt–quench", "800 / 100 / 100 glasses"],
        *green,
    )
    _box(
        ax,
        cx[1],
        y1,
        bw,
        h1,
        "2",
        "Label defects",
        [r"D1$^{-}$: A atom, CN ≤ 5", r"D1$^{+}$: A atom, CN ≥ 8", "D2: B atom, ≥ 2 B nbrs"],
        *green,
    )
    _box(
        ax,
        cx[2],
        y1,
        bw,
        h1,
        "3",
        r"Train denoiser $d_\theta$",
        ["periodic EGNN, 0.53 M", r"in: $x_\sigma$, $\sigma$, pins, labels", "15% label dropout"],
        *blue,
    )
    _box(
        ax,
        cx[3],
        y1,
        bw,
        h1,
        "",
        "Natural reference",
        ["natural test-glass defects", "floor: W1(train, test)", r"D2 0.013, D1$^{-}$ 0.016"],
        *grey,
        ec=P["grey"],
        ls=(0, (3, 2)),
    )
    _box(
        ax,
        cx[0],
        y2,
        bw,
        h2,
        "4",
        "Request",
        [
            "host: a test glass",
            r"site atom at $p$, label $c$",
            r"pins: site, $r$ > $R_\mathrm{out}$ = 4σ",
            r"≈ 58 atoms to place",
        ],
        *green,
    )
    _box(ax, cx[1], y2, bw, h2, "5", "Generate the disk", [], "white", P["ink"])
    for (fc, ec, tc, lines), yy in zip(
        (
            (
                P["blue_1"],
                P["blue_main"],
                P["blue_main"],
                ["score dynamics + CFG", "100 + 100 steps"],
            ),
            (
                P["red_1"],
                P["red_strong"],
                P["red_strong"],
                ["baselines: hand insertion,", "local melt-quench"],
            ),
        ),
        (y2 + 0.285, y2 + 0.04),
    ):
        ax.add_patch(
            FancyBboxPatch(
                (cx[1] + 0.05, yy),
                bw - 0.1,
                0.23,
                boxstyle="round,pad=0,rounding_size=0.04",
                fc=fc,
                ec=ec,
                lw=0.6,
                zorder=2,
            )
        )
        for k, line in enumerate(lines):
            ax.text(
                cx[1] + 0.1,
                yy + 0.165 - 0.1 * k,
                line,
                fontsize=6,
                va="center",
                color=tc if k == 0 and fc == P["blue_1"] else P["ink"],
                zorder=4,
            )
    _box(
        ax,
        cx[2],
        y2,
        bw,
        h2,
        "6",
        "Relax",
        ["FIRE, all atoms free", r"$f_\mathrm{max}$ = 10$^{-6}$", "same for every method"],
        *grey,
    )
    _box(
        ax,
        cx[3],
        y2,
        bw,
        h2,
        "7",
        "Evaluate",
        [
            r"success: $c$ within 0.5σ of $p$",
            r"local W1, $r_\mathrm{loc}$ = 3σ",
            "256 samples × 3 seeds",
            "bootstrap 95% CIs",
        ],
        *grey,
    )
    # lane arrows
    for k in range(3):
        if k < 2:
            _block_arrow(ax, cx[k] + bw + 0.02, y1 + h1 / 2, cx[k + 1] - 0.02, y1 + h1 / 2)
        _block_arrow(ax, cx[k] + bw + 0.02, y2 + h2 / 2, cx[k + 1] - 0.02, y2 + h2 / 2)
    # between the lanes: host glasses, the trained model, the natural floor
    mid = (y2 + h2 + y1) / 2
    _block_arrow(ax, cx[0] + 0.3, y1 - 0.02, cx[0] + 0.3, y2 + h2 + 0.02, fc=P["green_2"])
    ax.text(
        cx[0] + 0.4,
        mid,
        "test glasses",
        fontsize=6,
        style="italic",
        va="center",
        color=P["green_dark"],
    )
    xs, xe = cx[2] + bw / 2, cx[1] + bw - 0.3
    ax.plot(
        [xs, xs, xe],
        [y1 - 0.01, mid, mid],
        color=P["blue_main"],
        lw=1.0,
        zorder=2,
        solid_capstyle="butt",
    )
    ax.annotate(
        "",
        xy=(xe, y2 + h2 + 0.01),
        xytext=(xe, mid + 0.005),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.18,head_length=0.35",
            color=P["blue_main"],
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    ax.text(
        (xs + xe) / 2,
        mid + 0.04,
        r"trained $d_\theta$",
        fontsize=6,
        style="italic",
        ha="center",
        va="bottom",
        color=P["blue_main"],
    )
    _block_arrow(ax, cx[3] + bw / 2, y1 - 0.02, cx[3] + bw / 2, y2 + h2 + 0.02, fc=P["neutral"])
    ax.text(
        cx[3] + bw / 2 + 0.1,
        mid,
        "floors",
        fontsize=6,
        style="italic",
        va="center",
        color=P["grey_dark"],
    )
    ax.set_xlim(0, 5.5)
    ax.set_ylim(-0.01, y1 + h1 + 0.01)
    ax.set_aspect("equal")
    ax.axis("off")


def _tag(ax, xy, text, offset, color=None):
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        fontsize=6,
        ha="center",
        va="center",
        color=color or P["ink"],
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
        arrowprops=dict(arrowstyle="-", color=P["ink"], lw=0.5, shrinkA=0, shrinkB=2.5),
    )


def _letter(fig, x, y, letter):
    """Panel letter with its top-left corner at (x, y) inches."""
    W, H = fig.get_size_inches()
    fig.text(x / W, y / H, letter, fontsize=9, fontweight="bold", va="top", ha="left")


def figure1() -> None:
    fig = plt.figure(figsize=(5.5, 3.72))
    _workflow(_inches(fig, 0.0, 1.78, 5.5, 1.76))
    _letter(fig, -0.06, 3.72, "a")

    test = np.load("data/ka2d_256/test.npz")
    g, i_minus, i_plus = NATURAL
    pos, types, box, lab = test["pos"][g], test["types"][g], test["box"][g], test["labels"][g]
    # (b) natural defects: window around a D1- / D1+ pair
    ax = _inches(fig, 0.12, 0.08, 1.38, 1.38)
    c = np.mod(pos[i_minus] + _min_image(pos[i_plus] - pos[i_minus], box) / 2, box)
    half = 3.9
    fs.draw_atoms(ax, pos, types, box, c, half, rings=lab > 0)
    tags = {1: r"D1$^{-}$", 2: r"D1$^{+}$", 3: "D2"}
    done = set()
    for j in np.argsort(np.linalg.norm(_min_image(pos - c, box), axis=1)):
        k = int(lab[j])
        if k and k not in done:
            p = c + _min_image(pos[j] - c, box)
            u = (p - c) / max(np.linalg.norm(p - c), 1e-6)
            _tag(ax, p, tags[k], tuple(17 * u + np.array([0, 3])))
            done.add(k)
    ax.set_title("Natural defects (test glass)", fontsize=7, pad=3)
    _letter(fig, -0.06, 1.62, "b")

    ex = np.load(DATA / "examples_D2_model.npz")
    e = EXAMPLE
    host, types, box = ex["host"][e], ex["types"][e], ex["box"][e]
    target, pin = ex["target"][e], ex["pin"][e]
    # (c) Task B request on the real host, viewed centred on the site
    ax = _inches(fig, 1.68, 0.08, 1.38, 1.38)
    fs.draw_atoms(
        ax, np.where(pin[:, None], host, np.nan), types, box, target, box[0] / 2, pale=pin
    )
    ax.add_patch(Circle(target, 4.0, fill=False, ec=P["ink"], lw=0.9, zorder=4))
    ax.add_patch(
        Circle(target, 3.0, fill=False, ec=P["blue_main"], lw=0.8, ls=(0, (3, 2)), zorder=4)
    )
    ax.plot(*target, marker="+", ms=6, mew=1.2, color=P["red_strong"], zorder=5)
    ax.text(
        target[0],
        target[1] - 0.75,
        r"site $p$ (D2)",
        fontsize=5.5,
        ha="center",
        va="top",
        color=P["red_strong"],
        zorder=5,
        linespacing=1.0,
    )
    for r, text, ang, col in (
        (4.0, r"$R_\mathrm{out}$ = 4σ", 50, P["ink"]),
        (3.0, r"$r_\mathrm{loc}$ = 3σ", 140, P["blue_main"]),
    ):
        a = np.deg2rad(ang)
        _tag(
            ax,
            target + r * np.array([np.cos(a), np.sin(a)]),
            text,
            (14 if ang < 90 else -14, 12),
            color=col,
        )
    ax.set_title("Task B request (pale: pinned)", fontsize=7, pad=3)
    _letter(fig, 1.5, 1.62, "c")

    # (d) sampling trajectory and relaxed result
    steps = list(ex["snap_steps"])
    frames = [ex["init"][e]] + [ex["snapshots"][e, steps.index(k)] for k in SNAP_STEPS[1:]]
    frames.append(ex["relaxed"][e])
    sig = np.concatenate([[1.0], ex["sigmas"], np.full(100, ex["sigmas"][-1])])  # sigma at k
    x0, tw, tg = 3.24, 0.53, 0.04
    axs = _inches(fig, x0 + 0.22, 1.02, 4 * tw + 3 * tg - 0.24, 0.36)
    axs.fill_between(np.arange(101), 0, sig[:101], color=P["blue_1"], lw=0)
    axs.plot(np.arange(201), sig, color=P["blue_main"], lw=1.1)
    axs.set_xlim(0, 200)
    axs.set_ylim(0, 1.05)
    axs.set_xticks([0, 100, 200], labels=[])
    axs.set_yticks([0, 1])
    axs.set_ylabel(r"$\sigma_k$", fontsize=6, labelpad=1)
    axs.tick_params(labelsize=5.5, pad=1)
    axs.text(52, 0.62, "noisy", fontsize=5.5, color=P["blue_main"])
    axs.text(150, 0.2, "noise-free", fontsize=5.5, color=P["blue_main"], ha="center")
    ok = bool(ex["success"][e])
    labels = [f"k = {k}" for k in SNAP_STEPS] + ["relaxed ✓" if ok else "relaxed ✗"]
    for j, frame in enumerate(frames):
        ax = _inches(fig, x0 + j * (tw + tg), 0.2, tw, tw)
        rings = None
        if j == len(frames) - 1:
            lab = ex["labels_relaxed"][e]
            d = np.linalg.norm(_min_image(frame - target, box), axis=1)
            near = np.argmin(np.where(types == 1, d, np.inf))
            rings = np.zeros(len(frame), bool)
            rings[near] = lab[near] == 3
        fs.draw_atoms(
            ax, frame, types, box, target, 4.6, pale=pin, rings=rings, edge_lw=0.2, ring_lw=0.8
        )
        ax.add_patch(Circle(target, 4.0, fill=False, ec=P["ink"], lw=0.5, zorder=4))
        ax.plot(*target, marker="+", ms=4, mew=0.9, color=P["red_strong"], zorder=5)
        last = j == len(frames) - 1
        ax.set_xlabel(
            labels[j],
            fontsize=6,
            labelpad=1.5,
            color=(P["green_dark"] if ok else P["red_strong"]) if last else P["ink"],
            fontweight="bold" if last else "normal",
        )
        k = SNAP_STEPS[j] if j < len(SNAP_STEPS) else 200
        fig.add_artist(
            ConnectionPatch(
                xyA=(k, sig[k]),
                coordsA=axs.transData,
                xyB=(0.5, 1.0),
                coordsB=ax.transAxes,
                color=P["grey"],
                lw=0.5,
                ls=(0, (2, 1.5)),
                zorder=0,
            )
        )
        axs.plot(k, sig[k], "o", ms=2.5, color=P["blue_main"], zorder=5)
    _letter(fig, 3.06, 1.62, "d")
    fig.text(
        (x0 + 2 * tw + 1.5 * tg) / 5.5,
        1.55 / 3.72,
        r"Sampler noise $\sigma_k$ over steps $k$, then FIRE",
        fontsize=7,
        ha="center",
        va="center",
    )
    fs.finalize_figure(fig, OUT / "fig1_overview")


# --------------------------------------------------------------------------- Figure 2


def figure2() -> None:
    parts = [
        ("model", "Conditional model, w = 6"),
        ("hand", "Hand insertion"),
        ("lmq", "Local melt-quench"),
    ]
    data = {p: np.load(DATA / f"examples_D2_{p}.npz") for p, _ in parts}
    e = EXAMPLE
    fig = plt.figure(figsize=(5.5, 2.05))
    w, gap = 1.62, 0.21
    for j, (p, title) in enumerate(parts):
        d = data[p]
        types, box, target, pin = d["types"][e], d["box"][e], d["target"][e], d["pin"][e]
        pos, lab = d["relaxed"][e], d["labels_relaxed"][e]
        ax = _inches(fig, 0.1 + j * (w + gap), 0.32, w, w)
        # held fixed while generating: the model pins the labelled site atom and the host
        # beyond R_out; local melt-quench freezes only the host; hand insertion pins nothing
        if p == "model":
            free = ~pin
        elif p == "lmq":
            free = np.linalg.norm(_min_image(d["host"][e] - target, box), axis=1) <= 4.0
        else:
            free = np.ones(len(pos), bool)
        fs.draw_atoms(ax, pos, types, box, target, 5.2, pale=~free, rings=lab > 0)
        if p != "hand":
            ax.add_patch(Circle(target, 4.0, fill=False, ec=P["ink"], lw=0.7, zorder=4))
        ax.add_patch(
            Circle(target, 0.5, fill=False, ec=P["red_strong"], lw=0.8, ls=(0, (2, 1)), zorder=5)
        )
        ax.plot(*target, marker="+", ms=6, mew=1.2, color=P["red_strong"], zorder=5)
        ok = bool(d["success"][e])
        ax.set_title(
            title, fontsize=7, pad=3, color=P["blue_main"] if p == "model" else P["red_strong"]
        )
        ax.text(
            0.5,
            -0.05,
            "success: D2 at p" if ok else "miss: no D2 at p",
            fontsize=6.5,
            ha="center",
            va="top",
            transform=ax.transAxes,
            color=P["green_dark"] if ok else P["red_strong"],
            fontweight="bold",
        )
        fig.text(
            (0.1 + j * (w + gap) - 0.02) / 5.5,
            (0.32 + w + 0.02) / 2.05,
            "abc"[j],
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
    handles = [
        Line2D(
            [], [], ls="", marker="o", ms=5, mfc=fs.SPECIES[0], mec=P["ink"], mew=0.3, label="A"
        ),
        Line2D(
            [], [], ls="", marker="o", ms=4.4, mfc=fs.SPECIES[1], mec=P["ink"], mew=0.3, label="B"
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=5,
            mfc=fs.SPECIES_PALE[0],
            mec=P["grey"],
            mew=0.3,
            label="held while generating",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            ms=6,
            mfc="none",
            mec=P["ink"],
            mew=0.9,
            label="defect (any class)",
        ),
        Line2D(
            [],
            [],
            ls="",
            marker="+",
            ms=6,
            mew=1.2,
            color=P["red_strong"],
            label=r"requested site, $r_\mathrm{tol}$ = 0.5σ",
        ),
        Line2D([], [], color=P["ink"], lw=0.7, label=r"$R_\mathrm{out}$ = 4σ"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=6,
        handletextpad=0.3,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fs.finalize_figure(fig, OUT / "fig2_taskB")


# --------------------------------------------------------------------------- Figure 3


SEED_MARKER = {0: "o", 1: "s", 2: "^"}


def _point(ax, m, style, marker, ms=4.2):
    s, w = m["success_after_relax"], m["local"]["w1_pe_atom"]
    ax.errorbar(
        w[0],
        s[0],
        xerr=[[w[0] - w[1]], [w[2] - w[0]]],
        yerr=[[s[0] - s[1]], [s[2] - s[0]]],
        marker=marker,
        ms=ms,
        ls="none",
        capsize=1.5,
        elinewidth=0.7,
        mew=0.8,
        color=style["color"],
        mfc=style["mfc"],
        mec=style["color"] if style["mfc"] == "white" else P["ink"],
        zorder=3,
    )


def figure3() -> None:
    seeds = _script("plot_seeds")
    fig = plt.figure(figsize=(5.5, 2.05))
    axes = [_inches(fig, 0.42 + k * 1.75, 0.62, 1.4, 1.22) for k in range(3)]
    titles = ["Task B, D2", r"Task B, D1$^{-}$", "Task A, D2"]
    for k, (a, (_, defect, model_rows, baselines)) in enumerate(zip(axes, seeds.PANELS)):
        floor = json.loads(Path(f"results/sensitivity/floor_{defect}.json").read_text())
        lo, hi = floor["variants"]["default"]["local_w1_pe_atom"][1:]
        a.axvspan(lo, hi, color=P["neutral"], alpha=0.6, lw=0, zorder=0)
        for label, row in model_rows.items():
            for seed in (0, 1, 2):
                name = row if seed == 0 else f"seed{seed}_{row}"
                _point(a, seeds.metrics(name), fs.METHOD[f"model, {label}"], SEED_MARKER[seed])
        for label, row in baselines.items():
            _point(a, seeds.metrics(row), fs.METHOD[label], "D")
        a.set_title(titles[k], fontsize=7, pad=3)
        a.set_xlabel("local W1 (lower = more natural)", fontsize=6.5, labelpad=1.5)
        a.set_xlim(left=0)
        a.set_ylim(0, 1)
        a.set_yticks([0, 0.5, 1])
        fig.text(
            (0.42 + k * 1.75 - 0.3) / 5.5,
            (0.62 + 1.22 + 0.04) / 2.05,
            "abc"[k],
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    axes[0].set_ylabel("success after relaxation", fontsize=6.5, labelpad=2)
    handles = (
        [
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                ms=4.2,
                mew=0.8,
                label=label,
                color=st["color"],
                mfc=st["mfc"],
                mec=st["color"] if st["mfc"] == "white" else P["ink"],
            )
            for label, st in fs.METHOD.items()
        ]
        + [Patch(fc=P["neutral"], alpha=0.6, label="natural floor (95% CI)")]
        + [
            Line2D(
                [],
                [],
                ls="",
                marker=SEED_MARKER[s],
                ms=4.2,
                mfc=P["grey"],
                mec=P["ink"],
                mew=0.5,
                label=f"seed {s}",
            )
            for s in (0, 1, 2)
        ]
    )
    for h in handles[2:4]:
        h.set_marker("D")
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=8,
        fontsize=6,
        handletextpad=0.2,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fs.finalize_figure(fig, OUT / "fig3_seeds")


# --------------------------------------------------------------------------- Figure 4


SENS_MARKER = {
    "model, w = 2": "o",
    "model, w = 6": "s",
    "hand insertion": "^",
    "local melt-quench": "D",
}


def _series(ax, x, rows, label, offset):
    st = fs.METHOD[label]
    est = [r[0] for r in rows]
    err = [[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]]
    ax.errorbar(
        [v + offset for v in x],
        est,
        yerr=err,
        capsize=1.5,
        lw=1.0,
        elinewidth=0.7,
        marker=SENS_MARKER[label],
        ms=3.8,
        mew=0.8,
        color=st["color"],
        mfc=st["mfc"],
        mec=st["color"] if st["mfc"] == "white" else P["ink"],
        zorder=3,
    )


def _floor(ax, x, rows):
    ax.fill_between(
        x, [r[1] for r in rows], [r[2] for r in rows], color=P["neutral"], alpha=0.6, lw=0, zorder=0
    )
    ax.plot(x, [r[0] for r in rows], color=P["grey"], lw=0.8, ls="--", zorder=1)


def figure4() -> None:
    sens = _script("plot_sensitivity")
    fig = plt.figure(figsize=(5.5, 3.55))
    W, H, x_l, x_r, y_b, y_t = 2.2, 1.12, 0.45, 3.1, 0.52, 2.12
    ax = [
        _inches(fig, x_l, y_t, W, H),
        _inches(fig, x_r, y_t, W, H),
        _inches(fig, x_l, y_b, W, H),
        _inches(fig, x_r, y_b, W, H),
    ]
    for k, (defect, key, values) in enumerate(
        (("D2", "d2_rc", (1.15, 1.2, 1.25)), ("D1_MINUS", "cn_factor", (1.35, 1.4, 1.45)))
    ):
        for j, (label, run) in enumerate(sens.RUNS[defect].items()):
            v = sens.load(run.format(4), "sensitivity.json")["variants"]
            names = [f"{key}={x:g}" if x != values[1] else "default" for x in values]
            _series(
                ax[k],
                values,
                [v[n]["success_after_relax"] for n in names],
                label,
                (j - 1.5) * 0.004,
            )
        ax[k].axvline(values[1], color=P["neutral"], lw=0.8, zorder=0)
        ax[k].set_xticks(values)
        ax[k].set_ylim(0, 1)
        ax[k].set_title(
            "D2: B with ≥ 2 B neighbours" if defect == "D2" else r"D1$^{-}$: A with CN ≤ 5",
            fontsize=7,
            pad=3,
        )
        ax[k].set_xlabel(
            "B–B cutoff (σ)" if defect == "D2" else "coordination cutoff factor",
            fontsize=6.5,
            labelpad=1.5,
        )
    ax[0].set_ylabel("Task B success", fontsize=6.5)
    radii = (2.5, 3.0, 3.5)
    floor = sens.load_floor("D2")
    _floor(ax[2], radii, [floor[f"r_loc={r:g}" if r != 3.0 else "default"] for r in radii])
    for j, (label, run) in enumerate(sens.RUNS["D2"].items()):
        v = sens.load(run.format(4), "sensitivity.json")["variants"]
        rows = [v[f"r_loc={r:g}" if r != 3.0 else "default"]["local_w1_pe_atom"] for r in radii]
        _series(ax[2], radii, rows, label, (j - 1.5) * 0.02)
    ax[2].set_xticks(radii)
    ax[2].set_title(r"D2 surroundings vs window ($R_\mathrm{out}$ = 4σ)", fontsize=7, pad=3)
    ax[2].set_xlabel(r"realism window $r_\mathrm{loc}$ (σ)", fontsize=6.5, labelpad=1.5)
    ax[2].set_ylabel("local W1", fontsize=6.5)
    r_outs = (3, 4, 5)
    _floor(ax[3], r_outs, [floor["default"]] * 3)
    for j, (label, run) in enumerate(sens.RUNS["D2"].items()):
        if "{}" in run:
            rows = [sens.load(run.format(r), "metrics.json")["local"]["w1_pe_atom"] for r in r_outs]
            _series(ax[3], r_outs, rows, label, (j - 1.5) * 0.04)
        else:
            w = sens.load(run, "metrics.json")["local"]["w1_pe_atom"]
            ax[3].axhline(w[0], color=fs.METHOD[label]["color"], lw=1.0, ls=":", zorder=1)
            ax[3].text(
                5.18,
                w[0] + 0.003,
                "hand insertion (no disk)",
                fontsize=6,
                ha="right",
                va="bottom",
                color=fs.METHOD[label]["color"],
            )
    ax[3].set_xticks(r_outs)
    ax[3].set_xlim(2.8, 5.2)
    ax[3].set_title(r"D2 surroundings vs disk ($r_\mathrm{loc}$ = 3σ)", fontsize=7, pad=3)
    ax[3].set_xlabel(r"regenerated disk radius $R_\mathrm{out}$ (σ)", fontsize=6.5, labelpad=1.5)
    for k, a in enumerate(ax):
        a.set_ylim(bottom=0)
        bb = a.get_position()
        fig.text(
            bb.x0 - 0.075, bb.y1 + 0.012, "abcd"[k], fontsize=9, fontweight="bold", va="bottom"
        )
    handles = [
        Line2D(
            [],
            [],
            marker=SENS_MARKER[label],
            ms=3.8,
            mew=0.8,
            lw=1.0,
            label=label,
            color=st["color"],
            mfc=st["mfc"],
            mec=st["color"] if st["mfc"] == "white" else P["ink"],
        )
        for label, st in fs.METHOD.items()
    ] + [Patch(fc=P["neutral"], alpha=0.6, label="natural floor (95% CI)")]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=6,
        handletextpad=0.4,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, -0.01),
    )
    fs.finalize_figure(fig, OUT / "fig4_sensitivity")


def main() -> None:
    fs.apply_publication_style()
    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for k in which:
        {"1": figure1, "2": figure2, "3": figure3, "4": figure4}[k]()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
