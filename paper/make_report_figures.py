"""Figures of the point-defect report (paper/point_defects_report.tex).

    python paper/make_report_figures.py            # all figures
    python paper/make_report_figures.py 1 3        # only some

    1  figures/pd_fig1_defects.pdf    the four point defects: the operation, and the glass
                                      after relaxation (real test examples)
    2  figures/pd_fig2_pipeline.pdf   find, fix, minimise, compare, on one test vacancy
    3  figures/pd_fig3_results.pdf    found and restored, learned finder against the baselines

Figure 2 needs the trained finder once; its example is cached in data/pd_report_example.npz.
Figure 3 reads results/point_defects_recheck_20261111/. Colours: a validated colour-blind-safe
set (atoms: light blue and yellow, also different in size; methods: blue, green, orange).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import apply_publication_style, finalize_figure, panel_letter  # noqa: E402

from glassdiff.data.point_defects import (  # noqa: E402
    AtomRole,
    PointDefect,
    largest_gap,
    load_point_defects,
    real_defects,
    wrap,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/figures"
RECHECK = ROOT / "results/point_defects_recheck_20261111"
FINDER = ROOT / "results/kaggle_pd_20261110-1042Z/training/pd_detect_v2/ema.pt"
EXAMPLE = ROOT / "paper/data/pd_report_example.npz"

INK, MUTED, GRID = "#272727", "#767676", "#E1E0D9"
ATOM = ("#6DA7EC", "#EDA100")  # A (large), B (small)
DIAMETER = (1.0, 0.88)
METHOD = {  # fixed order: learned finder, the two structural measures, energy
    "finder": ("#2A78D6", "Learned finder"),
    "geometry": ("#1BAF7A", "Cell size"),
    "bonds": ("#6250D6", "Unlike neighbours"),
    "energy": ("#EB6834", "Atom energy (uses the potential)"),
}
NAMES = {
    "vacancy": "Vacancy",
    "interstitial": "Interstitial",
    "frenkel": "Frenkel pair",
    "antisite": "Antisite",
}
BLUES = LinearSegmentedColormap.from_list(
    "prob", ["#F3F7FD", "#CDE2FB", "#6DA7EC", "#2A78D6", "#104281"]
)


def images(xy: np.ndarray, box: np.ndarray, centre: np.ndarray, half: float) -> np.ndarray:
    """Periodic images of the points that fall in the window (as offsets-applied copies)."""
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * box
    out = (np.mod(xy, box)[None] + shifts[:, None]).reshape(-1, 2)
    idx = np.tile(np.arange(len(xy)), len(shifts))
    keep = (np.abs(out - centre) <= half + 0.7).all(1)
    return out[keep], idx[keep]


def draw(ax, pos, types, box, centre, half, face=None, lw=0.35) -> None:
    """Atoms at their physical size in a window around ``centre``."""
    xy, idx = images(pos, box, centre, half)
    d = np.array(DIAMETER)[types[idx]]
    colours = [ATOM[t] for t in types[idx]] if face is None else [face[i] for i in idx]
    ax.add_collection(
        EllipseCollection(
            d,
            d,
            np.zeros(len(xy)),
            units="xy",
            offsets=xy,
            offset_transform=ax.transData,
            facecolors=colours,
            edgecolors=INK,
            linewidths=lw,
            zorder=2,
        )  # fmt: skip
    )
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.5)


def near(p: np.ndarray, box: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """The periodic image of point(s) p nearest to the window centre."""
    return centre + wrap(np.mod(p, box) - centre, box)


def ring(ax, p, diameter, style="-", lw=1.1, colour=INK, z=4) -> None:
    ax.add_patch(
        Circle(p, diameter / 2, fill=False, ec=colour, lw=lw, ls=style, zorder=z, clip_on=True)
    )


def arrow(ax, a, b, lw=0.7, colour=INK, head=3.0, z=5) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=f"-|>,head_length={head * 0.6},head_width={head * 0.3}",
            lw=lw,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=1.6,
            zorder=z,
            clip_on=True,
        )  # fmt: skip
    )


def moved_atoms(d, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(original positions, relaxed positions, displacement size) of the atoms that exist in
    both glasses, in the order of the relaxed glass minus any extra atom."""
    orig, pos, box = d.orig_pos[e].numpy(), d.pos[e].numpy(), d.box[e].numpy()
    a = int(d.atom[e])
    if d.kind == PointDefect.VACANCY:
        ref, cur = np.delete(orig, a, 0), pos
    elif d.kind == PointDefect.INTERSTITIAL:
        ref, cur = orig, pos[:-1]
    else:
        ref, cur = orig, pos
    return ref, cur, np.linalg.norm(wrap(cur - ref, box), axis=-1)


def typical_example(d) -> int:
    """The example whose count of atoms moved by more than 0.1 sigma is the median one."""
    counts = []
    for e in range(len(d)):
        _, _, disp = moved_atoms(d, e)
        if d.kind == PointDefect.FRENKEL:
            disp = np.delete(disp, int(d.atom[e]))
        counts.append(int((disp > 0.1).sum()))
    counts = np.array(counts)
    return int(np.argmin(np.abs(counts - np.median(counts))))


def fig_defects() -> None:
    data = real_defects(load_point_defects(ROOT / "data/ka2d_256_point_defects/test.npz"))
    half = 2.7
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.95))
    fig.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.075, wspace=0.05, hspace=0.06)
    for col, (name, d) in enumerate(data.items()):
        e = typical_example(d)
        box = d.box[e].numpy()
        orig, otypes = d.orig_pos[e].numpy(), d.orig_types[e].numpy()
        pos, types = d.pos[e].numpy(), d.types[e].numpy()
        a = int(d.atom[e])
        site = np.mod(d.sites[e, 0].numpy(), box)
        if d.kind == PointDefect.FRENKEL:  # show the empty site and the moved atom
            end = near(d.sites[e, 1].numpy(), box, site)
            centre = site + 0.5 * (end - site)
        else:
            centre = site
        top, bottom = axes[0, col], axes[1, col]
        draw(top, orig, otypes, box, centre, half)
        draw(bottom, pos, types, box, centre, half)
        top.set_title(NAMES[name], fontsize=7.5, pad=3)
        # --- the operation, on the original glass
        if d.kind == PointDefect.VACANCY:
            p = near(orig[a], box, centre)
            ring(top, p, DIAMETER[otypes[a]] + 0.22)
            top.plot(*p, marker="x", ms=4.5, mew=1.1, color=INK, zorder=6)
        elif d.kind == PointDefect.INTERSTITIAL:
            p = near(pos[-1], box, centre)
            top.plot(*p, marker="+", ms=6, mew=1.3, color=INK, zorder=6)
        elif d.kind == PointDefect.FRENKEL:
            p = near(orig[a], box, centre)
            ring(top, p, DIAMETER[otypes[a]] + 0.22)
            arrow(top, p, near(pos[a], box, centre), lw=1.0, head=4.5)
        else:
            a2 = int(d.atom2[e])
            for i in (a, a2):
                ring(top, near(orig[i], box, centre), DIAMETER[otypes[i]] + 0.22)
        # --- after relaxation: how far each atom moved, and what is left of the defect
        ref, cur, disp = moved_atoms(d, e)
        for i in np.nonzero(disp > 0.1)[0]:
            if d.kind == PointDefect.FRENKEL and i == a:
                continue
            p1 = near(cur[i], box, centre)
            arrow(bottom, p1 - wrap(cur[i] - ref[i], box), p1, lw=0.55, head=2.6)
        if d.kind in (PointDefect.VACANCY, PointDefect.FRENKEL):
            ring(
                bottom,
                near(orig[a], box, centre),
                DIAMETER[otypes[a]],
                style=(0, (1.2, 1.2)),
                lw=0.9,
            )
        if d.kind == PointDefect.INTERSTITIAL:
            ring(bottom, near(pos[-1], box, centre), DIAMETER[1] + 0.22)
        elif d.kind == PointDefect.FRENKEL:
            ring(bottom, near(pos[a], box, centre), DIAMETER[types[a]] + 0.22)
        elif d.kind == PointDefect.ANTISITE:
            for i in (a, int(d.atom2[e])):
                ring(bottom, near(pos[i], box, centre), DIAMETER[types[i]] + 0.22)
    axes[0, 0].set_ylabel("Original glass,\nwith the operation", fontsize=6.5, labelpad=3)
    axes[1, 0].set_ylabel("After relaxation", fontsize=6.5, labelpad=3)
    # key, drawn by hand (in inches) so that the empty site keeps its dotted outline
    w, h = fig.get_size_inches()
    key = fig.add_axes([0.0, 0.0, 1.0, 0.062])
    key.set_xlim(0, w)
    key.set_ylim(0, 0.062 * h)
    key.axis("off")
    x, y, r = 0.42, 0.031 * h, 0.062
    for kind, label in (
        ("A", "A atom (large)"),
        ("B", "B atom (small)"),
        ("ring", "atom removed, added, moved or swapped"),
        ("site", "empty site"),
        ("arrow", r"moved by more than 0.1$\sigma$"),
    ):
        if kind in ("A", "B"):
            t = "AB".index(kind)
            key.add_patch(Circle((x, y), r * DIAMETER[t], fc=ATOM[t], ec=INK, lw=0.35))
        elif kind == "ring":
            key.add_patch(Circle((x, y), r * 1.15, fill=False, ec=INK, lw=1.1))
        elif kind == "site":
            key.add_patch(Circle((x, y), r, fill=False, ec=INK, lw=0.9, ls=(0, (1.2, 1.2))))
        else:
            arrow(key, (x - 0.09, y), (x + 0.09, y), lw=0.7, head=3.2)
        txt = key.text(x + 0.12, y, label, va="center", ha="left", fontsize=6)
        fig.canvas.draw()
        x = key.transData.inverted().transform(txt.get_window_extent().get_points())[1, 0] + 0.24
    finalize_figure(fig, FIG / "pd_fig1_defects")


def make_example() -> None:
    """One test vacancy that the finder locates and the repair restores, with every step of
    the repair, saved to EXAMPLE (the finder is only needed here)."""
    from glassdiff.models.registry import load_denoiser
    from glassdiff.physics.fire import fire_minimize
    from glassdiff.types import Structures

    d = load_point_defects(ROOT / "data/ka2d_256_point_defects/test.npz")["vacancy"]
    ckpt = torch.load(FINDER, map_location="cpu", weights_only=False)
    model = load_denoiser(str(FINDER), torch.device("cpu"))
    log_w = torch.log(torch.tensor(ckpt["class_weights"]))
    counts = []
    for e in range(len(d)):
        counts.append(int((moved_atoms(d, e)[2] > 0.1).sum()))
    order = np.argsort(np.abs(np.array(counts) - np.median(counts)), kind="stable")
    for e in order:  # the most typical example on which the whole procedure succeeds
        e = int(e)
        pos, types, box = d.pos[e].numpy(), d.types[e].numpy(), d.box[e].numpy()
        s = Structures(d.pos[e : e + 1].float(), d.types[e : e + 1], d.box[e : e + 1].float())
        with torch.no_grad():
            prob = (model(s) - log_w).softmax(-1)[0, :, int(AtomRole.NEAR_VACANCY)].numpy()
        flag = int(prob.argmax())
        site = d.sites[e, 0].numpy()
        if np.linalg.norm(wrap(pos[flag] - site, box)) >= 1.5:
            continue
        gap = largest_gap(pos, box, pos[flag], 0.0, 1.5)
        if np.linalg.norm(wrap(gap - site, box)) > 0.5:
            continue  # for the picture: the open point is at the site, not a neighbouring one
        a = int(d.atom[e])
        full = np.insert(pos, a, gap, axis=0)
        otypes = d.orig_types[e].numpy()
        out = fire_minimize(
            Structures(torch.as_tensor(full)[None], d.orig_types[e : e + 1], d.box[e : e + 1]),
            fmax=1e-6,
        )
        final = out.structures.pos[0].numpy()
        orig = d.orig_pos[e].numpy()
        from glassdiff.data.point_defects import same_structure

        if not same_structure(final, otypes, orig, otypes, box):
            continue
        EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(EXAMPLE, example=e, pos=pos, types=types, box=box, prob=prob, flag=flag,
                 site=site, atom=a, gap=gap, full=full, final=final, orig=orig, otypes=otypes,
                 moved=counts[e])  # fmt: skip
        print(f"example {e}: {counts[e]} atoms displaced; flagged atom "
              f"{np.linalg.norm(wrap(pos[flag] - site, box)):.2f} sigma from the site")  # fmt: skip
        return
    raise RuntimeError("no example found")


def fig_pipeline() -> None:
    if not EXAMPLE.exists():
        make_example()
    z = np.load(EXAMPLE)
    box, site, half = z["box"], np.mod(z["site"], z["box"]), 3.1
    fig = plt.figure(figsize=(5.5, 1.72))
    width, gap_w, left, bottom, height = 0.212, 0.036, 0.012, 0.17, 0.70
    axes = [fig.add_axes([left + k * (width + gap_w), bottom, width, height]) for k in range(4)]
    titles = (
        "Glass with a vacancy",
        "1. Find",
        "2. Correct the count",
        "3. Minimise the energy",
    )
    for k, (ax, title) in enumerate(zip(axes, titles)):
        x_left, y_top = ax.get_position().x0, ax.get_position().y1 + 0.025
        fig.text(x_left, y_top, "abcd"[k], fontsize=8, fontweight="bold", ha="left", va="bottom")
        fig.text(x_left + 0.022, y_top, title, fontsize=6.8, ha="left", va="bottom")
    # (a) the relaxed defected glass; the empty site is not known to the methods
    draw(axes[0], z["pos"], z["types"], box, site, half)
    ring(axes[0], site, DIAMETER[z["otypes"][z["atom"]]], style=(0, (1.2, 1.2)), lw=0.9)
    # (b) finder probability per atom
    top = float(z["prob"].max())
    colours = [BLUES(v / top) for v in z["prob"]]
    draw(axes[1], z["pos"], z["types"], box, site, half, face=colours)
    ring(axes[1], near(z["pos"][z["flag"]], box, site), DIAMETER[z["types"][z["flag"]]] + 0.24)
    x0 = axes[1].get_position().x0
    cax = fig.add_axes([x0 + 0.062, 0.088, 0.09, 0.032])
    cax.imshow(np.linspace(0, 1, 64)[None], cmap=BLUES, aspect="auto", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([])
    for sp in cax.spines.values():
        sp.set_linewidth(0.4)
        sp.set_color(INK)
    fig.text(x0 + 0.056, 0.104, "0", fontsize=5.5, ha="right", va="center")
    fig.text(x0 + 0.158, 0.104, f"{top:.2f}", fontsize=5.5, ha="left", va="center")
    fig.text(
        x0 + 0.107,
        0.0,
        "probability: next to an empty site",
        fontsize=5.5,
        ha="center",
        va="bottom",
    )
    # (c) the missing atom added at the most open point near the flagged atom
    draw(axes[2], z["full"], z["otypes"], box, site, half)
    ring(axes[2], near(z["gap"], box, site), DIAMETER[z["otypes"][z["atom"]]] + 0.24)
    # (d) after minimisation. Dots in (c) and (d): where each atom sits in the original glass
    draw(axes[3], z["final"], z["otypes"], box, site, half)
    for ax in axes[2:]:
        xy, _ = images(z["orig"], box, site, half)
        ax.scatter(xy[:, 0], xy[:, 1], s=2.2, color="black", linewidths=0, zorder=6)
    x2 = axes[2].get_position().x0
    fig.text(x2 + width + gap_w / 2, 0.085, "dots: atom positions in the original glass",
             fontsize=5.5, ha="center", va="center")  # fmt: skip
    for k in range(3):  # arrows between the panels
        xa = axes[k].get_position().x1 + 0.004
        fig.patches.append(
            FancyArrowPatch(
                (xa, bottom + height / 2),
                (xa + gap_w - 0.008, bottom + height / 2),
                transform=fig.transFigure,
                arrowstyle="-|>,head_length=3,head_width=1.8",
                lw=0.9,
                color=INK,
            )  # fmt: skip
        )
    finalize_figure(fig, FIG / "pd_fig2_pipeline")


def rounded_bar(ax, x: float, height: float, width: float, colour: str) -> None:
    """A bar with a rounded top and a square foot on the baseline."""
    if height <= 0:
        return
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    aspect = (x1 - x0) / (y1 - y0)  # data units of y per data unit of x, on the page
    r = 0.28 * width
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, -5.0),
            width,
            height + 5.0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            mutation_aspect=aspect,
            fc=colour,
            ec="none",
            zorder=3,
            clip_on=True,
        )  # fmt: skip
    )


def fig_results() -> None:
    res = json.loads((ROOT / "paper/data/pd_report_results.json").read_text())["kinds"]
    kinds = ("vacancy", "interstitial", "antisite", "frenkel")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.8, bottom=0.11, wspace=0.2)
    step, width = 0.2, 0.165
    for ax, what, title in zip(
        axes, ("found", "restored"), ("Defects found", "Original glasses restored")
    ):
        ax.set_xlim(-0.78, 3.55)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(4), [NAMES[k] for k in kinds])
        ax.tick_params(axis="x", length=0, pad=3)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#C3C2B7")
        ax.set_title(title, fontsize=7.5, pad=4, loc="left")
        ax.set_ylabel("% of test defects", fontsize=6.5, labelpad=2)
        fig.canvas.draw()
        for i, k in enumerate(kinds):
            for j, m in enumerate(METHOD):
                d = res[k][what][m]
                x = i + (j - 1.5) * step
                rounded_bar(ax, x, 100 * d["rate"], width, METHOD[m][0])
                ax.plot([x, x], [100 * d["lo"], 100 * d["hi"]], color=INK, lw=0.6, zorder=4,
                        solid_capstyle="butt")  # fmt: skip
                if m == "finder":  # direct label on the method the figure is about
                    top = 100 * d["hi"] + 1.5
                    line = (
                        100 * res[k]["restored"]["ceiling"]["rate"] if what == "restored" else -99
                    )
                    if top - 2 < line < top + 9:  # the label would sit on the reference line
                        ax.text(x - width / 2 - 0.035, 100 * d["rate"], f"{100 * d['rate']:.0f}",
                                ha="right", va="center", fontsize=5.8, color=INK)  # fmt: skip
                    else:
                        ax.text(x, top, f"{100 * d['rate']:.0f}", ha="center", va="bottom",
                                fontsize=5.8, color=INK)  # fmt: skip
            if what == "found":
                c = 100 * res[k]["chance"]
                ax.plot([i - 2 * step, i + 2 * step], [c, c], color=MUTED, lw=1.1, zorder=5)
            else:
                c = res[k]["restored"]["ceiling"]
                ax.plot([i - 2 * step, i + 2 * step], [100 * c["rate"]] * 2, color=MUTED,
                        lw=1.1, zorder=5)  # fmt: skip
        note = "random atom" if what == "found" else "true location given"
        ax.plot([], [], color=MUTED, lw=1.1, label=note)
        ax.legend(loc="upper right", fontsize=5.8, handlelength=1.4, borderaxespad=0.1,
                  bbox_to_anchor=(1.0, 1.13))  # fmt: skip
    for ax, letter in zip(axes, "ab"):
        panel_letter(fig, (ax.get_position().x0 - 0.035, ax.get_position().y1 + 0.035), letter)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="none") for c, _ in METHOD.values()]
    fig.legend(handles, [label for _, label in METHOD.values()], loc="upper center", ncol=4,
               fontsize=6.2, handlelength=1.0, handleheight=0.8, columnspacing=1.6,
               bbox_to_anchor=(0.54, 1.005))  # fmt: skip
    finalize_figure(fig, FIG / "pd_fig3_results")


def main() -> None:
    apply_publication_style()
    which = set(sys.argv[1:]) or {"1", "2", "3"}
    if "1" in which:
        fig_defects()
    if "2" in which:
        fig_pipeline()
    if "3" in which:
        fig_results()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
