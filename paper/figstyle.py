"""House style for the paper's figures, following the figures4papers conventions
(https://github.com/ChenLiu-1996/figures4papers, scientific-figure-making/references/api.md):
Helvetica-type sans fonts, top/right spines off, frameless legends, the blue-green-red-
neutral palette (blues = our method, reds = baselines, neutrals = references), lowercase
bold panel letters, 300 dpi PNG plus vector PDF.

Figures are drawn at their printed size (the paper's text block is 5.5 in wide), so font
sizes here are the sizes on the page.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_1": "#DCE8F5",  # pastel fill for boxes of our method
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "green_dark": "#2E6B30",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_1": "#F0F0F0",
    "grey": "#767676",
    "grey_dark": "#4D4D4D",
    "ink": "#272727",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}
# methods in result plots: blues for the model, reds for the baselines, grey for floors
METHOD = {
    "model, w = 2": dict(color=PALETTE["blue_secondary"], mfc="white"),
    "model, w = 6": dict(color=PALETTE["blue_main"], mfc=PALETTE["blue_main"]),
    "hand insertion": dict(color=PALETTE["red_strong"], mfc=PALETTE["red_strong"]),
    "local melt-quench": dict(color=PALETTE["red_2"], mfc=PALETTE["red_2"]),
}
# atoms: species differ in colour, lightness and size; pinned atoms are drawn pale
SPECIES = (PALETTE["blue_secondary"], PALETTE["red_2"])
SPECIES_PALE = ("#C9DAEE", "#F7DEDC")
DIAMETER = (1.0, 0.88)  # sigma_AA, sigma_BB


def _sans() -> list[str]:
    """Helvetica-type fonts that are installed (Liberation Sans has Arial's metrics)."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ("Helvetica", "Arial", "Liberation Sans") if f in have] + ["DejaVu Sans"]


def apply_publication_style(font_size: float = 7.0, axes_linewidth: float = 0.8) -> None:
    sans = _sans()
    plt.rcParams.update(
        {
            "font.family": sans,
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
            "legend.fontsize": font_size - 0.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": axes_linewidth,
            "xtick.major.width": axes_linewidth,
            "ytick.major.width": axes_linewidth,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.frameon": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": sans[0],
            "mathtext.it": f"{sans[0]}:italic",
            "mathtext.bf": f"{sans[0]}:bold",
            "mathtext.cal": sans[0],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def panel_letter(fig, ax_or_xy, letter: str, size: float = 9.0) -> None:
    """Lowercase bold panel letter at the top-left of an axes (or at figure coords)."""
    if isinstance(ax_or_xy, tuple):
        x, y = ax_or_xy
    else:
        bb = ax_or_xy.get_position()
        x, y = bb.x0 - 0.01, bb.y1 + 0.01
    fig.text(x, y, letter, fontsize=size, fontweight="bold", ha="right", va="bottom")


def finalize_figure(fig, out_path: str | Path, formats=("png", "pdf"), dpi: int = 300) -> list:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
        paths.append(p)
    plt.close(fig)
    return paths


def draw_atoms(
    ax,
    pos: np.ndarray,  # [N, 2]
    types: np.ndarray,  # [N]
    box: np.ndarray,  # [2]
    centre: np.ndarray | None = None,
    half: float | None = None,
    pale: np.ndarray | None = None,  # [N] bool, drawn in pale colours (pinned host)
    rings: np.ndarray | None = None,  # [N] bool, defect atoms
    edge_lw: float = 0.3,
    ring_lw: float = 1.0,
) -> None:
    """Atoms at physical size (diameter sigma) with periodic images, in a window of
    half-width ``half`` around ``centre`` (default: the whole box)."""
    pos = np.mod(pos, box)
    if centre is None:
        centre, half = box / 2, float(box.max()) / 2
    shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], float) * box
    pale = np.zeros(len(pos), bool) if pale is None else pale
    for t in (0, 1):
        for is_pale in (True, False):
            sel = (types == t) & (pale == is_pale)
            if not sel.any():
                continue
            xy = (pos[sel][None] + shifts[:, None]).reshape(-1, 2)
            keep = (np.abs(xy - centre) <= half + 0.6).all(1)
            xy = xy[keep]
            d = np.full(len(xy), DIAMETER[t])
            ax.add_collection(
                EllipseCollection(
                    d,
                    d,
                    np.zeros(len(xy)),
                    units="xy",
                    offsets=xy,
                    offset_transform=ax.transData,
                    facecolors=(SPECIES_PALE if is_pale else SPECIES)[t],
                    edgecolors=PALETTE["grey"] if is_pale else PALETTE["ink"],
                    linewidths=edge_lw,
                    zorder=2,
                )
            )
    if rings is not None and rings.any():
        xy = (pos[rings][None] + shifts[:, None]).reshape(-1, 2)
        d = np.tile([DIAMETER[t] + 0.32 for t in types[rings]], len(shifts))
        keep = (np.abs(xy - centre) <= half + 0.6).all(1)
        ax.add_collection(
            EllipseCollection(
                d[keep],
                d[keep],
                np.zeros(int(keep.sum())),
                units="xy",
                offsets=xy[keep],
                offset_transform=ax.transData,
                facecolors="none",
                edgecolors=PALETTE["ink"],
                linewidths=ring_lw,
                zorder=3,
            )
        )
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(PALETTE["grey"])
        s.set_linewidth(0.6)
