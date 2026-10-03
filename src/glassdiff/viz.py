"""Static figures of 2D configurations for reports and failure analysis.

Atoms are drawn at physical size (radius sigma/2: A 0.5, B 0.44) in the validated
categorical pair (A blue, B orange); identity never relies on colour alone: species also
differ in size, defects get a dark ring, pinned atoms a 45-degree hatch, and the
requested location a cross with the success-tolerance circle. A legend is always shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import EllipseCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
SPECIES_COLOURS = ("#2a78d6", "#eb6834")  # categorical slots 1-2, validated light mode
SPECIES_DIAMETER = (1.0, 0.88)


def _draw(ax, pos, types, box, labels=None, pin=None, target=None, r_tol=0.5, title=""):
    pos = np.mod(pos, box)
    # periodic images near the edges so atoms are not cut in half
    shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    for t in (0, 1):
        for hatch_on in (False, True):
            sel = (types == t) & (
                (pin if pin is not None else np.zeros_like(types, bool)) == hatch_on
            )
            if not sel.any():
                continue
            xy = np.concatenate([pos[sel] + np.array(s) * box for s in shifts])
            d = np.full(len(xy), SPECIES_DIAMETER[t])
            ax.add_collection(
                EllipseCollection(
                    d,
                    d,
                    np.zeros(len(xy)),
                    units="xy",
                    offsets=xy,
                    offset_transform=ax.transData,
                    facecolors=SPECIES_COLOURS[t],
                    edgecolors=SURFACE,
                    linewidths=0.6,
                    hatch="////" if hatch_on else None,
                )
            )
    if labels is not None and (labels > 0).any():
        xy = np.concatenate([pos[labels > 0] + np.array(sh) * box for sh in shifts])
        d = np.tile([SPECIES_DIAMETER[t] for t in types[labels > 0]], len(shifts))
        ax.add_collection(
            EllipseCollection(
                d,
                d,
                np.zeros(len(xy)),
                units="xy",
                offsets=xy,
                offset_transform=ax.transData,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.6,
            )
        )
    if target is not None:
        tgt = np.mod(target, box)
        ax.add_patch(Circle(tgt, r_tol, fill=False, ls="--", lw=1.0, ec=INK))
        ax.plot(*tgt, marker="+", ms=10, mew=2, color=INK)
    ax.set_xlim(0, box[0])
    ax.set_ylim(0, box[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(INK_SECONDARY)
        spine.set_linewidth(0.8)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, color=INK)


def plot_structures(
    pos: np.ndarray,  # [B, N, 2]
    types: np.ndarray,  # [B, N]
    box: np.ndarray,  # [B, 2]
    path: str | Path,
    labels: np.ndarray | None = None,  # [B, N] DefectClass
    pin: np.ndarray | None = None,  # [B, N] bool
    targets: np.ndarray | None = None,  # [B, 2]
    titles: list[str] | None = None,
    ncols: int = 4,
    panel_inches: float = 2.6,
) -> Path:
    n = len(pos)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        min(ncols, n),
        figsize=(panel_inches * min(ncols, n), panel_inches * nrows + 0.5),
        squeeze=False,
        facecolor=SURFACE,
    )
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        _draw(
            ax,
            pos[k],
            types[k],
            box[k],
            None if labels is None else labels[k],
            None if pin is None else pin[k],
            None if targets is None else targets[k],
            title="" if titles is None else titles[k],
        )
    handles = [
        Line2D(
            [], [], ls="", marker="o", ms=9, mfc=SPECIES_COLOURS[0], mec=SURFACE, label="A (large)"
        ),
        Line2D(
            [], [], ls="", marker="o", ms=8, mfc=SPECIES_COLOURS[1], mec=SURFACE, label="B (small)"
        ),
    ]
    if labels is not None:
        handles.append(
            Line2D(
                [], [], ls="", marker="o", ms=9, mfc="none", mec=INK, mew=1.6, label="defect atom"
            )
        )
    if pin is not None and pin.any():
        handles.append(Patch(fc=SPECIES_COLOURS[0], hatch="////", ec=SURFACE, label="pinned"))
    if targets is not None:
        handles.append(
            Line2D([], [], ls="", marker="+", ms=10, mew=2, color=INK, label="requested location")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # categorical slots 1-4 (light)


def plot_training(log_csv: str | Path, path: str | Path) -> Path:
    """Validation loss / sigma^2 per noise level against training step.

    Dividing by sigma^2 (the loss of predicting zero displacement) puts every noise level on
    one scale: the fraction of the injected noise the model has not removed. Lines are
    direct-labelled because two of the four colours are below 3:1 contrast.
    """
    import csv

    with open(log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    steps = np.array([float(r["step"]) for r in rows])
    sigmas = [k for k in rows[0] if k.startswith("val_")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    y_max = 1.1
    for colour, key in zip(SERIES_COLOURS, sigmas):
        sigma = float(key.split("_", 1)[1])
        y = np.array([float(r[key]) for r in rows]) / sigma**2
        y_max = max(y_max, 1.05 * float(y.max()))
        ax.plot(steps, y, color=colour, lw=2, marker="o", ms=4, label=f"σ = {sigma:g}")
        ax.annotate(
            f"σ = {sigma:g}",
            (steps[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(1.0, color=INK_SECONDARY, lw=1, ls="--")
    ax.annotate(
        "predict zero",
        (steps[0], 1.0),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlabel("training step", color=INK)
    ax.set_ylabel("val loss / σ²", color=INK)
    ax.set_ylim(0, y_max)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e4e3dd", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    ax.margins(x=0.12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
