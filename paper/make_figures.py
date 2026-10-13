"""Build the paper's figures from the pushed results (run from the repository root).

    python paper/make_figures.py

Figure 1 crops one panel each from the relaxed-sample plots of three Task B experiments
(sensitivity run, seed 0, D2): the retrained model at w = 6, hand insertion and local
melt-quench. Figures 2 and 3 are paper-sized versions of results/seeds/seeds.png and
results/sensitivity/sensitivity.png, drawn by scripts/plot_seeds.py and
scripts/plot_sensitivity.py.
"""

import glob
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path("paper/figures")
# plot_samples.py writes a 2 x 4 grid at 1560 x 855 px; panel boxes (x0, y0, x1, y1)
COLS = [(22, 386), (405, 770), (790, 1154), (1175, 1539)]
ROW0 = (34, 390)
LEGEND = (300, 800, 1262, 855)
PANELS = [
    ("sens_cond_smax1_w6_rout4_host_D2", 2, "(a) Conditional model, w = 6\n(success)"),
    ("sens_hand_insert_D2", 1, "(b) Hand insertion\n(success)"),
    ("sens_lmq_rout4_D2", 2, "(c) Local melt-quench\n(miss)"),
]


def plot_path(name: str) -> str:
    paths = sorted(glob.glob(f"results/*/experiments/{name}/relaxed.png"))
    if not paths:
        raise FileNotFoundError(name)
    return paths[-1]


def figure1() -> None:
    fig = plt.figure(figsize=(7.0, 3.0))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.12], hspace=0.05, wspace=0.06)
    legend = None
    for k, (name, col, title) in enumerate(PANELS):
        im = Image.open(plot_path(name)).convert("RGB")
        x0, x1 = COLS[col]
        ax = fig.add_subplot(grid[0, k])
        ax.imshow(im.crop((x0, ROW0[0], x1, ROW0[1])))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if k == 0:
            legend = im.crop(LEGEND)
    ax = fig.add_subplot(grid[1, :])
    ax.imshow(legend)
    ax.axis("off")
    fig.savefig(OUT / "fig1_taskB.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    common = ["--config", "configs/eval/default.yaml", "font=7", "ms=4", "dpi=300"]
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_seeds.py",
            *common,
            "figsize=[7.0,2.6]",
            "legend_font=5",
            f"out={OUT / 'fig2_seeds.png'}",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_sensitivity.py",
            *common,
            "layout=grid",
            "figsize=[7.0,5.2]",
            "legend_font=5.5",
            "lw=1.2",
            f"out={OUT / 'fig3_sensitivity.png'}",
        ],
        check=True,
    )
    print("figures in", OUT)


if __name__ == "__main__":
    main()
