"""Stage 0a [B-1]: run the course proof-of-concept notebook and reproduce the proposal's Fig. 1.

notebooks/stage0_poc.ipynb (provided with the project) trains small MLP denoisers (DDPM, 200
steps, time as an input) on tiny 2D grids. Its four code cells:

  1. grid      3x3 grid with Gaussian jitter (sd 0.15), unconditional generation (Fig. 1)
  2. grid_gpu  the same with a wider MLP and jitter 0.12 (written for a GPU; a CPU works)
  3. frenkel   half the training grids have one atom moved to a square centre; generation
               pins atom 4 at (1.5, 1.5) at every step (naive clamping)
  4. antisite  2x2 two-species grid, half with A0 and B0 swapped; generation pins the swap

The antisite cell has a bug: the A sites are built from np.arange (integers), so the swap
writes A0's new position (0.5, 0.5) as (0, 0). Its "antisite" training structures therefore
hold A0 and B0 on top of each other at A0's site, with B0's site empty, and the model never
sees the configuration that generation pins. ``antisite_fixed`` runs the same cell with
float A sites (one added ``.astype(float)``); the notebook itself is left unchanged.

This script runs each cell exactly as written (fresh namespace, fixed seed, figures saved
instead of shown) and measures:

  - grid, per seed: what Fig. 1 reports. 2,000 training and 2,000 generated structures; the
    distribution of all atom-pair distances; W1 between the two (grid spacings), against a
    floor (training set vs a fresh set from the same generator); and the positional disorder
    (RMS displacement from the ideal site, per coordinate).
  - frenkel / antisite(_fixed), seed 0: the share of 500 generated structures whose unpinned atoms all
    stay within 0.35 / 0.3 of their own sites (and, for Frenkel, the vacancy stays empty),
    next to the same share in the training structures that carry that defect.

    python scripts/stage0_poc.py --out results/stage0/poc --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

DEMOS = ("grid", "grid_gpu", "frenkel", "antisite")
VARIANTS = (*DEMOS, "antisite_fixed")
ANTISITE_A = "pos_A = np.stack([xv.flatten(), yv.flatten()], axis=1)"
GEN_COLOR, REF_COLOR = "#2a6fdb", "0.25"


def code_cells(path):
    nb = json.loads(Path(path).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cells = [c for c in cells if c.strip()]
    assert len(cells) == len(DEMOS), f"expected {len(DEMOS)} code cells, found {len(cells)}"
    out = dict(zip(DEMOS, cells))
    assert out["antisite"].count(ANTISITE_A) == 1
    out["antisite_fixed"] = out["antisite"].replace(ANTISITE_A, f"{ANTISITE_A}.astype(float)")
    return out


def run_cell(code, seed, figure=None):
    """Run one notebook cell as written; its plt.show() saves to ``figure`` (or discards)."""

    def show(*_, **__):
        if figure is not None:
            plt.savefig(figure, dpi=110, bbox_inches="tight")
        plt.close("all")

    np.random.seed(seed)
    torch.manual_seed(seed)
    ns = {"__name__": "stage0_poc_cell"}
    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log), mock.patch.object(plt, "show", show):
        exec(compile(code, "<stage0_poc.ipynb>", "exec"), ns)
    return ns, log.getvalue(), round(time.time() - t0, 1)


def grid_sites(grid_size):
    x = np.linspace(0, grid_size - 1, grid_size)
    xv, yv = np.meshgrid(x, x)
    return np.stack([xv.flatten(), yv.flatten()], axis=1)


def last_loss(log):
    lines = [s for s in log.splitlines() if "Loss" in s]
    return float(lines[-1].split("Loss:")[1]) if lines else None


def grid_metrics(ns, n):
    sites = grid_sites(ns["grid_size"]).ravel()
    train = ns["data"][:n].numpy()
    fresh = ns["get_lattice_data"](n_samples=n, grid_size=ns["grid_size"])[0].numpy()
    gen = np.stack(ns["generate_lattices"](n)).reshape(n, -1)

    def pairs(x):
        return np.concatenate([pdist(s.reshape(-1, 2)) for s in x])

    def disorder(x):
        return round(float(np.sqrt(((x - sites) ** 2).mean())), 4)

    d_train, d_fresh, d_gen = pairs(train), pairs(fresh), pairs(gen)
    out = {
        "W1_pair_distance_generated": round(float(wasserstein_distance(d_train, d_gen)), 4),
        "W1_pair_distance_floor": round(float(wasserstein_distance(d_train, d_fresh)), 4),
        "disorder_training": disorder(train),
        "disorder_fresh": disorder(fresh),
        "disorder_generated": disorder(gen),
        "min_pair_distance_generated": round(float(d_gen.min()), 3),
    }
    return out, (train, gen, d_train, d_gen)


def frenkel_metrics(ns, n, tol=0.35, target_idx=4, target_pos=(1.5, 1.5)):
    sites = grid_sites(ns["grid_size"])
    gen = ns["sample_frenkel_batch"](n_batch=n, target_idx=target_idx, target_pos=list(target_pos))
    train = ns["train_data"].view(-1, len(sites), 2).cpu().numpy()
    train = train[np.linalg.norm(train[:, target_idx] - sites[target_idx], axis=1) > tol]
    others = np.arange(len(sites)) != target_idx

    def intact(x):
        on_site = (np.linalg.norm(x[:, others] - sites[others], axis=2) < tol).all(1)
        vacancy_empty = (np.linalg.norm(x[:, others] - sites[target_idx], axis=2) > tol).all(1)
        return round(float((on_site & vacancy_empty).mean()), 3)

    return {
        "tolerance": tol,
        "intact_generated": intact(gen),
        "intact_training_with_this_defect": intact(train),
        "n_generated": n,
        "n_training_with_this_defect": len(train),
    }


def antisite_metrics(ns, n, tol=0.3):
    k = ns["n_per_type"]
    side = np.arange(int(round(k**0.5)))
    xv, yv = np.meshgrid(side, side)
    a_sites = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(float)
    sites = np.concatenate([a_sites, a_sites + 0.5])
    free = np.ones(2 * k, bool)
    free[[0, k]] = False  # A0 and B0 are pinned (swapped)
    gen = ns["sample_gen"](n_batch=n)
    train = ns["train_data"].cpu().numpy()[np.array(ns["train_labels"], bool)]
    swap_gap = float(np.linalg.norm(train[:, 0] - train[:, k], axis=1).mean())

    def intact(x):
        return round(
            float((np.linalg.norm(x[:, free] - sites[free], axis=2) < tol).all(1).mean()), 3
        )

    return {
        "tolerance": tol,
        "intact_generated": intact(gen),
        "intact_training_with_this_defect": intact(train),
        "n_generated": n,
        "n_training_with_this_defect": len(train),
        "A0_B0_distance_training_swapped": round(swap_gap, 3),  # 0.71 for a real swap
        "A0_B0_distance_generated": round(
            float(np.linalg.norm(gen[:, 0] - gen[:, k], axis=1).mean()), 3
        ),
    }


def plot_figure1(path, train, gen, d_train, d_gen, m, n_show=300):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, x, color, name in (
        (ax[0], train, REF_COLOR, "training"),
        (ax[1], gen, GEN_COLOR, "generated"),
    ):
        pts = x[:n_show].reshape(-1, 2)
        a.scatter(pts[:, 0], pts[:, 1], s=4, color=color, alpha=0.25, linewidths=0)
        a.set_title(f"{name}, {n_show} overlaid\ndisorder {m[f'disorder_{name}']:.3f}", fontsize=10)
        a.set_aspect("equal")
        a.set_xlim(-0.8, 2.8)
        a.set_ylim(-0.8, 2.8)
        a.set_xlabel("x (grid spacings)")
    ax[0].set_ylabel("y (grid spacings)")
    bins = np.linspace(0, 3.5, 71)
    kw = dict(bins=bins, density=True, histtype="step")
    ax[2].hist(d_train, color=REF_COLOR, lw=2.2, label=f"training ({len(train)})", **kw)
    ax[2].hist(d_gen, color=GEN_COLOR, lw=1.4, label=f"generated ({len(gen)})", **kw)
    ax[2].set_xlabel("atom-pair distance (grid spacings)")
    ax[2].set_ylabel("density")
    ax[2].set_title(
        f"pair distances\nW1 = {m['W1_pair_distance_generated']:.3f}"
        f" (floor {m['W1_pair_distance_floor']:.3f})",
        fontsize=10,
    )
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--notebook", default="notebooks/stage0_poc.ipynb")
    ap.add_argument("--out", default="results/stage0/poc")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=2000, help="structures per set for Fig. 1")
    ap.add_argument("--n-defect", type=int, default=500)
    ap.add_argument("--demos", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = code_cells(a.notebook)
    res = {"device": "cuda" if torch.cuda.is_available() else "cpu", "demos": {}}

    for demo in a.demos:
        seeds = a.seeds if demo == "grid" else a.seeds[:1]
        runs = []
        for seed in seeds:
            fig = out / f"{demo}.png" if seed == seeds[0] else None
            ns, log, secs = run_cell(cells[demo], seed, fig)
            run = {"seed": seed, "seconds": secs, "final_loss": last_loss(log)}
            if demo == "grid":
                m, arrays = grid_metrics(ns, a.n)
                run.update(m)
                if seed == seeds[0]:
                    plot_figure1(out / "figure1_reproduction.png", *arrays, m)
            elif demo == "frenkel":
                run.update(frenkel_metrics(ns, a.n_defect))
            elif demo.startswith("antisite"):
                run.update(antisite_metrics(ns, a.n_defect))
            runs.append(run)
            print(demo, json.dumps(run), flush=True)
        res["demos"][demo] = runs

    grid = res["demos"].get("grid", [])
    if len(grid) > 1:
        keys = [k for k in grid[0] if k.startswith(("W1", "disorder"))]
        res["grid_over_seeds"] = {
            k: {
                "mean": round(float(np.mean([r[k] for r in grid])), 4),
                "min": min(r[k] for r in grid),
                "max": max(r[k] for r in grid),
            }
            for k in keys
        }
    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res.get("grid_over_seeds", {}), indent=2))


if __name__ == "__main__":
    main()
