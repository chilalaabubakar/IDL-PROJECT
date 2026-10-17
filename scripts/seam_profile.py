"""Realism around the site, shell by shell: is there a seam at the edge of the rebuilt region?

    python scripts/seam_profile.py --config configs/eval/default.yaml data=data/ka2d_256 \\
        runs="[runs/eval/*_seam_*]" out=results/<dir>/seam.json [shell=1.0] [r_max=7.0]

Task B rebuilds the disk of radius R_out around the site and keeps every atom outside it
fixed. If the rebuilt atoms do not join the fixed ones naturally, the energies of atoms near
r = R_out should look less natural than those further in or out. The default local W1 (all
atoms within r_loc = 3 of the site) cannot tell this apart from a better region for R_out = 5,
because the edge then lies outside the window.

For each run (samples.npz + relaxed.npz, as for evaluate.py) this computes the W1 between
the per-atom energies of generated and natural defects of the same class (test split):
  - in shells r in [k, k + shell) around the site, k = 0 .. r_max;
  - in windows r <= r_loc for several r_loc (``r_locs``), the default metric at other sizes.
Natural train defects against natural test defects give the floor of each. 95% bootstrap CIs
over samples. Writes ``out`` (JSON) and a figure next to it (.png).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from glassdiff.data.dataset import load_split
from glassdiff.eval.bootstrap import bootstrap_ci_items
from glassdiff.eval.metrics import QUANTILE_LEVELS, quantile_w1
from glassdiff.geometry import minimum_image
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import DefectClass, Structures
from glassdiff.utils.config import cli_config


def radial_energies(s: Structures, centres: torch.Tensor, chunk: int = 64):
    """Per sample: (distance of every atom to the centre, its energy), numpy arrays."""
    out = []
    for i in range(0, s.batch_size, chunk):
        sub = Structures(s.pos[i : i + chunk], s.types[i : i + chunk], s.box[i : i + chunk])
        pe = ka_energy(sub, per_atom=True)
        d = minimum_image(sub.pos - centres[i : i + chunk, None, :].to(sub.pos), sub.box)
        r = d.norm(dim=-1)
        out += [(r[b].numpy(), pe[b].detach().numpy()) for b in range(sub.batch_size)]
    return out


def natural(data_dir: Path, split: str, defect: int, max_sites: int = 512):
    data = load_split(data_dir / f"{split}.npz")
    sites = torch.nonzero(data.labels == defect)[:max_sites]
    s = data.structures
    g, a = sites[:, 0], sites[:, 1]
    sub = Structures(s.pos[g].double(), s.types[g], s.box[g].double())
    return radial_energies(sub, sub.pos[torch.arange(len(g)), a])


def generated(run: Path):
    with np.load(run / "samples.npz") as f:
        samp = {k: f[k] for k in f.files}
    with np.load(run / "relaxed.npz") as f:
        pos = torch.from_numpy(f["pos"]).double()
    types, box = torch.from_numpy(samp["types"]).long(), torch.from_numpy(samp["box"]).double()
    s = Structures(pos, types, box)
    return int(samp["defect"]), radial_energies(s, torch.from_numpy(samp["target"]).double())


def w1_in(gen, ref, lo: float, hi: float, n_boot: int) -> list[float]:
    pool = np.concatenate([e[(r >= lo) & (r < hi)] for r, e in ref])
    if len(pool) == 0:
        return [float("nan")] * 3
    ref_q = np.quantile(pool, QUANTILE_LEVELS)
    items = [e[(r >= lo) & (r < hi)] for r, e in gen]
    return list(
        bootstrap_ci_items(
            items, lambda sel: quantile_w1(np.concatenate(sel), ref_q), n_boot, seed=0
        )
    )


def profile(gen, ref, shells, r_locs, n_boot) -> dict:
    return {
        "shells": [[lo, hi, *w1_in(gen, ref, lo, hi, n_boot)] for lo, hi in shells],
        "windows": [[r, *w1_in(gen, ref, 0.0, r + 1e-9, n_boot)] for r in r_locs],
    }


def r_out_of(run: Path) -> float | None:
    cfg_path = run / "config.yaml"
    if not cfg_path.exists():
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    if "r_melt" in cfg:
        return float(cfg["r_melt"])
    req = cfg.get("request", {}) or {}
    return float(req.get("host_r_out", 4.0)) if req.get("mode") == "host" else None


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    defects = sorted({r["defect"] for r in result["runs"].values()})
    fig, axes = plt.subplots(1, len(defects), figsize=(5.5 * len(defects), 3.8), squeeze=False)
    for ax, defect in zip(axes[0], defects):
        floor = np.array(result["floor"][defect]["shells"])
        mid = (floor[:, 0] + floor[:, 1]) / 2
        ax.fill_between(mid, floor[:, 3], floor[:, 4], color="0.85", label="natural floor")
        for i, (name, r) in enumerate(result["runs"].items()):
            if r["defect"] != defect:
                continue
            sh = np.array(r["shells"])
            line = ax.errorbar(
                (sh[:, 0] + sh[:, 1]) / 2,
                sh[:, 2],
                yerr=np.nan_to_num(np.clip([sh[:, 2] - sh[:, 3], sh[:, 4] - sh[:, 2]], 0, None)),
                marker="o",
                ms=3,
                capsize=2,
                label=name,
            )
            if r.get("r_out"):
                ax.axvline(r["r_out"], color=line[0].get_color(), ls=":", lw=1)
        ax.set_xlabel("distance from the site (sigma)")
        ax.set_ylabel("W1 of per-atom energy in the shell")
        ax.set_title(f"{defect}: dotted lines mark R_out")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    cfg = cli_config(__doc__)
    data_dir = Path(cfg["data"])
    shell, r_max = float(cfg.get("shell", 1.0)), float(cfg.get("r_max", 7.0))
    shells = [(k, k + shell) for k in np.arange(0.0, r_max, shell).tolist()]
    r_locs = [float(r) for r in cfg.get("r_locs", [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])]
    n_boot = int(cfg.get("n_boot", 1000))
    runs = [Path(p) for p in cfg["runs"]]
    result: dict = {"shell": shell, "r_locs": r_locs, "floor": {}, "runs": {}}
    refs: dict[int, list] = {}
    for run in runs:
        defect, gen = generated(run)
        if defect not in refs:
            refs[defect] = natural(data_dir, "test", defect)
            floor = profile(
                natural(data_dir, "train", defect), refs[defect], shells, r_locs, n_boot
            )
            result["floor"][DefectClass(defect).name] = floor
        name = run.name.split("_", 1)[1] if run.name[:8].isdigit() else run.name
        row = profile(gen, refs[defect], shells, r_locs, n_boot)
        result["runs"][name] = {"defect": DefectClass(defect).name, "r_out": r_out_of(run), **row}
        w = ", ".join(f"{s[0]:g}-{s[1]:g}: {s[2]:.3f}" for s in row["shells"])
        print(f"{name} (R_out {r_out_of(run)}): {w}", flush=True)
    out = Path(cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    plot(result, out.with_suffix(".png"))
    print("->", out, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
