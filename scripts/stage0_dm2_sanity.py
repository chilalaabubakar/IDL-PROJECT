"""Stage 0 [B-1]: DM2's pretrained unconditional a-SiO2 model against its training glasses.

DM2 (Yang & Schwalbe-Koda; https://github.com/digital-synthesis-lab/DM2, MIT licence) turns a
random box of atoms into amorphous SiO2 by repeated denoising. This script reruns its
unconditional demo and checks that the samples have the structure of the glasses the model was
trained on (1 K/ps melt-quench, samples 0-2 of demo/demo_training/simu_data): partial g(r),
Si and O coordination, Si-O bond length, O-Si-O and Si-O-Si angles. The g(r) distance to the
training glasses is compared with the distance between training and held-out glasses at the
same number of atoms (the floor).

Separate environment (DM2 pins python 3.10, torch 2.5; torch 2.14 on CPU works with the two
fixes in `generate`). A CPU is enough: about 25 min per 300-atom sample on one thread.

    git clone https://github.com/digital-synthesis-lab/DM2 && pip install -e DM2
    pip install e3nn==0.4.4 torch_geometric ase scikit-learn matplotlib

    python scripts/stage0_dm2_sanity.py generate --dm2 DM2 --seed 0 \\
        --out runs/stage0/gen_seed0.extxyz
    python scripts/stage0_dm2_sanity.py analyze --dm2 DM2 --out results/stage0 \\
        runs/stage0/gen_seed*.extxyz

`generate` follows demo/demo_generating/denoise_generate_unconditional.py: the 300-atom random
box, 2,900 noisy steps with sigma linear from 1.0 to 0.001 (noise from graphite's
RattleParticles), then 100 noise-free steps, graph cutoff 5 A.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL = "demo/model/gen-a-sio2-uncond-v1.pt"
INIT = "demo/demo_generating/inital_data/random_sio2_size_300_demo.dat"
SIMU = "demo/demo_training/simu_data"
CUTOFF = 5.0

R_MAX, DR = 8.0, 0.05
EDGES = np.arange(0.0, R_MAX + DR / 2, DR)
CENTRES = 0.5 * (EDGES[1:] + EDGES[:-1])
SHELL = 4.0 / 3.0 * np.pi * (EDGES[1:] ** 3 - EDGES[:-1] ** 3)
PAIRS = {"Si-O": ("Si", "O"), "O-O": ("O", "O"), "Si-Si": ("Si", "Si")}
ANGLES = ("O-Si-O", "Si-O-Si")
RC_SIO = 2.2  # first minimum of g_SiO(r)
RATES = {"0_1k": "0.1 K/ps", "1k": "1 K/ps", "10k": "10 K/ps", "100k": "100 K/ps"}


# --------------------------------------------------------------------------- generate


class InitialEmbedding(nn.Module):
    """As in DM2's scripts. The pickled model refers to ``__main__.InitialEmbedding``, so this
    class must live in the script that is run."""

    def __init__(self, num_species, cutoff):
        from graphite.nn.basis import bessel

        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def _patch_torch_for_dm2() -> None:
    # e3nn 0.4.4 loads its constants.pt (tensors and slices) with a bare torch.load, which
    # torch >= 2.6 refuses by default (weights_only=True).
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    # The pickled model holds e3nn TorchScript modules saved on a GPU; e3nn reloads them with
    # torch.jit.load(buffer) and no map_location, which fails on a CPU-only machine.
    jit_load = torch.jit.load

    def jit_load_here(f, map_location=None, *args, **kwargs):
        if map_location is None and not torch.cuda.is_available():
            map_location = "cpu"
        return jit_load(f, map_location, *args, **kwargs)

    torch.jit.load = jit_load_here


@torch.no_grad()
def denoise(atoms, model, steps: int, sigma_max: float | None = None) -> np.ndarray:
    """DM2's denoising loop; with ``sigma_max`` the noisy phase, without it the final one."""
    from ase.neighborlist import primitive_neighbor_list
    from graphite.transforms import RattleParticles
    from sklearn.preprocessing import LabelEncoder
    from torch_geometric.data import Data

    dev = next(model.parameters()).device
    data = Data(
        x=torch.tensor(LabelEncoder().fit_transform(atoms.numbers), device=dev).long(),
        pos=torch.tensor(atoms.positions, device=dev).float(),
        cell=torch.tensor(np.array(atoms.cell), device=dev).float(),
        pbc=atoms.pbc,
        numbers=atoms.numbers,
    )
    sigmas = None if sigma_max is None else torch.linspace(sigma_max, 0.001, steps)
    for k in range(steps):
        i, j, d = primitive_neighbor_list(
            "ijD",
            cutoff=CUTOFF,
            pbc=data.pbc,
            cell=data.cell.cpu().numpy(),
            positions=data.pos.cpu().numpy(),
            numbers=data.numbers,
        )
        data.edge_index = torch.from_numpy(np.stack((i, j))).long().to(dev)
        data.edge_attr = torch.from_numpy(d).float().to(dev)
        disp = model(data)
        if sigmas is not None:
            disp = disp + RattleParticles(sigma_max=float(sigmas[k]))(data.clone()).dx.to(dev)
        data.pos.sub_(disp)
        if (k + 1) % 500 == 0 or k + 1 == steps:
            print(f"  step {k + 1}/{steps}", flush=True)
    return data.pos.cpu().numpy()


def generate(a: argparse.Namespace) -> None:
    import ase
    import ase.io

    _patch_torch_for_dm2()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(a.model or f"{a.dm2}/{MODEL}", map_location=dev, weights_only=False)
    model = model.to(dev).eval()
    atoms = ase.io.read(a.init or f"{a.dm2}/{INIT}", format="lammps-data")

    def moved(pos):
        return ase.Atoms(atoms.get_chemical_symbols(), positions=pos, cell=atoms.cell, pbc=True)

    t0 = time.time()
    atoms = moved(denoise(atoms, model, a.n_noisy, sigma_max=1.0))
    atoms = moved(denoise(atoms, model, a.n_final))
    atoms.wrap()
    atoms.info.update(seed=a.seed, n_noisy=a.n_noisy, n_final=a.n_final, device=str(dev))
    atoms.info["minutes"] = round((time.time() - t0) / 60, 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(a.out, atoms)
    print(f"done in {atoms.info['minutes']} min on {dev} -> {a.out}", flush=True)


# --------------------------------------------------------------------------- analyze


def read_simulated(path):
    import ase.io

    atoms = ase.io.read(path, format="lammps-data", atom_style="charge")
    atoms.set_chemical_symbols(np.where(atoms.get_initial_charges() > 0, "Si", "O"))
    return atoms


_PAIR_CACHE: dict[int, tuple] = {}


def gr_counts(atoms, centres=None):
    """Pair histograms and their ideal-gas normalisation; ``centres`` restricts the first atom
    of each pair (to match the sample size of a smaller set)."""
    from ase.neighborlist import neighbor_list

    if id(atoms) not in _PAIR_CACHE:  # the structures stay alive for the whole run
        _PAIR_CACHE[id(atoms)] = neighbor_list("ijd", atoms, R_MAX)
    i, j, d = _PAIR_CACHE[id(atoms)]
    sym = np.array(atoms.get_chemical_symbols())
    keep = np.ones(len(atoms), bool)
    if centres is not None:
        keep[:] = False
        keep[centres] = True
    out = {}
    for name, (a, b) in PAIRS.items():
        hist = np.histogram(d[(sym[i] == a) & (sym[j] == b) & keep[i]], EDGES)[0]
        n_centre = int(((sym == a) & keep).sum())
        density_b = ((sym == b).sum() - (a == b)) / atoms.get_volume()
        out[name] = (hist.astype(float), n_centre * density_b * SHELL)
    return out


def mean_gr(counts):
    return {k: sum(c[k][0] for c in counts) / sum(c[k][1] for c in counts) for k in PAIRS}


def local_structure(atoms):
    """Si and O coordination numbers, Si-O bond lengths, O-Si-O and Si-O-Si angles."""
    from ase.neighborlist import neighbor_list

    i, j, d, vec = neighbor_list("ijdD", atoms, RC_SIO)
    sym = np.array(atoms.get_chemical_symbols())
    sio = (sym[i] == "Si") & (sym[j] == "O")
    osi = (sym[i] == "O") & (sym[j] == "Si")
    n_si = np.bincount(i[sio], minlength=len(atoms))[sym == "Si"]
    n_o = np.bincount(i[osi], minlength=len(atoms))[sym == "O"]
    angles = {}
    for name, mask in zip(ANGLES, (sio, osi)):
        order = np.argsort(i[mask], kind="stable")
        ii, unit = i[mask][order], (vec[mask] / d[mask, None])[order]
        starts = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
        cos = [
            unit[p] @ unit[q]
            for s, e in zip(starts, np.r_[starts[1:], len(ii)])
            for p in range(s, e)
            for q in range(p + 1, e)
        ]
        angles[name] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return n_si, n_o, d[sio], angles


def summarize(structures):
    gr = mean_gr([gr_counts(x) for x in structures])
    parts = [local_structure(x) for x in structures]
    n_si, n_o, bonds = (np.concatenate([p[k] for p in parts]) for k in range(3))
    ang = {k: np.concatenate([p[3][k] for p in parts]) for k in ANGLES}
    density = np.mean([x.get_masses().sum() / x.get_volume() * 1.66054 for x in structures])
    stats = {
        "n_structures": len(structures),
        "n_atoms": int(np.mean([len(x) for x in structures])),
        "density_g_cm3": round(float(density), 4),
        "Si_4fold_fraction": round(float((n_si == 4).mean()), 4),
        "Si_mean_CN": round(float(n_si.mean()), 4),
        "O_2fold_fraction": round(float((n_o == 2).mean()), 4),
        "Si-O_bond_mean_A": round(float(bonds.mean()), 4),
    }
    for k in ANGLES:
        stats[f"{k}_mean_deg"] = round(float(ang[k].mean()), 2)
        stats[f"{k}_std_deg"] = round(float(ang[k].std()), 2)
    return gr, ang, stats


def l1(g1, g2, r_max):
    m = CENTRES <= r_max
    return round(float(np.abs(g1[m] - g2[m]).sum() * DR), 4)


def same_size_floor(heldout, g_train, n_si, n_o, r_max, draws, rng):
    """L1 to the training g(r) of held-out glasses restricted to n_si Si and n_o O centres."""
    out = {k: [] for k in PAIRS}
    for _ in range(draws):
        counts = []
        split = zip(
            heldout,
            np.array_split(np.arange(n_si), len(heldout)),
            np.array_split(np.arange(n_o), len(heldout)),
        )
        for x, k_si, k_o in split:
            sym = np.array(x.get_chemical_symbols())
            centres = np.r_[
                rng.choice(np.flatnonzero(sym == "Si"), len(k_si), replace=False),
                rng.choice(np.flatnonzero(sym == "O"), len(k_o), replace=False),
            ]
            counts.append(gr_counts(x, centres))
        g = mean_gr(counts)
        for k in PAIRS:
            out[k].append(l1(g[k], g_train[k], r_max))
    return {
        k: {"mean": round(float(np.mean(v)), 4), "p95": round(float(np.percentile(v, 95)), 4)}
        for k, v in out.items()
    }


def plot(path, g_train, g_100k, g_gen, a_train, a_gen, n_gen, n_atoms, r_cmp):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 5, figsize=(20, 3.6))
    for k, name in enumerate(PAIRS):
        ax[k].plot(CENTRES, g_train[name], color="0.25", lw=2.2, label="MD, 1 K/ps (training)")
        ax[k].plot(CENTRES, g_100k[name], color="0.6", lw=1, ls="--", label="MD, 100 K/ps")
        ax[k].plot(
            CENTRES, g_gen[name], color="#2a6fdb", lw=1.4, label=f"DM2 ({n_gen} x {n_atoms})"
        )
        ax[k].axvline(r_cmp, color="0.85", lw=0.8)
        ax[k].set_xlim(0, R_MAX)
        ax[k].set_xlabel("r (Å)")
        ax[k].set_title(f"g(r), {name}")
    ax[0].legend(fontsize=7, frameon=False)
    bins = np.linspace(60, 180, 61)
    for k, name in enumerate(ANGLES):
        kw = dict(bins=bins, density=True, histtype="step")
        ax[3 + k].hist(a_train[name], color="0.25", lw=2.2, label="MD, 1 K/ps", **kw)
        ax[3 + k].hist(a_gen[name], color="#2a6fdb", lw=1.4, label="DM2", **kw)
        ax[3 + k].set_xlabel("angle (deg)")
        ax[3 + k].set_title(f"{name} angle")
    ax[3].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def analyze(a: argparse.Namespace) -> None:
    import ase.io

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    gen = [ase.io.read(f) for f in a.gen]
    init = ase.io.read(a.init or f"{a.dm2}/{INIT}", format="lammps-data")
    r_cmp = round(min(min(np.diag(np.array(x.cell))) for x in gen) / 2, 3)  # valid below L/2
    r_cmp = min(r_cmp, R_MAX)
    simu = a.simu or f"{a.dm2}/{SIMU}"
    refs = {
        r: [read_simulated(f) for f in sorted(glob.glob(f"{simu}/sio2_3000_glass_{r}_sample*.dat"))]
        for r in RATES
    }
    train, heldout = refs["1k"][:3], refs["1k"][3:]

    g_gen, a_gen, s_gen = summarize(gen)
    g_train, a_train, s_train = summarize(train)
    g_init, _, s_init = summarize([init])
    dist = {
        "DM2 generated": {k: l1(g_gen[k], g_train[k], r_cmp) for k in PAIRS},
        "random start": {k: l1(g_init[k], g_train[k], r_cmp) for k in PAIRS},
    }
    simulated = {}
    curves = {}
    for r, structs in refs.items():
        curves[r], _, simulated[RATES[r]] = summarize(structs)
        if r != "1k":
            dist[f"MD {RATES[r]}"] = {k: l1(curves[r][k], g_train[k], r_cmp) for k in PAIRS}
    g_held = mean_gr([gr_counts(x) for x in heldout])
    dist["floor: held-out 1 K/ps, all atoms"] = {k: l1(g_held[k], g_train[k], r_cmp) for k in PAIRS}
    n_si = sum(int((np.array(x.get_chemical_symbols()) == "Si").sum()) for x in gen)
    n_o = sum(int((np.array(x.get_chemical_symbols()) == "O").sum()) for x in gen)
    dist["floor: held-out 1 K/ps, same size"] = same_size_floor(
        heldout, g_train, n_si, n_o, r_cmp, a.floor_draws, rng
    )
    res = {
        "training_set": "1 K/ps samples 0-2 (DM2's unconditional training data)",
        "r_max_compared_A": r_cmp,
        "generation_minutes": [x.info.get("minutes") for x in gen],
        "structure": {
            "DM2 generated": s_gen,
            "random start": s_init,
            "MD 1 K/ps training": s_train,
            **{f"MD {k}": v for k, v in simulated.items()},
        },
        "gr_L1_to_training": dist,
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2) + "\n")
    plot(
        out / "stage0_dm2_sio2.png",
        g_train,
        curves["100k"],
        g_gen,
        a_train,
        a_gen,
        len(gen),
        s_gen["n_atoms"],
        r_cmp,
    )
    print(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="one sample with DM2's unconditional demo settings")
    g.add_argument("--dm2", default="DM2", help="DM2 checkout")
    g.add_argument("--model", help=f"default: DM2/{MODEL}")
    g.add_argument("--init", help=f"default: DM2/{INIT}")
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-noisy", type=int, default=2900)
    g.add_argument("--n-final", type=int, default=100)
    g.add_argument("--threads", type=int, default=1)
    g.set_defaults(func=generate)
    s = sub.add_parser("analyze", help="compare samples with DM2's simulated glasses")
    s.add_argument("--dm2", default="DM2", help="DM2 checkout")
    s.add_argument("--simu", help=f"default: DM2/{SIMU}")
    s.add_argument("--init", help=f"default: DM2/{INIT}")
    s.add_argument("--out", default="results/stage0")
    s.add_argument("--floor-draws", type=int, default=50)
    s.add_argument("gen", nargs="+", help="generated structures (extxyz)")
    s.set_defaults(func=analyze)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
