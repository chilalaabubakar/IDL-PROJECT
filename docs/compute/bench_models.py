"""CPU timing of one training step: DM2's NequIP (3D silica) vs our EGNN (2D Kob-Andersen).

    git clone https://github.com/digital-synthesis-lab/dm2 && pip install -e dm2 e3nn==0.4.4
    DM2_DIR=dm2 python docs/compute/bench_models.py        # from the repository root

Measured 2026-10-14 on 4 CPU cores: NequIP (DM2 settings, 0.60 M parameters) 8.3 s per
training step and 1.8 s per forward pass on a 3,000-atom glass (33 edges per atom within
5 A); our EGNN 0.25 s per training step on a 256-atom 2D glass. Per atom, a NequIP training
step costs 2.9x an EGNN one, and a NequIP forward pass 0.21x a NequIP training step.
These ratios are used in budget.py. (Our conditional model, src/glassdiff/silica/model.py,
1.0 M parameters, takes 6.7 s per step on the same 3,000-atom glass.)
"""

# e3nn loads a pickled constants file at import, so torch.load must be patched first.
# ruff: noqa: E402

import functools
import os
import sys
import time

import numpy as np
import torch

torch.serialization.add_safe_globals([slice])
torch.set_num_threads(4)
_load = torch.load
torch.load = functools.partial(_load, weights_only=False)
import ase.io
from ase.neighborlist import primitive_neighbor_list
from graphite.nn.basis import bessel
from graphite.nn.models.e3nn_nequip import NequIP_CoolingRateEmbed
from graphite.transforms import DownselectEdges, RattleParticles
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch_geometric.data import Batch, Data


class InitialEmbedding(nn.Module):
    def __init__(self, num_species, cutoff):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, 8)
        self.embed_node_z = nn.Embedding(num_species, 8)
        self.embed_edge = functools.partial(bessel, start=0.0, end=cutoff, num_basis=16)

    def forward(self, data):
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        return data


def graph(atoms, cutoff=10.0):
    i, j, D = primitive_neighbor_list(
        "ijD",
        cutoff=cutoff,
        pbc=atoms.pbc,
        cell=atoms.cell,
        positions=atoms.positions,
        numbers=atoms.numbers,
    )
    return Data(
        x=torch.tensor(LabelEncoder().fit_transform(atoms.numbers)).long(),
        pos=torch.tensor(atoms.positions).float(),
        cell=atoms.cell,
        pbc=atoms.pbc,
        numbers=atoms.numbers,
        cooling_rate=torch.tensor([1.0]),
        edge_index=torch.tensor(np.stack((i, j)), dtype=torch.long),
        edge_attr=torch.tensor(D, dtype=torch.float),
    )


def timeit(fn, n=3):
    fn()  # warm-up
    t = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t) / n


DM2 = os.environ.get("DM2_DIR", "DM2") + "/demo/demo_training/simu_data/"
atoms = ase.io.read(DM2 + "sio2_3000_glass_1k_sample0.dat", format="lammps-data")
print("silica atoms", len(atoms))
model = NequIP_CoolingRateEmbed(
    init_embed=InitialEmbedding(2, 5.0),
    irreps_node_x="8x0e",
    irreps_node_z="8x0e",
    irreps_hidden="64x0e + 32x1e",
    irreps_edge="4x0e + 4x1e + 2x2e",
    irreps_out="1x1e",
    num_convs=3,
    radial_neurons=[16, 64],
    num_neighbors=12,
)
print("NequIP params", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
rattle, down = RattleParticles(sigma_max=0.75), DownselectEdges(cutoff=5.0)
base = graph(atoms)


def nequip_step():
    d = down(rattle(Batch.from_data_list([base.clone()])))
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(model(d, d.cooling_rate), d.dx)
    loss.backward()
    opt.step()


def nequip_fwd():
    with torch.no_grad():
        d = down(rattle(Batch.from_data_list([base.clone()])))
        model(d, d.cooling_rate)


t_nq = timeit(nequip_step)
t_nqf = timeit(nequip_fwd)
d = down(rattle(Batch.from_data_list([base.clone()])))
print(
    f"NequIP 3000 atoms: train step {t_nq:.2f}s, forward {t_nqf:.2f}s, edges/atom {d.edge_index.shape[1] / len(atoms):.1f}"
)

sys.path.insert(0, "src")
from glassdiff.data.dataset import load_split
from glassdiff.models.registry import build_model
from glassdiff.types import Structures

test = load_split("data/ka2d_256/test.npz").structures
s = Structures(test.pos[:1].float(), test.types[:1], test.box[:1].float())
eg = build_model({"name": "egnn_pbc", "hidden": 128, "n_layers": 4, "cutoff": 2.5, "k_max": 64})
opt2 = torch.optim.AdamW(eg.parameters(), lr=2e-4)
sig = torch.full((1,), 0.3)


def egnn_step():
    x = s.with_pos(s.pos + 0.3 * torch.randn_like(s.pos))
    opt2.zero_grad()
    loss = (eg(x, sig) - 0.3).pow(2).mean()
    loss.backward()
    opt2.step()


t_eg = timeit(egnn_step, 5)
print(f"EGNN 256 atoms (2D): train step {t_eg:.3f}s")
print(f"per-atom cost ratio NequIP/EGNN = {(t_nq / 3000) / (t_eg / 256):.2f}")
