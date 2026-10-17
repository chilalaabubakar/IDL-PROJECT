"""Conditional NequIP denoiser for silica (docs/SILICA_3D_PLAN.md section 3).

DM2's E(3)-equivariant NequIP (graphite package, MIT licence) with the conditioning of our
2D model added: noise level sigma, held-fixed atoms, a per-atom defect label, and the
cooling rate (as in DM2's conditional model).

    d = model(pos, types, box, batch, sigma, pin, label, cooling_rate)    # [M, 3]

* Input features are scalars (irreps "0e"): species embedding for the node features x; a
  conditioning vector for the auxiliary features z that every interaction layer and the
  output layer read (pin flag + label token + Fourier features of log sigma + Gaussian basis
  of log10 cooling rate, as DM2 embeds cooling rates).
* Edges: all pairs within ``cutoff`` (5 A, as DM2), rebuilt from the current positions;
  Bessel radial basis and spherical harmonics of the minimum-image edge vectors.
* Defaults follow DM2's conditional silica model (hidden 64x0e + 32x1e, edges
  4x0e + 4x1e + 2x2e, 3 interactions, 12 neighbours for normalisation): about 1.0 M
  parameters, about 6.7 s per training step for a 3,000-atom glass on 4 CPU cores.
* Output: one vector per atom (irreps "1x1e"), the predicted displacement added by the noise,
  set to zero on held atoms. Equivariant to rotations, translations and permutations;
  tests/test_silica.py checks this numerically.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

torch.serialization.add_safe_globals([slice])  # e3nn 0.4.4 loads a pickled constants file
from e3nn import o3  # noqa: E402
from e3nn.nn import Gate  # noqa: E402
from graphite.nn.basis import bessel  # noqa: E402
from graphite.nn.conv.e3nn_nequip import Interaction  # noqa: E402
from graphite.nn.models.e3nn_nequip import Compose, tp_path_exists  # noqa: E402

from glassdiff.silica.graph import batched_radius_graph  # noqa: E402
from glassdiff.silica.requests import NUM_LABEL_TOKENS, UNLABELLED  # noqa: E402


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FourierSigma(nn.Module):
    """Gaussian random Fourier features of log(sigma), then an MLP (as our 2D model)."""

    def __init__(self, dim: int, n_features: int = 32) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.randn(n_features // 2))
        self.net = mlp([n_features, dim, dim])

    def forward(self, sigma: Tensor) -> Tensor:
        x = 2 * math.pi * torch.log(sigma)[:, None] * self.freqs
        return self.net(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class GaussianRate(nn.Module):
    """Gaussian basis on log10(cooling rate / K/ps) in [-2, 3], then an MLP (as DM2)."""

    def __init__(self, dim: int, n_basis: int = 12, lo: float = -2.0, hi: float = 3.0) -> None:
        super().__init__()
        self.register_buffer("centres", torch.linspace(lo, hi, n_basis))
        self.width = (hi - lo) / (n_basis - 1)
        self.net = mlp([n_basis, dim, dim])

    def forward(self, rate: Tensor) -> Tensor:
        x = torch.log10(rate)[:, None]
        return self.net(torch.exp(-0.5 * ((x - self.centres) / self.width) ** 2))


class ConditionalNequIP(nn.Module):
    def __init__(
        self,
        n_species: int = 2,
        node_dim: int = 8,
        cond_dim: int = 16,
        hidden_irreps: str = "64x0e + 32x1e",
        edge_irreps: str = "4x0e + 4x1e + 2x2e",
        n_convs: int = 3,
        cutoff: float = 5.0,
        n_basis: int = 16,
        radial_neurons: tuple[int, ...] = (16, 64),
        num_neighbors: float = 12.0,
        sigma_aware: bool = True,
        rate_aware: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff, self.n_basis = float(cutoff), int(n_basis)
        self.sigma_aware, self.rate_aware = sigma_aware, rate_aware
        self.species = nn.Embedding(n_species, node_dim)
        self.pin = nn.Embedding(2, cond_dim)
        self.label = nn.Embedding(NUM_LABEL_TOKENS, cond_dim)
        self.sigma = FourierSigma(cond_dim) if sigma_aware else None
        self.rate = GaussianRate(cond_dim) if rate_aware else None
        self.node_in = mlp([node_dim + cond_dim, node_dim, node_dim])
        self.cond_in = mlp([node_dim + cond_dim, cond_dim, cond_dim])

        irreps_x = o3.Irreps(f"{node_dim}x0e")
        self.irreps_z = o3.Irreps(f"{cond_dim}x0e")
        self.irreps_edge = o3.Irreps(edge_irreps)
        hidden = o3.Irreps(hidden_irreps)
        act_scalars = {1: nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}
        irreps = irreps_x
        self.interactions = nn.ModuleList()
        # layer construction as in graphite.nn.models.e3nn_nequip.NequIP (DM2)
        for _ in range(n_convs):
            scalars = o3.Irreps(
                [
                    (m, ir)
                    for m, ir in hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge, ir)
                ]
            )
            gated = o3.Irreps(
                [
                    (m, ir)
                    for m, ir in hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge, ir)
                ]
            )
            gates = o3.Irreps([(m, "0e") for m, _ in gated]).simplify()
            gate = Gate(
                scalars,
                [act_scalars[ir.p] for _, ir in scalars],
                gates,
                [act_gates[ir.p] for _, ir in gates],
                gated,
            )
            conv = Interaction(
                irreps_in=irreps,
                irreps_node=self.irreps_z,
                irreps_edge=self.irreps_edge,
                irreps_out=gate.irreps_in,
                radial_neurons=list(radial_neurons),
                num_neighbors=num_neighbors,
            )
            irreps = gate.irreps_out
            self.interactions.append(Compose(conv, gate))
        self.out = o3.FullyConnectedTensorProduct(irreps, self.irreps_z, "1x1e")

    def forward(
        self,
        pos: Tensor,  # [M, 3] atoms of all structures, structure after structure
        types: Tensor,  # [M]
        box: Tensor,  # [G, 3]
        batch: Tensor,  # [M] structure index of each atom
        sigma: Tensor,  # [G]
        pin: Tensor | None = None,  # [M] bool
        label: Tensor | None = None,  # [M] long tokens
        cooling_rate: Tensor | None = None,  # [G] K/ps
    ) -> Tensor:
        pin_idx = torch.zeros_like(types) if pin is None else pin.long()
        lab = torch.full_like(types, UNLABELLED) if label is None else label
        cond = self.pin(pin_idx) + self.label(lab)
        if self.sigma is not None:
            cond = cond + self.sigma(sigma)[batch]
        if self.rate is not None and cooling_rate is not None:
            cond = cond + self.rate(cooling_rate)[batch]
        spec = self.species(types)
        both = torch.cat([spec, cond], dim=-1)
        x, z = self.node_in(both), self.cond_in(both)

        edge_index, vec = batched_radius_graph(pos, box, batch, self.cutoff)
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge, vec, normalize=True, normalization="component"
        )
        edge_len = bessel(vec.norm(dim=-1), start=0.0, end=self.cutoff, num_basis=self.n_basis)
        for layer in self.interactions:
            x = layer(x, z, edge_index, edge_sh, edge_len)
        out = self.out(x, z)
        if pin is not None:
            out = out * (~pin)[:, None]
        return out


def build_model(cfg: dict) -> ConditionalNequIP:
    cfg = dict(cfg)
    cfg.pop("name", None)
    if "radial_neurons" in cfg:
        cfg["radial_neurons"] = tuple(cfg["radial_neurons"])
    return ConditionalNequIP(**cfg)


def load_model(path, device="cpu") -> ConditionalNequIP:
    """EMA weights of a checkpoint written by scripts/silica/train.py."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_cfg"])
    model.load_state_dict(ckpt["ema"])
    return model.to(device).eval()
