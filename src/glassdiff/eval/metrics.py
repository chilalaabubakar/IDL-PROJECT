"""Evaluation metrics (docs/PROJECT_PLAN.md §4).  [Ticket E-4]

Every metric is computed on RELAXED samples (physics.fire.fire_minimize, no pins), one
value or one feature list per sample, so bootstrap CIs resample samples, never atoms.
References are natural defects from a held-out split (``natural_windows``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch import Tensor

from glassdiff.analysis.defects import DefectThresholds, detect_defects
from glassdiff.analysis.descriptors import psi6_global
from glassdiff.analysis.structure import bond_angles, pair_distances, radial_distribution
from glassdiff.data.dataset import GlassSplit
from glassdiff.data.requests import DEFECT_SPECIES
from glassdiff.eval.bootstrap import bootstrap_ci, bootstrap_ci_items
from glassdiff.geometry import minimum_image
from glassdiff.physics.ka_potential import ka_energy
from glassdiff.types import DefectClass, Request, Structures

SPECIES_PAIRS = {"AA": (0, 0), "AB": (0, 1), "BB": (1, 1)}
OVERLAP_DIST = 0.56  # 0.7 sigma_AB


def nearest_atom(s: Structures, target: Tensor, species: Tensor) -> tuple[Tensor, Tensor]:
    """Index [B] and distance [B] of the atom of the given species [B] nearest target [B, 2]."""
    d = minimum_image(s.pos - target[:, None, :].to(s.pos), s.box).norm(dim=-1)
    d = torch.where(s.types == species[:, None], d, torch.full_like(d, float("inf")))
    dist, idx = d.min(dim=-1)
    return idx, dist


def success(s: Structures, request: Request, th: DefectThresholds, r_tol: float = 0.5) -> Tensor:
    """[B] bool: the atom of the requested species nearest the target lies within r_tol of it
    and carries the requested defect label."""
    species = torch.tensor([int(DEFECT_SPECIES[DefectClass(int(k))]) for k in request.defect])
    idx, dist = nearest_atom(s, request.target, species.to(s.types.device))
    labels = detect_defects(s, th)
    hit = labels[torch.arange(s.batch_size), idx] == request.defect.to(labels.device)
    return (dist < r_tol) & hit


@dataclass
class LocalFeatures:
    """Per-sample feature lists inside the window of radius r_loc around a centre."""

    pe_atom: list[Tensor]
    pair_dist: dict[str, list[Tensor]]  # "AA", "AB", "BB"
    bond_angle: list[Tensor]
    relax_disp: list[Tensor]
    min_pair_dist: Tensor  # [B] smallest pair distance touching the window

    def __len__(self) -> int:
        return len(self.pe_atom)

    def extend(self, other: LocalFeatures) -> LocalFeatures:
        return LocalFeatures(
            self.pe_atom + other.pe_atom,
            {k: self.pair_dist[k] + other.pair_dist[k] for k in self.pair_dist},
            self.bond_angle + other.bond_angle,
            self.relax_disp + other.relax_disp,
            torch.cat([self.min_pair_dist, other.min_pair_dist]),
        )


def window_mask(s: Structures, centres: Tensor, r_loc: float) -> Tensor:
    return minimum_image(s.pos - centres[:, None, :].to(s.pos), s.box).norm(dim=-1) <= r_loc


def local_features(
    s: Structures, centres: Tensor, relax_disp: Tensor | None = None, r_loc: float = 3.0
) -> LocalFeatures:
    mask = window_mask(s, centres, r_loc)
    pe = ka_energy(s, per_atom=True)
    pair = {k: pair_distances(s, p, 2.5, mask) for k, p in SPECIES_PAIRS.items()}
    disp = torch.zeros_like(s.pos) if relax_disp is None else relax_disp.to(s.pos)
    min_pair = torch.stack(
        [
            torch.cat([pair[k][b] for k in pair] + [torch.tensor([9.9])]).min()
            for b in range(len(mask))
        ]
    )
    return LocalFeatures(
        pe_atom=[pe[b][mask[b]] for b in range(len(mask))],
        pair_dist=pair,
        bond_angle=bond_angles(s, centre_mask=mask),
        relax_disp=[disp[b][mask[b]].norm(dim=-1) for b in range(len(mask))],
        min_pair_dist=min_pair,
    )


def natural_windows(
    split: GlassSplit, defect: int, r_loc: float = 3.0, max_windows: int = 512, chunk: int = 64
) -> LocalFeatures:
    """Windows around naturally formed defects of one class (relaxation displacement 0)."""
    sites = torch.nonzero(split.labels == defect)[:max_windows]
    if len(sites) == 0:
        raise ValueError(f"no natural {DefectClass(defect).name} sites in this split")
    feats = None
    for i in range(0, len(sites), chunk):
        g, a = sites[i : i + chunk, 0], sites[i : i + chunk, 1]
        s = split.structures
        sub = Structures(s.pos[g], s.types[g], s.box[g])
        f = local_features(sub, sub.pos[torch.arange(len(g)), a], None, r_loc)
        feats = f if feats is None else feats.extend(f)
    return feats


def wasserstein1(x: Tensor, y: Tensor) -> float:
    """1D Wasserstein-1 distance between two empirical samples."""
    return float(wasserstein_distance(x.detach().cpu().numpy(), y.detach().cpu().numpy()))


def _pooled_w1(items: list[Tensor], ref: Tensor) -> float:
    return wasserstein1(torch.cat(items), ref) if items else float("nan")


def local_realism(
    gen: LocalFeatures, ref: LocalFeatures, n_resamples: int = 1000, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """W1 per feature (pooled over atoms, bootstrapped over samples) and tail metrics."""
    out = {}
    lists = {"pe_atom": (gen.pe_atom, ref.pe_atom), "bond_angle": (gen.bond_angle, ref.bond_angle)}
    lists.update({f"pair_{k}": (gen.pair_dist[k], ref.pair_dist[k]) for k in SPECIES_PAIRS})
    lists["relax_disp"] = (gen.relax_disp, ref.relax_disp)
    for name, (g, r) in lists.items():
        ref_pool = torch.cat(r)
        out[f"w1_{name}"] = bootstrap_ci_items(
            g, lambda sel, ref_pool=ref_pool: _pooled_w1(sel, ref_pool), n_resamples, seed=seed
        )
    pe_hi = torch.quantile(torch.cat(ref.pe_atom), 0.99)
    frac_hi = np.array([float((p > pe_hi).double().mean()) if len(p) else 0.0 for p in gen.pe_atom])
    out["frac_pe_above_ref_p99"] = bootstrap_ci(frac_hi, seed=seed)
    out["min_pair_dist"] = bootstrap_ci(gen.min_pair_dist.cpu().numpy(), seed=seed)
    out["overlap_rate"] = bootstrap_ci(
        (gen.min_pair_dist < OVERLAP_DIST).double().cpu().numpy(), seed=seed
    )
    return out


def global_realism(gen: Structures, ref: Structures) -> dict[str, float]:
    """g_ab(r) L1 distances, per-atom PE W1, and crystallinity (global |psi6|)."""
    out = {}
    for name, pair in SPECIES_PAIRS.items():
        r, g_gen = radial_distribution(gen, pair, r_max=5.0)
        _, g_ref = radial_distribution(ref, pair, r_max=5.0)
        out[f"gr_l1_{name}"] = float((g_gen - g_ref).abs().sum() * (r[1] - r[0]))
    out["w1_pe_atom"] = wasserstein1(
        ka_energy(gen, per_atom=True).flatten(), ka_energy(ref, per_atom=True).flatten()
    )
    out["pe_atom_mean"] = float(ka_energy(gen, per_atom=True).mean())
    psi = psi6_global(gen)
    out["psi6_global_mean"] = float(psi.mean())
    out["crystallized_fraction"] = float((psi > 0.3).double().mean())
    return out


def window_descriptors(f: LocalFeatures) -> Tensor:
    """[B, D] fixed-length descriptor per window: normalised histograms of pair distances
    (per species pair) and bond angles."""
    rows = []
    for b in range(len(f)):
        parts = []
        for k in SPECIES_PAIRS:
            h = torch.histc(f.pair_dist[k][b].float(), bins=20, min=0.6, max=2.5)
            parts.append(h / max(float(h.sum()), 1.0))
        h = torch.histc(f.bond_angle[b].float(), bins=18, min=0.0, max=float(np.pi))
        parts.append(h / max(float(h.sum()), 1.0))
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def diversity(gen: LocalFeatures, ref: LocalFeatures, train: LocalFeatures) -> dict[str, float]:
    """Spread of generated windows relative to natural ones, and a memorization check:
    nearest-training-window distance of generated windows relative to held-out natural ones."""
    dg, dr, dt = window_descriptors(gen), window_descriptors(ref), window_descriptors(train)
    spread = lambda d: float(torch.cdist(d, d).sum() / max(len(d) * (len(d) - 1), 1))  # noqa: E731
    nn_gen = torch.cdist(dg, dt).min(dim=1).values
    nn_ref = torch.cdist(dr, dt).min(dim=1).values
    return {
        "spread_ratio": spread(dg) / spread(dr),
        "nn_train_ratio": float(nn_gen.median() / nn_ref.median()),
    }


def cost_per_success(seconds: float, n_success: int) -> float:
    return float("inf") if n_success == 0 else seconds / n_success
