"""Shared test fixtures: small synthetic configurations (no dataset needed)."""

from __future__ import annotations

import math

import pytest
import torch

from glassdiff.types import Structures

# Test tensors are tiny; extra intra-op threads only add overhead (30x slower when the
# cores are busy), so keep the suite single-threaded.
torch.set_num_threads(1)


def random_structures(
    batch: int, n_atoms: int, rho: float = 1.2, seed: int = 0, dtype=torch.float64
) -> Structures:
    """Uniform random positions, 65:35 composition, square box at density rho."""
    g = torch.Generator().manual_seed(seed)
    side = math.sqrt(n_atoms / rho)
    box = torch.full((batch, 2), side, dtype=dtype)
    pos = torch.rand(batch, n_atoms, 2, generator=g, dtype=dtype) * side
    n_b = n_atoms - round(0.65 * n_atoms)
    types = torch.zeros(batch, n_atoms, dtype=torch.long)
    for b in range(batch):
        types[b, torch.randperm(n_atoms, generator=g)[:n_b]] = 1
    return Structures(pos, types, box)


def jittered_lattice(
    batch: int,
    n_side: int,
    rho: float = 1.2,
    jitter: float = 0.05,
    seed: int = 0,
    dtype=torch.float64,
) -> Structures:
    """Square lattice at density rho with small Gaussian jitter (no overlapping atoms)."""
    g = torch.Generator().manual_seed(seed)
    a = 1.0 / math.sqrt(rho)
    ij = torch.stack(torch.meshgrid(torch.arange(n_side), torch.arange(n_side), indexing="ij"), -1)
    base = (ij.reshape(-1, 2).to(dtype) + 0.5) * a
    n_atoms = n_side * n_side
    pos = base[None] + jitter * torch.randn(batch, n_atoms, 2, generator=g, dtype=dtype)
    box = torch.full((batch, 2), n_side * a, dtype=dtype)
    n_b = n_atoms - round(0.65 * n_atoms)
    types = torch.zeros(batch, n_atoms, dtype=torch.long)
    for b in range(batch):
        types[b, torch.randperm(n_atoms, generator=g)[:n_b]] = 1
    return Structures(pos, types, box)


def triangular_lattice(nx: int = 8, ny: int = 8, a: float = 1.1, dtype=torch.float64) -> Structures:
    """Perfect periodic triangular lattice of A atoms (ny even); every atom has 6 neighbours at a."""
    assert ny % 2 == 0
    pts = [
        ((i + 0.5 * (j % 2)) * a, j * a * math.sqrt(3) / 2) for j in range(ny) for i in range(nx)
    ]
    pos = torch.tensor(pts, dtype=dtype)[None]
    box = torch.tensor([[nx * a, ny * a * math.sqrt(3) / 2]], dtype=dtype)
    types = torch.zeros(1, nx * ny, dtype=torch.long)
    return Structures(pos, types, box)


@pytest.fixture
def make_random():
    return random_structures


@pytest.fixture
def make_lattice():
    return jittered_lattice


@pytest.fixture
def make_triangular():
    return triangular_lattice
