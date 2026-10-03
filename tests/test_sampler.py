"""Acceptance tests for Ticket S-1 (DM2 score-dynamics sampler), using an oracle denoiser."""

import pytest
import torch

from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, random_init, sample
from glassdiff.geometry import minimum_image
from glassdiff.models.base import Denoiser

pytestmark = pytest.mark.skip(reason="Ticket S-1: delete this line when implementing")


class Oracle(Denoiser):
    """Predicts exactly the displacement away from a known clean structure."""

    def __init__(self, clean):
        super().__init__()
        self.clean = clean

    def forward(self, s, sigma, pin=None, label=None):
        return minimum_image(s.pos - self.clean.pos, s.box)


def test_random_init_in_box(make_lattice):
    s = make_lattice(3, 8, seed=0)
    init = random_init(s.types, s.box, generator=torch.Generator().manual_seed(0))
    assert init.pos.shape == s.pos.shape
    assert (init.pos >= 0).all() and (init.pos < s.box[:, None, :]).all()


def test_oracle_recovers_clean_structure(make_lattice):
    clean = make_lattice(2, 8, seed=1)
    init = random_init(clean.types, clean.box, generator=torch.Generator().manual_seed(1))
    schedule = ScoreDynamicsSchedule(n_noisy=50, n_final=5, sigma_start=0.3)
    out, _ = sample(Oracle(clean), init, schedule, generator=torch.Generator().manual_seed(2))
    err = minimum_image(out.pos - clean.pos, clean.box).norm(dim=-1)
    assert err.max() < 1e-6
