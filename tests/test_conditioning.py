"""Tests for Tickets S-2..S-4 (conditioning strategies and CFG), using an oracle denoiser."""

import torch

from glassdiff.diffusion.conditioning import Clamp, NoisePatch, PinnedLabel, RePaint
from glassdiff.diffusion.guidance import CFGDenoiser
from glassdiff.diffusion.sampler import ScoreDynamicsSchedule, random_init, sample
from glassdiff.geometry import minimum_image
from glassdiff.models.base import Denoiser
from glassdiff.types import LabelToken, Request, defect_token


class Oracle(Denoiser):
    sigma_aware = False

    def __init__(self, clean):
        super().__init__()
        self.clean = clean

    def forward(self, s, sigma, pin=None, label=None):
        return minimum_image(s.pos - self.clean.pos, s.box)


def _request(clean, n_pin=5):
    b, n = clean.types.shape
    pin = torch.zeros(b, n, dtype=torch.bool)
    pin[:, :n_pin] = True
    label = torch.zeros(b, n, dtype=torch.long)
    label[:, 0] = defect_token(1)
    return Request(clean.pos[:, 0], torch.ones(b, dtype=torch.long), pin, clean.pos.clone(), label)


def _run(strategy, clean, request, n_noisy=40):
    init = random_init(clean.types, clean.box, generator=torch.Generator().manual_seed(0))
    schedule = ScoreDynamicsSchedule(n_noisy=n_noisy, n_final=3, sigma_start=0.3)
    gen = torch.Generator().manual_seed(1)
    out, _ = sample(Oracle(clean), init, schedule, strategy, request, generator=gen)
    return out


def test_pinned_atoms_end_at_targets(make_lattice):
    clean = make_lattice(2, 8, seed=0)
    req = _request(clean)
    for strategy in (Clamp(), PinnedLabel(), RePaint(n_resample=1), NoisePatch()):
        out = _run(strategy, clean, req)
        err = minimum_image(out.pos - clean.pos, clean.box).norm(dim=-1)
        assert err[req.pin_mask].max() < 1e-9, type(strategy).__name__
        assert err.max() < 1e-6  # the oracle recovers everything else too


def test_pinned_label_never_moves_pinned_atoms(make_lattice):
    clean = make_lattice(1, 8, seed=1)
    req = _request(clean)
    seen = []

    class Spy(Oracle):
        def forward(self, s, sigma, pin=None, label=None):
            seen.append(s.pos[pin].clone())
            return super().forward(s, sigma, pin, label) * (~pin)[..., None]

    init = random_init(clean.types, clean.box, generator=torch.Generator().manual_seed(0))
    sample(Spy(clean), init, ScoreDynamicsSchedule(n_noisy=10, n_final=2), PinnedLabel(), req)
    assert all(torch.equal(p, clean.pos[req.pin_mask]) for p in seen)


def test_repaint_resampling_runs_extra_steps(make_lattice):
    clean = make_lattice(1, 8, seed=2)
    req = _request(clean)
    calls = []

    class Counter(Oracle):
        def forward(self, *args, **kwargs):
            calls.append(1)
            return super().forward(*args, **kwargs)

    init = random_init(clean.types, clean.box, generator=torch.Generator().manual_seed(0))
    schedule = ScoreDynamicsSchedule(n_noisy=20, n_final=0)
    sample(Counter(clean), init, schedule, RePaint(n_resample=3, jump=5), req)
    assert len(calls) == 20 * 3  # every 5-step segment is run 3 times


def test_cfg_combination(make_lattice):
    s = make_lattice(1, 8, seed=3)
    label = torch.zeros(1, 64, dtype=torch.long)
    label[0, 0] = defect_token(2)

    class LabelSensitive(Denoiser):
        def forward(self, s, sigma, pin=None, label=None):
            out = torch.zeros_like(s.pos)
            out[..., 0] = (label != LabelToken.NULL).to(s.pos.dtype)  # 1 if conditioned
            return out

    base = LabelSensitive()
    assert torch.equal(CFGDenoiser(base, 0.0)(s, None, None, label), base(s, None, None, label))
    guided = CFGDenoiser(base, 2.0)(s, None, None, label)
    # labelled atom: (1 + w) * 1 - w * 0 = 3; unlabelled atoms: (1 + w) * 1 - w * 1 = 1
    assert guided[0, 0, 0] == 3.0 and (guided[0, 1:, 0] == 1.0).all()
