"""Tests for Ticket M-4 (training request sampler)."""

import torch

from glassdiff.data.requests import MODES, RequestSampler, RequestSamplerConfig
from glassdiff.geometry import minimum_image
from glassdiff.types import DefectClass, LabelToken, defect_token


def test_request_frequencies_and_pins(make_random):
    s = make_random(400, 64, seed=0)
    labels = torch.zeros(400, 64, dtype=torch.long)
    labels[:, 5] = DefectClass.D2  # one defect site per glass
    cfg = RequestSamplerConfig(p_defect_centre=0.5, p_uncond=0.2)
    req = RequestSampler(cfg)(s, labels, generator=torch.Generator().manual_seed(0))

    bidx = torch.arange(400)
    centre = req.label.ne(LabelToken.UNLABELLED).float().argmax(-1)
    assert (req.label.ne(LabelToken.UNLABELLED).sum(-1) == 1).all()  # exactly one labelled atom
    assert torch.equal(req.target, s.pos[bidx, centre])
    assert req.pin_mask[bidx, centre].all()

    dropped = req.label[bidx, centre] == LabelToken.NULL
    assert 0.13 < dropped.float().mean() < 0.27
    kept = ~dropped
    assert torch.equal(req.label[bidx, centre][kept], defect_token(req.defect[kept]))
    # about half the centres are the defect site; the rest are uniform (1/64 of them hit it too)
    assert 0.42 < (centre == 5).float().mean() < 0.60

    dist = minimum_image(s.pos - req.target[:, None], s.box).norm(dim=-1)
    n_pinned = req.pin_mask.sum(-1)
    centre_only = n_pinned == 1
    host_like = req.pin_mask & (dist > cfg.host_r_out)
    # "centre" mode (p = 0.5), plus patches that happen to have no neighbours
    assert centre_only.float().mean() > 0.4
    assert host_like.any(-1).float().mean() > 0.1  # "host" mode (p = 0.2)
    assert len(MODES) == 3


def test_host_r_out_range_draws_one_radius_per_request(make_random):
    s = make_random(16, 64, seed=3)
    labels = torch.zeros(16, 64, dtype=torch.long)
    cfg = RequestSamplerConfig(mode_probs={"host": 1.0}, host_r_out_range=(1.5, 3.0))
    req = RequestSampler(cfg)(s, labels, generator=torch.Generator().manual_seed(0))
    d = minimum_image(s.pos - req.target[:, None, :], s.box).norm(dim=-1)
    free = torch.where(req.pin_mask, torch.zeros_like(d), d).amax(-1)  # largest free distance
    pinned = torch.where(req.pin_mask & (d > 0), d, torch.full_like(d, 99.0)).amin(-1)
    assert (free <= 3.0).all() and (pinned > 1.5).all()
    assert len(torch.unique((free * 100).round())) > 4  # radii differ between requests
