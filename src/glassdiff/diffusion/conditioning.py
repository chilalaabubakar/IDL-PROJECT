"""Conditioning strategies plugged into the sampler.  [Tickets S-2, S-3, S-4]

| Strategy        | Baseline | What it does                                                       |
|-----------------|----------|--------------------------------------------------------------------|
| Unconditional   | B1       | nothing (filter afterwards)                                        |
| NoisePatch      | B3       | initial state contains the patch / host atoms; nothing is pinned    |
| Clamp           | B4       | reset pinned atoms to their target positions after every step      |
| RePaint         | B5       | known atoms <- x0 + sigma_k * z each step; resample U times, jump j |
| PinnedLabel     | main     | pass pin + label to a trained model; pinned atoms never move        |

Classifier-free guidance wraps the model (diffusion/guidance.py) and combines with PinnedLabel.
"""

from __future__ import annotations

import torch
from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import Request, Structures


def _place_pinned(s: Structures, request: Request) -> Structures:
    pin = request.pin_mask[..., None]
    return s.with_pos(torch.where(pin, request.pin_pos.to(s.pos), s.pos))


class Strategy:
    """Hooks called by diffusion.sampler.sample. Override what you need."""

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return init

    def displacement(
        self, model: Denoiser, s: Structures, sigma: Tensor, request: Request | None
    ) -> Tensor:
        return model(s, sigma)

    def frozen(self, request: Request | None) -> Tensor | None:
        """[B, N] atoms that receive no sampler noise and must not move."""
        return None

    def after_step(
        self,
        s: Structures,
        sigma: float,
        request: Request | None,
        generator: torch.Generator | None = None,
    ) -> Structures:
        return s


class Unconditional(Strategy):
    pass


class NoisePatch(Strategy):
    """DM2 "pore" style: put the patch / host atoms in the starting state, pin nothing."""

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return _place_pinned(init, request)


class Clamp(Strategy):
    """Naive inpainting with an unconditional model: pinned atoms are reset every step."""

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return _place_pinned(init, request)

    def frozen(self, request: Request | None) -> Tensor | None:
        return request.pin_mask

    def after_step(self, s, sigma, request, generator=None) -> Structures:
        return _place_pinned(s, request)


class RePaint(Strategy):
    """Lugmayr et al. (2022) adapted to the variance-exploding score-dynamics sampler.

    Known (pinned) atoms are set to their target plus noise of the current level after every
    step, so the model sees them at the same noise level as everything else. With
    ``n_resample > 1`` the sampler jumps back ``jump`` steps (re-noising the unknown atoms
    from sigma_k up to sigma_{k-jump}) and repeats, up to ``n_resample`` times per segment.
    """

    def __init__(self, n_resample: int = 10, jump: int = 10) -> None:
        self.n_resample = n_resample
        self.jump = jump

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return _place_pinned(init, request)

    def after_step(self, s, sigma, request, generator=None) -> Structures:
        z = torch.randn(s.pos.shape, generator=generator, dtype=s.pos.dtype).to(s.pos.device)
        known = request.pin_pos.to(s.pos) + sigma * z
        return s.with_pos(torch.where(request.pin_mask[..., None], known, s.pos))


class PinnedLabel(Strategy):
    """Trained conditioning: the model receives pin flags and the centre's label token."""

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return _place_pinned(init, request)

    def displacement(self, model, s, sigma, request) -> Tensor:
        return model(s, sigma, request.pin_mask, request.label)

    def frozen(self, request: Request | None) -> Tensor | None:
        return request.pin_mask

    def after_step(self, s, sigma, request, generator=None) -> Structures:
        return _place_pinned(s, request)


def build_strategy(cfg: dict) -> Strategy:
    cfg = dict(cfg)
    name = cfg.pop("name")
    cfg.pop("guidance_w", None)  # handled by wrapping the model (diffusion/guidance.py)
    table = {
        "unconditional": Unconditional,
        "noise_patch": NoisePatch,
        "clamp": Clamp,
        "repaint": RePaint,
        "pinned_label": PinnedLabel,
    }
    if name not in table:
        raise ValueError(f"unknown strategy {name!r}")
    return table[name](**cfg)
