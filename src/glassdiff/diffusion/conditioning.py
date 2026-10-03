"""Conditioning strategies plugged into the sampler.  [Tickets S-2, S-3]

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

from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import Request, Structures


class Strategy:
    """Hooks called by diffusion.sampler.sample. Override what you need."""

    def initialize(self, init: Structures, request: Request | None) -> Structures:
        return init

    def displacement(
        self, model: Denoiser, s: Structures, sigma: Tensor, request: Request | None
    ) -> Tensor:
        return model(s, sigma)

    def after_step(self, s: Structures, sigma: Tensor, request: Request | None) -> Structures:
        return s


class Unconditional(Strategy):
    pass


class NoisePatch(Strategy):
    def initialize(self, init: Structures, request: Request | None) -> Structures:
        raise NotImplementedError("Ticket S-2")


class Clamp(Strategy):
    def after_step(self, s: Structures, sigma: Tensor, request: Request | None) -> Structures:
        raise NotImplementedError("Ticket S-2")


class RePaint(Strategy):
    """Lugmayr et al. (2022) adapted to the variance-exploding score-dynamics sampler.
    Resampling changes the step loop, so the sampler calls ``resample`` when it is set."""

    def __init__(self, n_resample: int = 10, jump: int = 10) -> None:
        self.n_resample = n_resample
        self.jump = jump

    def after_step(self, s: Structures, sigma: Tensor, request: Request | None) -> Structures:
        raise NotImplementedError("Ticket S-3")


class PinnedLabel(Strategy):
    def initialize(self, init: Structures, request: Request | None) -> Structures:
        raise NotImplementedError("Ticket S-4")

    def displacement(
        self, model: Denoiser, s: Structures, sigma: Tensor, request: Request | None
    ) -> Tensor:
        raise NotImplementedError("Ticket S-4")
