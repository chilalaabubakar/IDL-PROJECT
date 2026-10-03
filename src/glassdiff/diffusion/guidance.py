"""Guidance.  [Tickets S-4, S-5]

Classifier-free guidance (Ho & Salimans 2022), with the label dropped but pins kept:
    d = (1 + w) * model(x, sigma, pin, label) - w * model(x, sigma, pin, NULL)
Classifier guidance (stretch): add s * sigma^2 * grad_x log p_phi(y_centre = k | x, sigma).
"""

from __future__ import annotations

import torch
from torch import Tensor

from glassdiff.models.base import Denoiser
from glassdiff.types import LabelToken, Structures


class CFGDenoiser(Denoiser):
    def __init__(self, model: Denoiser, w: float) -> None:
        super().__init__()
        self.model = model
        self.w = w
        self.sigma_aware = model.sigma_aware

    def forward(
        self, s: Structures, sigma: Tensor, pin: Tensor | None = None, label: Tensor | None = None
    ) -> Tensor:
        cond = self.model(s, sigma, pin, label)
        if self.w == 0 or label is None:
            return cond
        null = torch.where(
            label == LabelToken.UNLABELLED, label, torch.full_like(label, int(LabelToken.NULL))
        )
        return (1 + self.w) * cond - self.w * self.model(s, sigma, pin, null)


class ClassifierGuidance:
    """Noise-aware per-atom defect classifier + gradient guidance (stretch)."""

    def __init__(self, classifier, scale: float) -> None:
        raise NotImplementedError("Ticket S-5")
