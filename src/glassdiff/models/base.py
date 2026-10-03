"""The one interface every denoiser implements.

    displacement = model(s, sigma, pin, label)     # [B, N, 2]

The prediction is the displacement that was added to the clean positions (DM2's target);
the sampler subtracts it. ``sigma`` is the per-structure noise level (ignored by σ-blind
models). ``pin`` marks atoms held fixed; ``label`` carries LabelToken ids. Unconditional
models receive ``pin=None, label=None``.
"""

from __future__ import annotations

from torch import Tensor, nn

from glassdiff.types import Structures


class Denoiser(nn.Module):
    sigma_aware: bool = True

    def forward(
        self,
        s: Structures,
        sigma: Tensor,  # [B]
        pin: Tensor | None = None,  # [B, N] bool
        label: Tensor | None = None,  # [B, N] long
    ) -> Tensor:
        raise NotImplementedError
