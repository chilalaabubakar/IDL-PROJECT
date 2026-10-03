"""Building defect requests, for conditional training and for evaluation.  [Tickets M-4, B-2]

Training (docs/PROJECT_PLAN.md §3.3): per structure, pick a request mode, pick a centre
atom (half the time a site of a uniformly chosen defect class), pin atoms according to the
mode, label the centre, and drop the label to NULL with probability ``p_uncond`` (CFG).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from glassdiff.data.patches import PatchLibrary
from glassdiff.types import Request, Structures


@dataclass
class RequestSamplerConfig:
    mode_probs: dict[str, float] = field(
        default_factory=lambda: {"centre": 0.5, "patch": 0.3, "host": 0.2}
    )
    p_defect_centre: float = 0.5  # otherwise a uniform random atom with its true label
    patch_radius: float = 1.5  # first shell
    host_r_out: float = 4.0  # Task B: pin every atom farther than this from the centre
    p_uncond: float = 0.15  # classifier-free guidance dropout (label -> NULL, pins kept)


class RequestSampler:
    """Draws training requests from labelled clean structures."""

    def __init__(self, cfg: RequestSamplerConfig) -> None:
        self.cfg = cfg

    def __call__(
        self, clean: Structures, labels: Tensor, generator: torch.Generator | None = None
    ) -> Request:
        raise NotImplementedError("Ticket M-4")


def make_eval_request(
    defect: int,
    target: Tensor,  # [B, 2]
    mode: str,  # "centre" | "patch" | "host"
    n_atoms: int,
    host: Structures | None = None,
    patches: PatchLibrary | None = None,
    generator: torch.Generator | None = None,
) -> Request:
    """Request for sampling. "patch" pastes a library patch (random rotation) at target;
    "host" pins every host atom farther than host_r_out from target (Task B)."""
    raise NotImplementedError("Ticket B-2")
