"""Building defect requests, for conditional training and for evaluation.  [Tickets M-4, B-2]

Training (docs/PROJECT_PLAN.md §3.3): per structure, pick a request mode, pick a centre
atom (with probability p_defect_centre a site of a uniformly chosen defect class present in
that glass), pin atoms according to the mode, label the centre with its true class, and
drop the label to NULL with probability ``p_uncond`` (classifier-free guidance; pins kept).

Modes: "centre" pins only the centre atom; "patch" pins the centre and every atom within
``patch_radius``; "host" (Task B) pins the centre and every atom farther than
``host_r_out``, so the model regenerates the disk around the centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from glassdiff.geometry import minimum_image
from glassdiff.types import DefectClass, LabelToken, Request, Structures, defect_token

MODES = ("centre", "patch", "host")


@dataclass
class RequestSamplerConfig:
    mode_probs: dict[str, float] = field(
        default_factory=lambda: {"centre": 0.5, "patch": 0.3, "host": 0.2}
    )
    p_defect_centre: float = 0.5  # otherwise a uniform random atom with its true label
    patch_radius: float = 1.5  # first shell
    host_r_out: float = 4.0  # Task B: pin every atom farther than this from the centre
    p_uncond: float = 0.15  # classifier-free guidance dropout (label -> NULL, pins kept)


def pin_mask_for_mode(
    s: Structures, centre: Tensor, mode: Tensor, patch_radius: float, host_r_out: float
) -> Tensor:
    """[B, N] bool pins for per-structure centre indices [B] and mode indices [B] (into MODES)."""
    bidx = torch.arange(s.batch_size, device=s.pos.device)
    target = s.pos[bidx, centre]
    dist = minimum_image(s.pos - target[:, None, :], s.box).norm(dim=-1)
    pin = torch.zeros_like(dist, dtype=torch.bool)
    pin |= (mode == MODES.index("patch"))[:, None] & (dist <= patch_radius)
    pin |= (mode == MODES.index("host"))[:, None] & (dist > host_r_out)
    pin[bidx, centre] = True
    return pin


class RequestSampler:
    """Draws training requests from labelled clean structures."""

    def __init__(self, cfg: RequestSamplerConfig) -> None:
        self.cfg = cfg
        probs = torch.tensor([cfg.mode_probs.get(m, 0.0) for m in MODES], dtype=torch.float)
        self.mode_probs = probs / probs.sum()

    def __call__(
        self, clean: Structures, labels: Tensor, generator: torch.Generator | None = None
    ) -> Request:
        b_size, n = clean.types.shape
        dev = clean.pos.device
        labels_cpu = labels.cpu()
        mode = torch.multinomial(self.mode_probs, b_size, replacement=True, generator=generator)
        centre = torch.randint(n, (b_size,), generator=generator)
        use_defect = torch.rand(b_size, generator=generator) < self.cfg.p_defect_centre
        for b in torch.nonzero(use_defect).flatten().tolist():
            present = [
                c for c in DefectClass if c != DefectClass.NONE and (labels_cpu[b] == c).any()
            ]
            if not present:
                continue
            cls = present[int(torch.randint(len(present), (1,), generator=generator))]
            sites = torch.nonzero(labels_cpu[b] == cls).flatten()
            centre[b] = sites[int(torch.randint(len(sites), (1,), generator=generator))]
        drop = torch.rand(b_size, generator=generator) < self.cfg.p_uncond

        mode, centre, drop = mode.to(dev), centre.to(dev), drop.to(dev)
        bidx = torch.arange(b_size, device=dev)
        defect = labels[bidx, centre]
        label = torch.full((b_size, n), int(LabelToken.UNLABELLED), dtype=torch.long, device=dev)
        label[bidx, centre] = torch.where(
            drop, torch.full_like(defect, int(LabelToken.NULL)), defect_token(defect)
        )
        pin = pin_mask_for_mode(clean, centre, mode, self.cfg.patch_radius, self.cfg.host_r_out)
        return Request(
            target=clean.pos[bidx, centre],
            defect=defect,
            pin_mask=pin,
            pin_pos=clean.pos.clone(),
            label=label,
        )


def make_eval_request(
    defect: int,
    target: Tensor,  # [B, 2]
    mode: str,  # "centre" | "patch" | "host"
    n_atoms: int,
    host: Structures | None = None,
    patches=None,
    generator: torch.Generator | None = None,
) -> Request:
    """Request for sampling. "patch" pastes a library patch (random rotation) at target;
    "host" pins every host atom farther than host_r_out from target (Task B)."""
    raise NotImplementedError("Ticket B-2")
