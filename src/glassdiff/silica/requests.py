"""Training requests for the 3D conditional denoiser (as data/requests.py in 2D).

A request picks a centre atom and a mode:
    centre  whole-box generation: only the centre atom is held fixed
    patch   the centre and every atom within ``patch_radius`` are held fixed
    host    the centre and every atom farther than R_out are held fixed (Task B); R_out is
            drawn from ``r_out_range`` per request, so the region size can be chosen at
            sampling time (a lesson from the 2D sensitivity checks)
The centre carries the label token of its own defect class (NONE included); every other atom
is UNLABELLED. With probability ``p_uncond`` the token becomes NULL (classifier-free
guidance). Half of the centres are natural defects, so rare classes are seen often.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from glassdiff.silica.defects import SilicaDefect
from glassdiff.silica.graph import minimum_image

UNLABELLED, NULL, FIRST_DEFECT_TOKEN = 0, 1, 2
NUM_LABEL_TOKENS = FIRST_DEFECT_TOKEN + len(SilicaDefect)
MODES = ("centre", "patch", "host")


def defect_token(c: int | Tensor) -> int | Tensor:
    return c + FIRST_DEFECT_TOKEN


@dataclass
class SilicaRequestConfig:
    mode_probs: dict = field(default_factory=lambda: {"centre": 0.5, "patch": 0.3, "host": 0.2})
    p_defect_centre: float = 0.5
    patch_radius: float = 3.3  # first two shells around a Si (Si-O 1.6 A, Si-Si 3.1 A)
    r_out_range: tuple[float, float] = (5.0, 8.0)
    p_uncond: float = 0.15


@dataclass
class SilicaRequest:
    pin: Tensor  # [B, N] bool
    label: Tensor  # [B, N] long tokens
    centre: Tensor  # [B] long
    defect: Tensor  # [B] long SilicaDefect of the centre
    r_out: Tensor  # [B] float (host mode; 0 otherwise)


class SilicaRequestSampler:
    def __init__(self, cfg: SilicaRequestConfig) -> None:
        self.cfg = cfg
        probs = torch.tensor([float(cfg.mode_probs.get(m, 0.0)) for m in MODES])
        self.mode_probs = probs / probs.sum()

    def __call__(
        self,
        pos: Tensor,  # [B, N, 3]
        box: Tensor,  # [B, 3]
        labels: Tensor,  # [B, N]
        generator: torch.Generator | None = None,
    ) -> SilicaRequest:
        b_size, n = labels.shape
        dev = pos.device
        cfg = self.cfg
        labels_cpu = labels.cpu()
        mode = torch.multinomial(self.mode_probs, b_size, replacement=True, generator=generator)
        centre = torch.randint(n, (b_size,), generator=generator)
        use_defect = torch.rand(b_size, generator=generator) < cfg.p_defect_centre
        for b in torch.nonzero(use_defect).flatten().tolist():
            present = torch.unique(labels_cpu[b][labels_cpu[b] > 0])
            if len(present) == 0:
                continue
            cls = present[int(torch.randint(len(present), (1,), generator=generator))]
            sites = torch.nonzero(labels_cpu[b] == cls).flatten()
            centre[b] = sites[int(torch.randint(len(sites), (1,), generator=generator))]
        lo, hi = cfg.r_out_range
        r_out = lo + (hi - lo) * torch.rand(b_size, generator=generator)
        drop = torch.rand(b_size, generator=generator) < cfg.p_uncond

        mode, centre, r_out, drop = mode.to(dev), centre.to(dev), r_out.to(dev), drop.to(dev)
        bidx = torch.arange(b_size, device=dev)
        dist = minimum_image(pos - pos[bidx, centre][:, None, :], box[:, None, :]).norm(dim=-1)
        pin = torch.zeros(b_size, n, dtype=torch.bool, device=dev)
        pin |= (mode == MODES.index("patch"))[:, None] & (dist <= cfg.patch_radius)
        is_host = mode == MODES.index("host")
        pin |= is_host[:, None] & (dist > r_out[:, None])
        pin[bidx, centre] = True

        defect = labels[bidx, centre]
        label = torch.full((b_size, n), UNLABELLED, dtype=torch.long, device=dev)
        label[bidx, centre] = torch.where(drop, torch.full_like(defect, NULL), defect_token(defect))
        return SilicaRequest(pin, label, centre, defect, torch.where(is_host, r_out, 0.0))
