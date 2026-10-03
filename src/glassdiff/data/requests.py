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

from glassdiff.data.patches import PatchLibrary
from glassdiff.geometry import minimum_image
from glassdiff.types import (
    DefectClass,
    LabelToken,
    Request,
    Species,
    Structures,
    defect_token,
)

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


DEFECT_SPECIES = {
    DefectClass.D1_MINUS: Species.A,
    DefectClass.D1_PLUS: Species.A,
    DefectClass.D2: Species.B,
}


def make_eval_request(
    defect: int,
    mode: str,
    types: Tensor,  # [B, N]
    box: Tensor,  # [B, 2]
    *,
    host: Structures | None = None,
    patches: PatchLibrary | None = None,
    target: Tensor | None = None,  # [B, 2]
    host_r_out: float = 4.0,
    generator: torch.Generator | None = None,
) -> Request:
    """Request "defect class k at location p" for sampling.

    Task A ("centre", "patch"): the box is generated from scratch; p defaults to the box
    centre. "centre" pins one atom of the defect's species at p; "patch" also pastes a
    randomly rotated library patch around it, using atoms of matching species.
    Task B ("host"): ``host`` glasses supply the frame. The centre is the host atom of the
    defect's species nearest p (p defaults to a random point); every host atom farther
    than ``host_r_out`` from it is pinned, and the disk inside is regenerated.
    """
    b_size, n = types.shape
    dev, dtype = box.device, box.dtype
    species = int(DEFECT_SPECIES[DefectClass(defect)])
    label = torch.full((b_size, n), int(LabelToken.UNLABELLED), dtype=torch.long, device=dev)
    pin = torch.zeros(b_size, n, dtype=torch.bool, device=dev)
    pin_pos = torch.zeros(b_size, n, 2, dtype=dtype, device=dev)
    centre = torch.zeros(b_size, dtype=torch.long, device=dev)

    if mode in ("centre", "patch"):
        target = box / 2 if target is None else target.to(dtype)
        for b in range(b_size):
            free = {sp: torch.nonzero(types[b] == sp).flatten().tolist() for sp in (0, 1)}
            centre[b] = free[species].pop(0)
            pin[b, centre[b]] = True
            pin_pos[b, centre[b]] = target[b]
            if mode == "patch":
                patch = patches.sample(defect, generator=generator)
                for rel, sp in zip(patch.rel_pos[1:], patch.types[1:].tolist()):
                    j = free[sp].pop(0)
                    pin[b, j] = True
                    pin_pos[b, j] = target[b] + rel.to(dtype)
    elif mode == "host":
        if host is None:
            raise ValueError("host mode needs host structures")
        if target is None:
            u = torch.rand(b_size, 2, generator=generator, dtype=dtype).to(dev)
            target = u * box
        dist = minimum_image(host.pos - target[:, None, :], host.box).norm(dim=-1)
        dist = torch.where(host.types == species, dist, torch.full_like(dist, float("inf")))
        centre = dist.argmin(dim=-1)
        bidx = torch.arange(b_size, device=dev)
        target = host.pos[bidx, centre]
        pin = pin_mask_for_mode(
            host, centre, torch.full_like(centre, MODES.index("host")), 0.0, host_r_out
        )
        pin_pos = host.pos.clone()
    else:
        raise ValueError(f"unknown request mode {mode!r}")

    bidx = torch.arange(b_size, device=dev)
    label[bidx, centre] = defect_token(defect)
    return Request(
        target=target,
        defect=torch.full((b_size,), int(defect), dtype=torch.long, device=dev),
        pin_mask=pin,
        pin_pos=pin_pos,
        label=label,
    )


def host_init(
    host: Structures, request: Request, generator: torch.Generator | None = None
) -> Structures:
    """Task B starting state: pinned host atoms in place, the others uniform in the disk
    around the target whose radius is the largest unpinned-atom distance."""
    dist = minimum_image(host.pos - request.target[:, None, :], host.box).norm(dim=-1)
    radius = torch.where(request.pin_mask, torch.zeros_like(dist), dist).amax(-1)  # [B]
    shape = host.pos.shape[:2]
    r = radius[:, None] * torch.rand(shape, generator=generator, dtype=host.pos.dtype).sqrt()
    phi = 2 * torch.pi * torch.rand(shape, generator=generator, dtype=host.pos.dtype)
    disk = request.target[:, None, :] + torch.stack([r * phi.cos(), r * phi.sin()], dim=-1)
    pos = torch.where(request.pin_mask[..., None], request.pin_pos, disk)
    return host.with_pos(pos)
