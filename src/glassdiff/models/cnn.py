"""A CNN finder: the glass drawn as an image, a U-Net, and the role read at each atom.

The comparison it serves (docs/POINT_DEFECTS.md, "Other finders"): an image model is what one
would apply to atomic-resolution microscope pictures of a 2D glass, which come as images, not
as atom lists. Does it find defects as well as the EGNN?

Design:

    image       each species is one channel; every atom is a small Gaussian blob centred on
                its exact position (blob width 0.15 sigma), so positions finer than a pixel
                still show in the image. grid x grid pixels over the periodic box (128 x 128:
                0.11 sigma per pixel)
    periodic    every convolution pads circularly, so the image wraps around exactly like the
                periodic box
    network     U-Net (Ronneberger et al., 2015): an encoder that halves the resolution
                `levels` times, a decoder back to full resolution, skip connections between
                them. With 4 levels the coarsest pixel is 1.8 sigma and the network sees about
                10 sigma around each atom, like the four-layer EGNN
    readout     the decoder's features at each atom's position (periodic bilinear sampling),
                with the atom's species, through a small MLP to the five role logits
    symmetry    a CNN is not rotation-invariant. A periodic square box keeps its shape only
                under the 8 rotations by 90 degrees and reflections (the group D4), so
                training draws one of these and a random shift for every glass
                (augment), and evaluation averages the logits over all 8 (tta), which makes
                the prediction exactly D4-invariant. The resolution is halved
                len(channels) - 1 times, so only shifts by whole coarsest pixels (1.8 sigma
                with the defaults) change nothing; other shifts change the output a little,
                and training with random shifts keeps that small.

The box must be square (it is: 14.6 sigma).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from glassdiff.models.base import Denoiser
from glassdiff.models.layers import mlp
from glassdiff.types import Structures


def rasterize(pos: Tensor, types: Tensor, box: Tensor, grid: int, blob: float, n_species: int):
    """[B, n_species, grid, grid] image: a Gaussian blob per atom, periodic."""
    b, n, _ = pos.shape
    h = box / grid  # [B, 2] pixel size
    pos = torch.remainder(pos, box[:, None, :])
    r = int(math.ceil(4 * blob / float(h.min())))  # weights beyond 4 widths are below 4e-4
    off = torch.arange(-r, r + 1, device=pos.device)
    off = torch.stack(torch.meshgrid(off, off, indexing="ij"), -1).reshape(-1, 2)  # [P, 2]
    centre = torch.floor(pos / h[:, None, :]).long()  # [B, N, 2]
    pix = centre[:, :, None, :] + off[None, None]  # [B, N, P, 2], may lie outside the grid
    d = (pix.to(pos.dtype) + 0.5) * h[:, None, None, :] - pos[:, :, None, :]
    w = torch.exp(-(d**2).sum(-1) / (2 * blob**2))  # [B, N, P]
    flat = (pix[..., 0] % grid) * grid + (pix[..., 1] % grid)  # wrap around the box
    flat = flat + (types[:, :, None] * grid * grid)  # channel of the atom's species
    img = torch.zeros(b, n_species * grid * grid, dtype=pos.dtype, device=pos.device)
    img.scatter_add_(1, flat.reshape(b, -1), w.reshape(b, -1))
    return img.view(b, n_species, grid, grid)


def sample_at_atoms(feat: Tensor, pos: Tensor, box: Tensor) -> Tensor:
    """Bilinear sampling of feat [B, C, G, G] at the atoms, periodic: [B, N, C]."""
    b, c, g, _ = feat.shape
    u = torch.remainder(pos, box[:, None, :]) / (box[:, None, :] / g) - 0.5  # pixel centres
    i0 = torch.floor(u)
    t = u - i0
    i0 = i0.long() % g
    i1 = (i0 + 1) % g
    f = feat.permute(0, 2, 3, 1)  # [B, G, G, C]
    bi = torch.arange(b, device=feat.device)[:, None]

    def at(ix, iy):
        return f[bi, ix, iy]  # [B, N, C]

    tx, ty = t[..., :1], t[..., 1:]
    return (
        at(i0[..., 0], i0[..., 1]) * (1 - tx) * (1 - ty)
        + at(i1[..., 0], i0[..., 1]) * tx * (1 - ty)
        + at(i0[..., 0], i1[..., 1]) * (1 - tx) * ty
        + at(i1[..., 0], i1[..., 1]) * tx * ty
    )


def d4(pos: Tensor, box: Tensor, k: Tensor, flip: Tensor) -> Tensor:
    """pos [B, N, 2] turned by k[b] x 90 degrees, then reflected where flip[b], inside the
    square periodic box: the 8 symmetries that keep the box as it is."""
    side = box[:, :1]  # [B, 1]
    x, y = pos[..., 0], pos[..., 1]
    k, flip = k[:, None], flip[:, None]
    out_x, out_y = x, y
    xq, yq = x, y
    for q in range(1, 4):  # (x, y) -> (y, L - x): a turn by 90 degrees
        xq, yq = yq, side - xq
        out_x, out_y = torch.where(k == q, xq, out_x), torch.where(k == q, yq, out_y)
    out_x = torch.where(flip, side - out_x, out_x)
    return torch.remainder(torch.stack([out_x, out_y], -1), box[:, None, :])


def conv_block(cin: int, cout: int) -> nn.Sequential:
    conv = dict(kernel_size=3, padding=1, padding_mode="circular")
    return nn.Sequential(
        nn.Conv2d(cin, cout, **conv),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, **conv),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(),
    )


class GridUNetClassifier(Denoiser):
    """Per-atom role logits [B, N, n_classes] from a periodic U-Net on the drawn glass."""

    sigma_aware = False

    def __init__(
        self,
        grid: int = 128,
        blob: float = 0.15,
        channels: tuple[int, ...] = (16, 32, 48, 64, 96),
        hidden: int = 64,
        n_species: int = 2,
        n_classes: int = 5,
        augment: bool = True,
        tta: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(channels)
        assert grid % 2 ** (len(channels) - 1) == 0, "grid must halve len(channels) - 1 times"
        self.grid, self.blob, self.n_species, self.n_classes = grid, blob, n_species, n_classes
        self.augment, self.tta = augment, tta
        self.enc = nn.ModuleList()
        cin = n_species
        for c in channels:
            self.enc.append(conv_block(cin, c))
            cin = c
        self.dec = nn.ModuleList(
            conv_block(channels[i + 1] + channels[i], channels[i])
            for i in reversed(range(len(channels) - 1))
        )
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.species = nn.Embedding(n_species, channels[0])
        self.head = mlp([2 * channels[0], hidden, n_classes])

    def _logits(self, pos: Tensor, types: Tensor, box: Tensor) -> Tensor:
        x = rasterize(pos, types, box, self.grid, self.blob, self.n_species)
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        for block, skip in zip(self.dec, reversed(skips[:-1])):
            x = block(torch.cat([self.up(x), skip], 1))
        feat = sample_at_atoms(x, pos, box)
        return self.head(torch.cat([feat, self.species(types)], -1))

    def forward(
        self,
        s: Structures,
        sigma: Tensor | None = None,
        pin: Tensor | None = None,
        label: Tensor | None = None,
    ) -> Tensor:
        pos, types, box = s.pos, s.types, s.box
        b = pos.shape[0]
        if self.training and self.augment:  # one random D4 element and shift per glass
            k = torch.randint(0, 4, (b,), device=pos.device)
            flip = torch.randint(0, 2, (b,), device=pos.device).bool()
            shift = torch.rand(b, 1, 2, device=pos.device, dtype=pos.dtype) * box[:, None, :]
            return self._logits(d4(pos, box, k, flip) + shift, types, box)
        if not self.training and self.tta:  # average over all 8: exactly D4-invariant
            out = 0
            for q in range(4):
                for f in (False, True):
                    k = torch.full((b,), q, device=pos.device)
                    flip = torch.full((b,), f, device=pos.device)
                    out = out + self._logits(d4(pos, box, k, flip), types, box)
            return out / 8
        return self._logits(pos, types, box)
