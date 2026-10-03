"""Structure statistics for realism metrics.  [Ticket E-4]"""

from __future__ import annotations

from torch import Tensor

from glassdiff.types import Structures


def radial_distribution(
    s: Structures, pair: tuple[int, int], r_max: float = 5.0, n_bins: int = 200
) -> tuple[Tensor, Tensor]:
    """Partial g_ab(r) averaged over the batch: (bin centres [n_bins], g [n_bins])."""
    raise NotImplementedError("Ticket E-4")


def pair_distances(s: Structures, pair: tuple[int, int], r_max: float = 2.5) -> list[Tensor]:
    """Per structure: all a-b pair distances below r_max."""
    raise NotImplementedError("Ticket E-4")


def bond_angles(s: Structures, cn_factor: float = 1.4) -> list[Tensor]:
    """Per structure: angles (radians) between first-shell bonds at every atom."""
    raise NotImplementedError("Ticket E-4")
