import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from atlas.core.utils import is_number_positive, is_real_number_valid, omega
from atlas.lines.geometry import Position


MU_0 = 4 * math.pi * 1e-7  # H/m

# Carson's equivalent earth-return depth: D_e = DE_COEFFICIENT * sqrt(rho / f),
# in m. 660 is the reference script's value; textbooks round Carson's series
# to 658.5.
DE_COEFFICIENT = 660.0


# Carson's earth return: the current returning through the soil behaves like a
# resistance R_e = w mu0 / 8 (Ω/m) and a return conductor at depth D_e (m).
def earth_return(f: float, rho: float, de_coefficient: float = DE_COEFFICIENT) -> tuple[float, float]:
    if not is_number_positive(f):
        raise ValueError(f"f must be a positive finite number, got {f!r}.")
    if not is_number_positive(rho):
        raise ValueError(f"rho must be a positive finite number, got {rho!r}.")
    if not is_number_positive(de_coefficient):
        raise ValueError(f"de_coefficient must be a positive finite number, got {de_coefficient!r}.")
    return omega(f) * MU_0 / 8, de_coefficient * math.sqrt(rho / f)


# One conductor as Carson–Clem sees it: where its axis is, its geometric mean
# radius, and its resistance at the operating temperature and frequency.
@dataclass(frozen=True, kw_only=True)
class CarsonConductor:
    position: Position
    gmr: float         # m
    resistance: float  # Ω/m

    def __post_init__(self) -> None:
        if not is_number_positive(self.gmr):
            raise ValueError(f"gmr must be a positive finite number, got {self.gmr!r}.")
        # Zero is allowed, for an ideal conductor.
        if not is_real_number_valid(self.resistance) or self.resistance < 0:
            raise ValueError(f"resistance must be a finite non-negative number, got {self.resistance!r}.")


# Per-metre series impedance of conductors running side by side over earth,
# in the order given. Carson–Clem, with k = j w mu0 / 2pi:
#   self:   R + R_e + k ln(D_e / GMR)
#   mutual: R_e + k ln(D_e / d)
# d is the distance between the axes, unless `distances` gives it for that
# pair (by index, in either order): conductors on the same axis, like a core
# and its own screen, need it, since their axes are 0 apart.
def carson_impedance(
    conductors: Sequence[CarsonConductor],
    f: float,
    rho: float,
    de_coefficient: float = DE_COEFFICIENT,
    distances: Mapping[tuple[int, int], float] | None = None,
) -> NDArray[np.complex128]:
    re, de = earth_return(f, rho, de_coefficient)
    k = 1j * omega(f) * MU_0 / (2 * math.pi)
    overrides = {tuple(sorted(pair)): d for pair, d in (distances or {}).items()}
    n = len(conductors)
    for (a, b), d in overrides.items():
        if a == b or not 0 <= a < n or not 0 <= b < n:
            raise ValueError(f"distances: ({a}, {b}) is not a pair of different conductors among {n}.")
        if not is_number_positive(d):
            raise ValueError(f"distances: ({a}, {b}) must be a positive finite number, got {d!r}.")

    z = np.zeros((n, n), dtype=np.complex128)
    for a, ca in enumerate(conductors):
        z[a, a] = ca.resistance + re + k * math.log(de / ca.gmr)
        for b in range(a + 1, n):
            d = overrides.get((a, b), ca.position.distance_to(conductors[b].position))
            if d == 0:
                raise ValueError(f"Conductors {a} and {b} are on the same axis; give their distance in `distances`.")
            z[a, b] = z[b, a] = re + k * math.log(de / d)
    return z
