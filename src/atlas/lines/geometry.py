import math
from dataclasses import dataclass

from atlas.core.utils import is_number_positive, is_real_number_valid


@dataclass(frozen=True, kw_only=True)
class Position:
    x: float  # m
    y: float  # m; Carson–Clem only uses distances between conductors, not depth

    def __post_init__(self) -> None:
        if not is_real_number_valid(self.x):
            raise ValueError(f"Position x must be a finite real number, got {self.x!r}.")
        if not is_real_number_valid(self.y):
            raise ValueError(f"Position y must be a finite real number, got {self.y!r}.")

    def distance_to(self, other: "Position") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


# Relative: circles closer than touching by less than this fraction of the sum
# of their radii still count as touching, so rounding in the position helpers
# doesn't make touching circles overlap.
# TODO (todo.txt, Cables): decide the overlap tolerance.
OVERLAP_TOLERANCE = 1e-9


# Whether the circles centred on a and b, with radii ra and rb (m), overlap.
# Touching is not overlapping.
def circles_overlap(a: Position, ra: float, b: Position, rb: float, tolerance: float = OVERLAP_TOLERANCE) -> bool:
    for what, r in (("ra", ra), ("rb", rb)):
        if not is_number_positive(r):
            raise ValueError(f"{what} must be a positive finite number, got {r!r}.")
    if not is_real_number_valid(tolerance) or not 0 <= tolerance < 1:
        raise ValueError(f"tolerance must be a real number in [0, 1), got {tolerance!r}.")
    touching = ra + rb
    return a.distance_to(b) < touching * (1 - tolerance)


# Three positions in a triangle with sides `spacing` (axis to axis). (x, y) is
# the midpoint between the two lower axes; the third sits above it. Order:
# top, bottom left, bottom right.
def trefoil(x: float, y: float, spacing: float) -> list[Position]:
    if not is_number_positive(spacing):
        raise ValueError(f"spacing must be a positive finite number, got {spacing!r}.")
    height = spacing * math.sin(math.pi / 3)
    return [
        Position(x=x, y=y + height),
        Position(x=x - spacing / 2, y=y),
        Position(x=x + spacing / 2, y=y),
    ]


# n positions in a row, `spacing` apart (axis to axis), centred on (x, y).
# angle is the direction of the row in radians: 0 is horizontal (left to
# right), pi/2 vertical (bottom to top).
def flat(x: float, y: float, spacing: float, n: int = 3, angle: float = 0.0) -> list[Position]:
    if not is_number_positive(spacing):
        raise ValueError(f"spacing must be a positive finite number, got {spacing!r}.")
    if not is_real_number_valid(angle):
        raise ValueError(f"angle must be a finite real number, got {angle!r}.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"A flat formation needs a positive integer number of positions, got {n!r}.")
    dx, dy = math.cos(angle), math.sin(angle)
    offset = (n - 1) / 2
    return [Position(x=x + (k - offset) * spacing * dx, y=y + (k - offset) * spacing * dy) for k in range(n)]
