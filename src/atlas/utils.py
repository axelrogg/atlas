import math


def is_number_valid(number: float | complex | int) -> bool:
    """Check if a number is finite and not NaN."""
    if isinstance(number, complex):
        return math.isfinite(number.real) and math.isfinite(number.imag)
    return math.isfinite(number)


def omega(f: float) -> float:
    return 2 * math.pi * f