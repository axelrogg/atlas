import math


def is_number_valid(number: float | complex | int) -> bool:
    """Check if a number is finite and not NaN."""
    # Type hints aren't enforced at runtime: refuse anything that isn't a
    # number, and bool too, since it is an int and True would pass as 1.
    if isinstance(number, bool) or not isinstance(number, (int, float, complex)):
        return False
    if isinstance(number, complex):
        return math.isfinite(number.real) and math.isfinite(number.imag)
    return math.isfinite(number)


def is_real_number_valid(number: float | int) -> bool:
    return not isinstance(number, complex) and is_number_valid(number)


def is_number_positive(number: float | int) -> bool:
    return is_real_number_valid(number) and number > 0


def omega(f: float) -> float:
    return 2 * math.pi * f
