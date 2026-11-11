import math

import pytest

from atlas.core.utils import is_number_valid, omega


# Every element validates its value with is_number_valid, so ordinary finite
# numbers of every type a user might pass (int, float, complex) must be accepted.
@pytest.mark.parametrize("number", [0, -3, 2.5, 1 + 2j, complex(0, -1)])
def test_finite_numbers_are_valid(number: complex) -> None:
    assert is_number_valid(number)


# NaN and infinity would poison the whole matrix and only show up as nonsense
# results, so they must be caught here, including inside either part of a complex.
@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf, complex(math.nan, 0), complex(0, math.inf)])
def test_nan_and_inf_are_invalid(number: complex) -> None:
    assert not is_number_valid(number)


# omega converts Hz to rad/s for every frequency-dependent impedance (jωL, jωC,
# jωM); a wrong factor here would scale every reactive result.
def test_omega_is_angular_frequency() -> None:
    assert omega(50) == pytest.approx(2 * math.pi * 50)
    assert omega(0) == 0
