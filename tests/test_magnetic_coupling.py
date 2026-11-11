import math

import pytest

from atlas.mna.labels import Branch, I
from atlas.mna.stamp import StampEntry
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.utils import omega


# A coupling is given by exactly one of m (mutual inductance) or z (mutual
# impedance). Both or neither is ambiguous, and NaN or inf would poison the
# matrix, so all of these must be refused.
@pytest.mark.parametrize(
    "m, z",
    [(0.5, 1j), (None, None), (math.nan, None), (None, complex(math.inf, 0))],
)
def test_invalid_values_are_rejected(m: float | None, z: complex | None) -> None:
    with pytest.raises(ValueError):
        MagneticCoupling("K1", "L1", "L2", m=m, z=z)


# Coupling an element with itself would add a second self-impedance term to its
# own branch row, which isn't a coupling at all.
def test_element_cannot_be_coupled_with_itself() -> None:
    with pytest.raises(ValueError):
        MagneticCoupling("K1", "L1", "L1", m=0.5)


# Given as m, the coupling is inductive, so its impedance must follow the solve
# frequency (jωM). This keeps the circuit frequency-independent.
def test_impedance_from_m_depends_on_frequency() -> None:
    k = MagneticCoupling("K1", "L1", "L2", m=0.5)
    assert k.impedance(50) == pytest.approx(1j * omega(50) * 0.5)


# Given as z (for example a cable's earth-return mutual impedance, already
# computed for a frequency), it must be used as is at any frequency.
def test_impedance_from_z_is_constant() -> None:
    k = MagneticCoupling("K1", "Z1", "Z2", z=0.1 + 0.4j)
    assert k.impedance(50) == k.impedance(60) == 0.1 + 0.4j


# The coupling adds -Zm * I(other) to each element's branch row, and nothing
# else. This is the whole mechanism: no new rows, no new nodes.
def test_stamp_writes_into_both_branch_rows() -> None:
    k = MagneticCoupling("K1", "Z1", "Z2", z=2j)
    s = k.stamp(50)
    assert set(s.entries) == {
        StampEntry(Branch("Z1"), I("Z2"), -2j),
        StampEntry(Branch("Z2"), I("Z1"), -2j),
    }
    assert s.rhs == []
