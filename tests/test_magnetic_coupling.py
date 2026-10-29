import math

import pytest

from atlas.mna.labels import Branch, I
from atlas.mna.stamp import StampEntry
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.utils import omega


@pytest.mark.parametrize(
    "kwargs",
    [dict(m=0.5, z=1j), dict(), dict(m=math.nan), dict(z=complex(math.inf, 0))],
)
def test_invalid_values_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MagneticCoupling("K1", "L1", "L2", **kwargs)


def test_element_cannot_be_coupled_with_itself() -> None:
    with pytest.raises(ValueError):
        MagneticCoupling("K1", "L1", "L1", m=0.5)


def test_impedance_from_m_depends_on_frequency() -> None:
    k = MagneticCoupling("K1", "L1", "L2", m=0.5)
    assert k.impedance(50) == pytest.approx(1j * omega(50) * 0.5)


def test_impedance_from_z_is_constant() -> None:
    k = MagneticCoupling("K1", "Z1", "Z2", z=0.1 + 0.4j)
    assert k.impedance(50) == k.impedance(60) == 0.1 + 0.4j


def test_stamp_writes_into_both_branch_rows() -> None:
    k = MagneticCoupling("K1", "Z1", "Z2", z=2j)
    s = k.stamp(50)
    assert set(s.entries) == {
        StampEntry(Branch("Z1"), I("Z2"), -2j),
        StampEntry(Branch("Z2"), I("Z1"), -2j),
    }
    assert s.rhs == []
