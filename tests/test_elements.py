import math

import pytest

from atlas.mna.labels import KCL, Branch, I, V
from atlas.mna.stamp import StampEntry, StampRHSEntry
from atlas.model.elements import (
    CCVS,
    Capacitor,
    CurrentSource,
    ElementGroup,
    ElementKind,
    Impedance,
    Inductor,
    Resistor,
    VoltageSource,
)
from atlas.model.node import GROUND_NODE, Node
from atlas.utils import omega

P, M = Node("p", ""), Node("m", "")


def g2_entries(eid: str, z: complex) -> set[StampEntry]:
    return {
        StampEntry(KCL("p"), I(eid), 1),
        StampEntry(KCL("m"), I(eid), -1),
        StampEntry(Branch(eid), V("p"), 1),
        StampEntry(Branch(eid), V("m"), -1),
        StampEntry(Branch(eid), I(eid), -z),
    }


def g1_entries(y: complex) -> set[StampEntry]:
    return {
        StampEntry(KCL("p"), V("p"), y),
        StampEntry(KCL("p"), V("m"), -y),
        StampEntry(KCL("m"), V("p"), -y),
        StampEntry(KCL("m"), V("m"), y),
    }


def test_kind_values_are_unique_id_prefixes() -> None:
    values = [k.value for k in ElementKind]
    assert len(values) == len(set(values))
    assert ElementKind.RESISTOR.value == "R"


def test_name_defaults_to_id_and_can_be_renamed() -> None:
    r = Resistor("R1", P, M, 1.0)
    assert r.name == "R1"
    r.rename("load")
    assert r.name == "load"
    r.rename("")
    assert r.name == "R1"


def test_name_and_group_do_not_affect_equality() -> None:
    a = Resistor("R1", P, M, 1.0, name="a")
    b = Resistor("R1", P, M, 1.0, name="b", group=ElementGroup.G2)
    assert a == b
    assert hash(a) == hash(b)


def test_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        Resistor("R1", P, P, 1.0)


@pytest.mark.parametrize("cls", [Resistor, Inductor, Capacitor])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_passive_values_must_be_finite_and_non_negative(cls: type, value: float) -> None:
    with pytest.raises(ValueError):
        cls("X1", P, M, value)


def test_impedance_needs_non_negative_real_part() -> None:
    with pytest.raises(ValueError):
        Impedance("Z1", P, M, -1 + 1j)
    assert Impedance("Z1", P, M, 1j).z == 1j


@pytest.mark.parametrize("cls", [VoltageSource, CurrentSource])
def test_source_values_must_be_finite(cls: type) -> None:
    with pytest.raises(ValueError):
        cls("S1", P, M, math.nan)


def test_impedance_and_admittance() -> None:
    f = 50
    assert Resistor("R1", P, M, 4.0).admittance == pytest.approx(0.25)
    assert Inductor("L1", P, M, 2.0).impedance(f) == pytest.approx(1j * omega(f) * 2.0)
    assert Capacitor("C1", P, M, 1e-3).admittance(f) == pytest.approx(1j * omega(f) * 1e-3)
    assert Impedance("Z1", P, M, 3 + 4j).admittance == pytest.approx(1 / (3 + 4j))


def test_infinite_admittance_or_impedance_raises() -> None:
    with pytest.raises(ValueError):
        _ = Resistor("R1", P, M, 0.0).admittance
    with pytest.raises(ValueError):
        Inductor("L1", P, M, 1.0).admittance(0)
    with pytest.raises(ValueError):
        Capacitor("C1", P, M, 1.0).impedance(0)


def test_sources_and_ccvs_are_fixed_to_g2() -> None:
    assert VoltageSource("V1", P, M, 1).group is ElementGroup.G2
    with pytest.raises(ValueError):
        VoltageSource("V1", P, M, 1, group=ElementGroup.G1)
    with pytest.raises(ValueError):
        VoltageSource("V1", P, M, 1).regroup(ElementGroup.G1)
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "R1", group=ElementGroup.G1)


def test_ccvs_validation() -> None:
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "")
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "H1")
    with pytest.raises(ValueError):
        CCVS("H1", P, M, math.nan, "R1")


def test_only_passive_elements_are_couplable() -> None:
    assert all(cls.is_couplable for cls in (Resistor, Inductor, Capacitor, Impedance))
    assert not any(cls.is_couplable for cls in (VoltageSource, CurrentSource, CCVS))


def test_resistor_stamps() -> None:
    r = Resistor("R1", P, M, 2.0)
    assert set(r.stamp(50).entries) == g1_entries(0.5)
    r.regroup(ElementGroup.G2)
    assert set(r.stamp(50).entries) == g2_entries("R1", 2.0)
    assert r.stamp(50).rhs == []


def test_zero_ohm_resistor_stamps_in_g2() -> None:
    r = Resistor("R0", P, M, 0.0, group=ElementGroup.G2)
    assert set(r.stamp(50).entries) == g2_entries("R0", 0.0)


def test_inductor_and_impedance_g2_stamps_use_their_impedance() -> None:
    f = 50
    assert set(Inductor("L1", P, M, 0.1, group=ElementGroup.G2).stamp(f).entries) == g2_entries("L1", 1j * omega(f) * 0.1)
    assert set(Impedance("Z1", P, M, 1 + 2j, group=ElementGroup.G2).stamp(f).entries) == g2_entries("Z1", 1 + 2j)


def test_capacitor_g2_at_dc_raises() -> None:
    with pytest.raises(ValueError):
        Capacitor("C1", P, M, 1e-6, group=ElementGroup.G2).stamp(0)


def test_voltage_source_stamp() -> None:
    s = VoltageSource("V1", P, M, 10).stamp(50)
    assert set(s.entries) == {
        StampEntry(KCL("p"), I("V1"), 1),
        StampEntry(KCL("m"), I("V1"), -1),
        StampEntry(Branch("V1"), V("p"), 1),
        StampEntry(Branch("V1"), V("m"), -1),
    }
    assert s.rhs == [StampRHSEntry(Branch("V1"), 10)]


def test_current_source_stamps() -> None:
    g1 = CurrentSource("I1", P, M, 2).stamp(50)
    assert g1.entries == []
    assert set(g1.rhs) == {StampRHSEntry(KCL("p"), -2), StampRHSEntry(KCL("m"), 2)}

    g2 = CurrentSource("I1", P, M, 2, group=ElementGroup.G2).stamp(50)
    assert StampEntry(Branch("I1"), I("I1"), 1) in g2.entries
    assert g2.rhs == [StampRHSEntry(Branch("I1"), 2)]


def test_ccvs_stamp_refers_to_its_control_current() -> None:
    s = CCVS("H1", P, GROUND_NODE, 4.0, "R1").stamp(50)
    assert StampEntry(Branch("H1"), I("R1"), -4.0) in s.entries
    assert StampEntry(Branch("H1"), V("p"), 1) in s.entries
