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
    Fixator,
    Impedance,
    Inductor,
    Norator,
    Resistor,
    VoltageSource,
)
from atlas.model.node import GROUND_NODE, Node
from atlas.utils import omega

P, M = Node("p", ""), Node("m", "")


# Group 2 form of a two-terminal impedance element: its current I(e) leaves p and
# enters m, and its branch row is V(p) - V(m) - Z * I(e) = 0.
def g2_entries(eid: str, z: complex) -> set[StampEntry]:
    return {
        StampEntry(KCL("p"), I(eid), 1),
        StampEntry(KCL("m"), I(eid), -1),
        StampEntry(Branch(eid), V("p"), 1),
        StampEntry(Branch(eid), V("m"), -1),
        StampEntry(Branch(eid), I(eid), -z),
    }


# Group 1 form of an admittance y between p and m: the current y * (V(p) - V(m))
# leaves p and enters m.
def g1_entries(y: complex) -> set[StampEntry]:
    return {
        StampEntry(KCL("p"), V("p"), y),
        StampEntry(KCL("p"), V("m"), -y),
        StampEntry(KCL("m"), V("p"), -y),
        StampEntry(KCL("m"), V("m"), y),
    }


# Kind values double as id prefixes. A repeated value would make Python treat
# two kinds as the same member, so every "kind is ..." check would quietly lie.
def test_kind_values_are_unique_id_prefixes() -> None:
    values = [k.value for k in ElementKind]
    assert len(values) == len(set(values))
    assert ElementKind.RESISTOR.value == "R"


# The name is a display label: it falls back to the id when empty, including
# after a rename to "", so an element always has something to show.
def test_name_defaults_to_id_and_can_be_renamed() -> None:
    r = Resistor("R1", P, M, 1.0)
    assert r.name == "R1"
    r.rename("load")
    assert r.name == "load"
    r.rename("")
    assert r.name == "R1"


# Name and group can change after creation. If they were part of equality or
# the hash, renaming or regrouping an element stored in a set or dict would
# make it impossible to find again.
def test_name_and_group_do_not_affect_equality() -> None:
    a = Resistor("R1", P, M, 1.0, name="a")
    b = Resistor("R1", P, M, 1.0, name="b", group=ElementGroup.G2)
    assert a == b
    assert hash(a) == hash(b)


# An element with both terminals on one node contributes nothing (or makes the
# matrix singular), so it is refused when it is created.
def test_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        Resistor("R1", P, P, 1.0)


# Negative, NaN or infinite R, L and C are not physical and would give
# meaningless matrices. This catches both the sign check and the finiteness
# check (NaN compares false with everything, so it needs its own guard).
@pytest.mark.parametrize("cls", [Resistor, Inductor, Capacitor])
@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf])
def test_passive_values_must_be_finite_and_non_negative(cls: type, value: float) -> None:
    with pytest.raises(ValueError):
        cls("X1", P, M, value)


# A negative real part means a passive impedance that generates power. A purely
# reactive impedance (zero real part) is fine.
def test_impedance_needs_non_negative_real_part() -> None:
    with pytest.raises(ValueError):
        Impedance("Z1", P, M, -1 + 1j)
    assert Impedance("Z1", P, M, 1j).z == 1j


# A source with a NaN value would silently turn the whole solution into NaN.
@pytest.mark.parametrize("cls", [VoltageSource, CurrentSource])
def test_source_values_must_be_finite(cls: type) -> None:
    with pytest.raises(ValueError):
        cls("S1", P, M, math.nan)


# These are the values the stamps are built from, so each formula is pinned:
# 1/R, jωL, jωC and 1/Z.
def test_impedance_and_admittance() -> None:
    f = 50
    assert Resistor("R1", P, M, 4.0).admittance == pytest.approx(0.25)
    assert Inductor("L1", P, M, 2.0).impedance(f) == pytest.approx(1j * omega(f) * 2.0)
    assert Capacitor("C1", P, M, 1e-3).admittance(f) == pytest.approx(1j * omega(f) * 1e-3)
    assert Impedance("Z1", P, M, 3 + 4j).admittance == pytest.approx(1 / (3 + 4j))


# A 0 Ω resistor, an inductor at DC and a capacitor at DC have no finite
# admittance or impedance. Asking for one must raise a clear error instead of
# dividing by zero.
def test_infinite_admittance_or_impedance_raises() -> None:
    with pytest.raises(ValueError):
        _ = Resistor("R1", P, M, 0.0).admittance
    with pytest.raises(ValueError):
        Inductor("L1", P, M, 1.0).admittance(0)
    with pytest.raises(ValueError):
        Capacitor("C1", P, M, 1.0).impedance(0)


# A voltage source's or CCVS's current can never be eliminated, so they must be
# G2 by default and refuse G1, both at creation and through regroup. Otherwise
# the layout would give them no Branch row while their stamp still writes one.
def test_sources_and_ccvs_are_fixed_to_g2() -> None:
    assert VoltageSource("V1", P, M, 1).group is ElementGroup.G2
    with pytest.raises(ValueError):
        VoltageSource("V1", P, M, 1, group=ElementGroup.G1)
    with pytest.raises(ValueError):
        VoltageSource("V1", P, M, 1).regroup(ElementGroup.G1)
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "R1", group=ElementGroup.G1)


# A CCVS needs a real control element that isn't itself (a source driven by its
# own current has no physical meaning), and a finite gain.
def test_ccvs_validation() -> None:
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "")
    with pytest.raises(ValueError):
        CCVS("H1", P, M, 1.0, "H1")
    with pytest.raises(ValueError):
        CCVS("H1", P, M, math.nan, "R1")


# Only elements whose branch row has the form V(p) - V(m) - Z * I = 0 can take a
# mutual term. Coupling into a source's branch row would change what the source
# means.
def test_only_passive_elements_are_couplable() -> None:
    assert all(cls.is_couplable for cls in (Resistor, Inductor, Capacitor, Impedance))
    assert not any(cls.is_couplable for cls in (VoltageSource, CurrentSource, CCVS))


# The resistor must write exactly the G1 admittance form or exactly the G2
# branch form, depending on its group. A wrong sign or a missing entry here
# gives wrong voltages everywhere it is used.
def test_resistor_stamps() -> None:
    r = Resistor("R1", P, M, 2.0)
    assert set(r.stamp(50).entries) == g1_entries(0.5)
    r.regroup(ElementGroup.G2)
    assert set(r.stamp(50).entries) == g2_entries("R1", 2.0)
    assert r.stamp(50).rhs == []


# G2 is the way to model a 0 Ω resistor. Its stamp must not compute the
# (infinite) admittance on the way, or G2 would be useless exactly where it is
# needed.
def test_zero_ohm_resistor_stamps_in_g2() -> None:
    r = Resistor("R0", P, M, 0.0, group=ElementGroup.G2)
    assert set(r.stamp(50).entries) == g2_entries("R0", 0.0)


# The G2 branch row of L and Z must use their impedance at the solve frequency
# (jωL for L, the fixed z for Z).
def test_inductor_and_impedance_g2_stamps_use_their_impedance() -> None:
    f = 50
    assert set(Inductor("L1", P, M, 0.1, group=ElementGroup.G2).stamp(f).entries) == g2_entries("L1", 1j * omega(f) * 0.1)
    assert set(Impedance("Z1", P, M, 1 + 2j, group=ElementGroup.G2).stamp(f).entries) == g2_entries("Z1", 1 + 2j)


# At DC a capacitor has infinite impedance, so its G2 (impedance-form) branch
# row can't be written. It must refuse with a clear error instead of dividing
# by zero.
def test_capacitor_g2_at_dc_raises() -> None:
    with pytest.raises(ValueError):
        Capacitor("C1", P, M, 1e-6, group=ElementGroup.G2).stamp(0)


# A voltage source fixes V(p) - V(m) = value through its branch row, and its
# unknown current links into both KCL rows. The value must go on the rhs of its
# own branch row.
def test_voltage_source_stamp() -> None:
    s = VoltageSource("V1", P, M, 10).stamp(50)
    assert set(s.entries) == {
        StampEntry(KCL("p"), I("V1"), 1),
        StampEntry(KCL("m"), I("V1"), -1),
        StampEntry(Branch("V1"), V("p"), 1),
        StampEntry(Branch("V1"), V("m"), -1),
    }
    assert s.rhs == [StampRHSEntry(Branch("V1"), 10)]


# In G1 a current source is a known current, so it only touches the rhs, with
# the sign that matches the plus -> minus convention. In G2 its current becomes
# an unknown pinned to its value by the branch row I = value.
def test_current_source_stamps() -> None:
    g1 = CurrentSource("I1", P, M, 2).stamp(50)
    assert g1.entries == []
    assert set(g1.rhs) == {StampRHSEntry(KCL("p"), -2), StampRHSEntry(KCL("m"), 2)}

    g2 = CurrentSource("I1", P, M, 2, group=ElementGroup.G2).stamp(50)
    assert StampEntry(Branch("I1"), I("I1"), 1) in g2.entries
    assert g2.rhs == [StampRHSEntry(Branch("I1"), 2)]


# The CCVS branch row V(p) - V(m) - gain * I(control) = 0 is what makes it
# "controlled": the control's current column must appear with -gain.
def test_ccvs_stamp_refers_to_its_control_current() -> None:
    s = CCVS("H1", P, GROUND_NODE, 4.0, "R1").stamp(50)
    assert StampEntry(Branch("H1"), I("R1"), -4.0) in s.entries
    assert StampEntry(Branch("H1"), V("p"), 1) in s.entries


# A fixator fixes the voltage across it and carries a known current, so it
# writes a branch equation V(p) - V(m) = voltage and puts its known current on
# the rhs like a G1 current source does (leaving p, arriving at m). It has no
# current column of its own.
def test_fixator_stamp() -> None:
    s = Fixator("FX1", P, M, voltage=10, current=2).stamp(50)
    assert set(s.entries) == {
        StampEntry(Branch("FX1"), V("p"), 1),
        StampEntry(Branch("FX1"), V("m"), -1),
    }
    assert set(s.rhs) == {
        StampRHSEntry(Branch("FX1"), 10),
        StampRHSEntry(KCL("p"), -2),
        StampRHSEntry(KCL("m"), 2),
    }


# A norator carries an unknown current and imposes nothing, so it only puts
# its own current column into the KCL rows of its two nodes.
def test_norator_stamp() -> None:
    s = Norator("NR1", P, M).stamp(50)
    assert set(s.entries) == {
        StampEntry(KCL("p"), I("NR1"), 1),
        StampEntry(KCL("m"), I("NR1"), -1),
    }
    assert s.rhs == []


# The layout reads these groups to give a fixator a row only and a norator a
# column only, so they must always be in FX and NR and never move.
def test_fixator_and_norator_are_fixed_to_their_groups() -> None:
    fx = Fixator("FX1", P, M, voltage=1, current=0)
    nr = Norator("NR1", P, M)
    assert fx.group is ElementGroup.FX
    assert nr.group is ElementGroup.NR
    with pytest.raises(ValueError):
        Fixator("FX1", P, M, voltage=1, current=0, group=ElementGroup.G2)
    with pytest.raises(ValueError):
        Norator("NR1", P, M, group=ElementGroup.G1)
    with pytest.raises(ValueError):
        fx.regroup(ElementGroup.G2)
    with pytest.raises(ValueError):
        nr.regroup(ElementGroup.G2)


# Any other element in FX or NR would get a row without a column (or the
# reverse) that its own stamp doesn't match. That must be refused both when it
# is created and when it is regrouped.
@pytest.mark.parametrize("group", [ElementGroup.FX, ElementGroup.NR])
def test_fx_and_nr_are_reserved_for_fixators_and_norators(group: ElementGroup) -> None:
    with pytest.raises(ValueError):
        Resistor("R1", P, M, 1.0, group=group)
    with pytest.raises(ValueError):
        Resistor("R1", P, M, 1.0).regroup(group)


# Like every other element, a fixator's voltage and current must be finite.
@pytest.mark.parametrize("voltage, current", [(math.nan, 0), (1, math.inf)])
def test_fixator_values_must_be_finite(voltage: float, current: float) -> None:
    with pytest.raises(ValueError):
        Fixator("FX1", P, M, voltage=voltage, current=current)
