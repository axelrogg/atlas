import math

import pytest

from atlas import Simulator, ElementKind
from atlas.core.model.elements import CCVS
from atlas._atlas import next_id, to_real_value


# Generated ids are the prefix followed by the next number after the last one
# used for that prefix.
def test_next_id_counts_up_from_the_last_number() -> None:
    assert next_id("R", 0, set()) == ("R1", 1)
    assert next_id("R", 4, set()) == ("R5", 5)


# Someone can add elements to atlas.circuit directly. A generated id must never
# collide with one that is already taken.
def test_next_id_skips_taken_ids() -> None:
    assert next_id("R", 0, {"R1", "R2"}) == ("R3", 3)


# R, L, C and the CCVS gain are stored as floats. Ints, floats and complex
# numbers with no imaginary part must all be accepted as real.
@pytest.mark.parametrize("value, expected", [(2, 2), (2.5, 2.5), (3 + 0j, 3.0)])
def test_to_real_value_accepts_real_numbers(value: complex, expected: float) -> None:
    assert to_real_value(value, ElementKind.RESISTOR) == expected


# A real imaginary part must be refused, not silently dropped, which would
# change the circuit without telling the user.
def test_to_real_value_rejects_a_non_zero_imaginary_part() -> None:
    with pytest.raises(ValueError):
        to_real_value(1 + 1j, ElementKind.RESISTOR)


# Users never choose ids. Each kind counts on its own, with its SPICE letter.
def test_ids_are_generated_per_kind() -> None:
    a = Simulator(f=50)
    ids = [
        a.add_element(ElementKind.VOLTAGE_SOURCE, 10).id,
        a.add_element(ElementKind.RESISTOR, 1.0).id,
        a.add_element(ElementKind.RESISTOR, 1.0).id,
        a.add_element(ElementKind.INDUCTOR, 1.0).id,
    ]
    assert ids == ["V1", "R1", "R2", "L1"]


# Users never create nodes either. An element added without terminals must get
# two new ones, with plain integer ids that never clash with ground ("0").
def test_unconnected_terminals_get_fresh_nodes() -> None:
    a = Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    assert (r.plus.id, r.minus.id) == ("1", "2")
    s = a.add_element(ElementKind.RESISTOR, 1.0, plus=r.minus)
    assert (s.plus.id, s.minus.id) == ("2", "3")


# Passing another element's terminal (or ground) must reuse that exact node, so
# the two elements are connected with no wire.
def test_terminals_passed_at_creation_are_shared() -> None:
    a = Simulator(f=50)
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=v.plus, minus=a.ground)
    assert r.plus is v.plus
    assert r.minus is a.ground


# A complex resistance would otherwise fail later with an unclear comparison
# error inside Resistor.
def test_real_valued_kinds_reject_complex_values() -> None:
    with pytest.raises(ValueError):
        Simulator(f=50).add_element(ElementKind.RESISTOR, 1 + 1j)


# A CCVS without a control can't be solved, and a control on any other kind
# would be silently ignored. Both are mistakes. The valid case must store the
# control's id.
def test_only_a_ccvs_takes_a_control() -> None:
    a = Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    with pytest.raises(ValueError):
        a.add_element(ElementKind.CCVS, 1.0)
    with pytest.raises(ValueError):
        a.add_element(ElementKind.RESISTOR, 1.0, control=r)
    h = a.add_element(ElementKind.CCVS, 2.0, control=r)
    assert isinstance(h, CCVS)
    assert h.control_id == "R1"


# Every Simulator generates the same ids (R1, node 1, ...). A node from one must
# not be accepted by another, even if a node with that id exists there.
def test_objects_from_another_atlas_are_rejected() -> None:
    a, b = Simulator(f=50), Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    with pytest.raises(ValueError):
        b.add_element(ElementKind.RESISTOR, 1.0, plus=r.plus)
    with pytest.raises(ValueError):
        b.connect(r.plus, b.ground)


# connect means "make these the same point". If they already are (the same
# node, or already joined through ground), it must do nothing instead of
# raising a wire-loop error, and it must not create a wire.
def test_connect_is_a_no_op_for_points_that_are_already_connected() -> None:
    a = Simulator(f=50)
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10)
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    assert a.connect(v.minus, a.ground) is not None
    assert a.connect(r.minus, a.ground) is not None
    assert a.connect(v.minus, r.minus) is None
    assert a.connect(v.plus, v.plus) is None
    assert len(a.circuit.wires) == 2


# Couplings take elements, not ids, and get a generated K id.
def test_couple_elements() -> None:
    a = Simulator(f=50)
    l1 = a.add_element(ElementKind.INDUCTOR, 1.0)
    l2 = a.add_element(ElementKind.INDUCTOR, 1.0)
    k = a.couple_elements(l1, l2, m=0.5)
    assert (k.id, k.a_id, k.b_id) == ("K1", "L1", "L2")


# A terminal left with nothing attached would float and make the next solve
# singular, so it must be removed. A terminal still shared with another element
# (the source's plus) must stay.
def test_delete_element_removes_its_unused_nodes_only() -> None:
    a = Simulator(f=50)
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=v.plus)
    a.delete_element(r)
    assert "R1" not in a.circuit.elements
    assert r.minus.id not in a.circuit.nodes
    assert v.plus.id in a.circuit.nodes


# A wire can be the last thing holding a node in place, after the node's element
# was deleted. Deleting that wire must clean up the node, and only that one.
def test_delete_wire_removes_nodes_left_with_nothing() -> None:
    a = Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    s = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    w = a.connect(r.plus, s.plus)
    assert w is not None
    a.delete_element(s)
    assert s.plus.id in a.circuit.nodes
    a.delete_wire(w)
    assert s.plus.id not in a.circuit.nodes
    assert r.plus.id in a.circuit.nodes


# Deleting something that is already gone is a bug in the caller and must not
# be silently ignored.
def test_deleting_twice_is_rejected() -> None:
    a = Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    s = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    w = a.connect(r.plus, s.plus)
    assert w is not None
    a.delete_wire(w)
    a.delete_element(r)
    with pytest.raises(ValueError):
        a.delete_wire(w)
    with pytest.raises(ValueError):
        a.delete_element(r)


# The circuit's CCVS guard must also protect users of the facade: control
# second is refused, CCVS first then control works.
def test_a_ccvs_control_cannot_be_deleted_first() -> None:
    a = Simulator(f=50)
    r = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    h = a.add_element(ElementKind.CCVS, 2.0, control=r, minus=a.ground)
    with pytest.raises(ValueError):
        a.delete_element(r)
    a.delete_element(h)
    a.delete_element(r)


# The README quickstart, built only through the public API: the divider must
# give 6 V and 2 A, in every element and every wire.
def test_solve_end_to_end() -> None:
    a = Simulator(f=50)
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10)
    r1 = a.add_element(ElementKind.RESISTOR, 2.0, plus=v.plus)
    r2 = a.add_element(ElementKind.RESISTOR, 3.0, minus=a.ground)
    w1 = a.connect(r1.minus, r2.plus)
    w2 = a.connect(v.minus, a.ground)
    assert w1 is not None and w2 is not None
    sol = a.solve()
    assert sol.voltages[r2.plus.id] == pytest.approx(6)
    assert sol.element_currents == pytest.approx({"V1": -2, "R1": 2, "R2": 2})
    # 2 A flows from R1 into R2, and 2 A returns from ground into V1's minus
    # terminal (against the wire's V1- -> ground direction, hence -2).
    assert sol.wire_currents == pytest.approx({w1.id: 2, w2.id: -2})


# The transformer built through the public API must give the same load voltage
# as the low-level engine test.
def test_solve_with_coupling_end_to_end() -> None:
    a = Simulator(f=1 / (2 * math.pi))
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    l1 = a.add_element(ElementKind.INDUCTOR, 1.0, plus=v.plus, minus=a.ground)
    l2 = a.add_element(ElementKind.INDUCTOR, 1.0, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=l2.plus, minus=a.ground)
    a.couple_elements(l1, l2, m=0.5)
    sol = a.solve()
    assert sol.voltages[r.plus.id] == pytest.approx(3.2 - 2.4j)


# Fixators and norators take no value, only a fixator takes a voltage and a
# current. Through the facade they get their own ids and groups, and the
# input/output case solves: 10 V and 2 A known at the input (2 A fed in, so the
# fixator's plus -> minus current is -2), 2 ohm, nothing known at the output.
def test_fixator_and_norator_through_the_simulator() -> None:
    s = Simulator(f=50)
    fx = s.add_element(ElementKind.FIXATOR, minus=s.ground, voltage=10, current=-2)
    r = s.add_element(ElementKind.RESISTOR, 2.0, plus=fx.plus)
    nr = s.add_element(ElementKind.NORATOR, plus=r.minus, minus=s.ground)
    assert (fx.id, nr.id) == ("FX1", "NR1")
    sol = s.solve()
    assert sol.voltages[nr.plus.id] == pytest.approx(6)
    assert sol.element_currents == pytest.approx({"FX1": -2, "R1": 2, "NR1": 2})


# Every kind except fixators and norators needs a value, and those two refuse
# one: a value they would never use is a mistake, not something to drop.
def test_value_is_required_except_for_fixators_and_norators() -> None:
    s = Simulator(f=50)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.RESISTOR)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.NORATOR, 123)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.FIXATOR, 1, voltage=1, current=0)


# A fixator needs both its voltage and its current; any other kind given a
# voltage or a current would silently ignore them, so it must be refused.
def test_only_a_fixator_takes_a_voltage_and_a_current() -> None:
    s = Simulator(f=50)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.FIXATOR, voltage=1)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.FIXATOR, current=1)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.RESISTOR, 1.0, voltage=5, current=1)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.NORATOR, current=1)


# Arguments are checked before an id is generated, so a rejected call doesn't
# leave a gap in the ids.
def test_a_rejected_call_does_not_use_up_an_id() -> None:
    s = Simulator(f=50)
    with pytest.raises(ValueError):
        s.add_element(ElementKind.RESISTOR, 1.0, voltage=5)
    assert s.add_element(ElementKind.RESISTOR, 1.0).id == "R1"


# The frequency is fixed when the simulator is created, so a bad one must be
# refused there, before anything is built at it.
@pytest.mark.parametrize("f", [0, -50, math.inf, math.nan])
def test_the_frequency_is_validated_at_creation(f: float) -> None:
    with pytest.raises(ValueError):
        Simulator(f=f)
