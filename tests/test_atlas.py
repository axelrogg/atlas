import math

import pytest

from atlas import Atlas, ElementKind
from atlas._atlas import next_id, to_real_value


def test_next_id_counts_up_from_the_last_number() -> None:
    assert next_id("R", 0, set()) == ("R1", 1)
    assert next_id("R", 4, set()) == ("R5", 5)


def test_next_id_skips_taken_ids() -> None:
    assert next_id("R", 0, {"R1", "R2"}) == ("R3", 3)


@pytest.mark.parametrize("value, expected", [(2, 2), (2.5, 2.5), (3 + 0j, 3.0)])
def test_to_real_value_accepts_real_numbers(value: complex, expected: float) -> None:
    assert to_real_value(value, ElementKind.RESISTOR) == expected


def test_to_real_value_rejects_a_non_zero_imaginary_part() -> None:
    with pytest.raises(ValueError):
        to_real_value(1 + 1j, ElementKind.RESISTOR)


def test_ids_are_generated_per_kind() -> None:
    a = Atlas()
    ids = [
        a.add_element(ElementKind.VOLTAGE_SOURCE, 10).id,
        a.add_element(ElementKind.RESISTOR, 1.0).id,
        a.add_element(ElementKind.RESISTOR, 1.0).id,
        a.add_element(ElementKind.INDUCTOR, 1.0).id,
    ]
    assert ids == ["V1", "R1", "R2", "L1"]


def test_unconnected_terminals_get_fresh_nodes() -> None:
    a = Atlas()
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    assert (r.plus.id, r.minus.id) == ("R1+", "R1-")


def test_terminals_passed_at_creation_are_shared() -> None:
    a = Atlas()
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=v.plus, minus=a.ground)
    assert r.plus is v.plus
    assert r.minus is a.ground


def test_real_valued_kinds_reject_complex_values() -> None:
    with pytest.raises(ValueError):
        Atlas().add_element(ElementKind.RESISTOR, 1 + 1j)


def test_only_a_ccvs_takes_a_control() -> None:
    a = Atlas()
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    with pytest.raises(ValueError):
        a.add_element(ElementKind.CCVS, 1.0)
    with pytest.raises(ValueError):
        a.add_element(ElementKind.RESISTOR, 1.0, control=r)
    h = a.add_element(ElementKind.CCVS, 2.0, control=r)
    assert h.control_id == "R1"


def test_objects_from_another_atlas_are_rejected() -> None:
    a, b = Atlas(), Atlas()
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    with pytest.raises(ValueError):
        b.add_element(ElementKind.RESISTOR, 1.0, plus=r.plus)
    with pytest.raises(ValueError):
        b.connect(r.plus, b.ground)


def test_connect_is_a_no_op_for_points_that_are_already_connected() -> None:
    a = Atlas()
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10)
    r = a.add_element(ElementKind.RESISTOR, 1.0)
    assert a.connect(v.minus, a.ground) is not None
    assert a.connect(r.minus, a.ground) is not None
    assert a.connect(v.minus, r.minus) is None
    assert a.connect(v.plus, v.plus) is None
    assert len(a.circuit.wires) == 2


def test_couple_elements() -> None:
    a = Atlas()
    l1 = a.add_element(ElementKind.INDUCTOR, 1.0)
    l2 = a.add_element(ElementKind.INDUCTOR, 1.0)
    k = a.couple_elements(l1, l2, m=0.5)
    assert (k.id, k.a_id, k.b_id) == ("K1", "L1", "L2")


def test_delete_element_removes_its_unused_nodes_only() -> None:
    a = Atlas()
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=v.plus)
    a.delete_element(r)
    assert "R1" not in a.circuit.elements
    assert "R1-" not in a.circuit.nodes
    assert "V1+" in a.circuit.nodes


def test_delete_wire_removes_nodes_left_with_nothing() -> None:
    a = Atlas()
    r = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    s = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    w = a.connect(r.plus, s.plus)
    assert w is not None
    a.delete_element(s)
    assert "R2+" in a.circuit.nodes
    a.delete_wire(w)
    assert "R2+" not in a.circuit.nodes
    assert "R1+" in a.circuit.nodes


def test_deleting_twice_is_rejected() -> None:
    a = Atlas()
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


def test_a_ccvs_control_cannot_be_deleted_first() -> None:
    a = Atlas()
    r = a.add_element(ElementKind.RESISTOR, 1.0, minus=a.ground)
    h = a.add_element(ElementKind.CCVS, 2.0, control=r, minus=a.ground)
    with pytest.raises(ValueError):
        a.delete_element(r)
    a.delete_element(h)
    a.delete_element(r)


def test_solve_end_to_end() -> None:
    a = Atlas()
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10)
    r1 = a.add_element(ElementKind.RESISTOR, 2.0, plus=v.plus)
    r2 = a.add_element(ElementKind.RESISTOR, 3.0, minus=a.ground)
    a.connect(r1.minus, r2.plus)
    a.connect(v.minus, a.ground)
    sol = a.solve(f=50)
    assert sol.voltages[r2.plus.id] == pytest.approx(6)
    assert sol.element_currents == pytest.approx({"V1": -2, "R1": 2, "R2": 2})


def test_solve_with_coupling_end_to_end() -> None:
    a = Atlas()
    v = a.add_element(ElementKind.VOLTAGE_SOURCE, 10, minus=a.ground)
    l1 = a.add_element(ElementKind.INDUCTOR, 1.0, plus=v.plus, minus=a.ground)
    l2 = a.add_element(ElementKind.INDUCTOR, 1.0, minus=a.ground)
    r = a.add_element(ElementKind.RESISTOR, 1.0, plus=l2.plus, minus=a.ground)
    a.couple_elements(l1, l2, m=0.5)
    sol = a.solve(f=1 / (2 * math.pi))
    assert sol.voltages[r.plus.id] == pytest.approx(3.2 - 2.4j)
