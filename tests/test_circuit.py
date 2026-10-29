import pytest

from atlas.model.circuit import Circuit
from atlas.model.elements import CCVS, Inductor, Resistor, VoltageSource
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.model.node import GROUND_NODE, Node

N1, N2, N3 = Node("n1", ""), Node("n2", ""), Node("n3", "")


def test_new_circuit_has_only_ground() -> None:
    c = Circuit()
    assert c.ground is GROUND_NODE
    assert list(c.nodes) == ["0"]
    assert c.netblocks() == [["0"]]


def test_ground_cannot_be_added_or_deleted() -> None:
    c = Circuit()
    with pytest.raises(ValueError):
        c.add_node(Node("0", ""))
    with pytest.raises(ValueError):
        c.delete_node("0")


def test_duplicates_are_rejected() -> None:
    c = Circuit()
    c.add_node(N1)
    with pytest.raises(ValueError):
        c.add_node(N1)
    c.add_element(Resistor("R1", N1, N2, 1.0))
    with pytest.raises(ValueError):
        c.add_element(Resistor("R1", N2, N3, 1.0))


def test_wires_merge_nodes_into_nets_named_after_the_first_node() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, N2, 1.0))
    c.add_element(Resistor("R2", N3, GROUND_NODE, 1.0))
    c.add_wire(N3, N2)
    assert c.netblocks() == [["0"], ["n1"], ["n2", "n3"]]
    assert c.node_net("n3") == "n2"
    assert c.are_connected("n2", "n3")
    assert not c.are_connected("n1", "n2")


def test_deleting_a_wire_splits_the_net() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, N2, 1.0))
    w = c.add_wire(N2, N3)
    c.delete_wire(w.id)
    assert c.netblocks() == [["0"], ["n1"], ["n2"], ["n3"]]
    assert not c.are_connected("n2", "n3")


def test_wire_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        Circuit().add_wire(N1, N1)


def test_wire_loops_are_rejected() -> None:
    c = Circuit()
    c.add_wire(N1, N2)
    with pytest.raises(ValueError):
        c.add_wire(N2, N1)
    c.add_wire(N2, N3)
    with pytest.raises(ValueError):
        c.add_wire(N3, N1)


def test_a_deleted_wire_can_be_added_again() -> None:
    c = Circuit()
    w = c.add_wire(N1, N2)
    c.delete_wire(w.id)
    c.add_wire(N1, N2)
    assert c.are_connected("n1", "n2")


def test_delete_node_removes_its_wires_and_elements() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, N2, 1.0))
    c.add_element(Resistor("R2", N2, GROUND_NODE, 1.0))
    c.add_wire(N2, N3)
    c.delete_node("n2")
    assert c.elements == {}
    assert c.wires == {}
    assert "n2" not in c.nodes


def circuit_with_inductors() -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 1))
    c.add_element(Inductor("L1", N1, GROUND_NODE, 1.0))
    c.add_element(Inductor("L2", N2, GROUND_NODE, 1.0))
    c.add_element(Inductor("L3", N3, GROUND_NODE, 1.0))
    return c


def test_add_coupling() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    assert list(c.couplings) == ["K1"]


@pytest.mark.parametrize(
    "k",
    [
        MagneticCoupling("K1", "L1", "L9", m=0.5),
        MagneticCoupling("K1", "L1", "V1", m=0.5),
    ],
)
def test_coupling_to_unknown_or_non_couplable_elements_is_rejected(k: MagneticCoupling) -> None:
    with pytest.raises(ValueError):
        circuit_with_inductors().add_coupling(k)


def test_a_pair_can_only_be_coupled_once() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    with pytest.raises(ValueError):
        c.add_coupling(MagneticCoupling("K1", "L2", "L3", m=0.5))
    with pytest.raises(ValueError):
        c.add_coupling(MagneticCoupling("K2", "L2", "L1", m=0.5))


def test_deleting_an_element_deletes_its_couplings() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K12", "L1", "L2", m=0.5))
    c.add_coupling(MagneticCoupling("K13", "L1", "L3", m=0.5))
    c.add_coupling(MagneticCoupling("K23", "L2", "L3", m=0.5))
    c.delete_element("L1")
    assert list(c.couplings) == ["K23"]


def test_deleting_a_node_deletes_the_couplings_of_its_elements() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K23", "L2", "L3", m=0.5))
    c.delete_node("n3")
    assert c.couplings == {}


def test_delete_coupling() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    c.delete_coupling("K1")
    assert c.couplings == {}
    with pytest.raises(ValueError):
        c.delete_coupling("K1")


def circuit_with_ccvs() -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 1))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 1.0))
    c.add_element(CCVS("H1", N2, GROUND_NODE, 2.0, "R1"))
    return c


def test_a_ccvs_control_cannot_be_deleted_before_the_ccvs() -> None:
    c = circuit_with_ccvs()
    with pytest.raises(ValueError):
        c.delete_element("R1")
    with pytest.raises(ValueError):
        c.delete_node("n1")
    assert set(c.elements) == {"V1", "R1", "H1"}


def test_a_ccvs_control_can_be_deleted_after_the_ccvs() -> None:
    c = circuit_with_ccvs()
    c.delete_element("H1")
    c.delete_element("R1")
    assert list(c.elements) == ["V1"]


def test_a_ccvs_and_its_control_on_the_same_node_are_deleted_together() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, GROUND_NODE, 1.0))
    c.add_element(CCVS("H1", N1, N2, 2.0, "R1"))
    c.delete_node("n1")
    assert c.elements == {}
