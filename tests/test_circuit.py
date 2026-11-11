import pytest

from atlas.core.model.circuit import Circuit
from atlas.core.model.elements import CCVS, Inductor, Resistor, VoltageSource
from atlas.core.model.magnetic_coupling import MagneticCoupling
from atlas.core.model.node import GROUND_NODE, Node

N1, N2, N3 = Node("n1", ""), Node("n2", ""), Node("n3", "")


# Every circuit must start with ground already in place and as its own net named
# "0". The layout relies on ground being registered first, which is what makes
# the ground net always be called "0".
def test_new_circuit_has_only_ground() -> None:
    c = Circuit()
    assert c.ground is GROUND_NODE
    assert list(c.nodes) == ["0"]
    assert c.netblocks() == [["0"]]


# Ground is the reference every voltage is measured against. Adding a second one
# or deleting it would leave the system without a reference.
def test_ground_cannot_be_added_or_deleted() -> None:
    c = Circuit()
    with pytest.raises(ValueError):
        c.add_node(Node("0", ""))
    with pytest.raises(ValueError):
        c.delete_node("0")


# Ids are how nodes and elements are looked up, deleted, and named in the
# matrix. A duplicate would silently replace or shadow the first one.
def test_duplicates_are_rejected() -> None:
    c = Circuit()
    c.add_node(N1)
    with pytest.raises(ValueError):
        c.add_node(N1)
    c.add_element(Resistor("R1", N1, N2, 1.0))
    with pytest.raises(ValueError):
        c.add_element(Resistor("R1", N2, N3, 1.0))


# A wire makes two nodes one net, named after whichever node was registered
# first (n2 here, through R1). Getting this wrong would put stamps on the wrong
# rows.
def test_wires_merge_nodes_into_nets_named_after_the_first_node() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, N2, 1.0))
    c.add_element(Resistor("R2", N3, GROUND_NODE, 1.0))
    c.add_wire(N3, N2)
    assert c.netblocks() == [["0"], ["n1"], ["n2", "n3"]]
    assert c.node_net("n3") == "n2"
    assert c.are_connected("n2", "n3")
    assert not c.are_connected("n1", "n2")


# A disjoint set can't un-merge, so deleting a wire marks the nets stale and the
# next read rebuilds them from the remaining wires. This checks that the split
# really shows up.
def test_deleting_a_wire_splits_the_net() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, N2, 1.0))
    w = c.add_wire(N2, N3)
    c.delete_wire(w.id)
    assert c.netblocks() == [["0"], ["n1"], ["n2"], ["n3"]]
    assert not c.are_connected("n2", "n3")


# A wire from a node to itself does nothing and is almost certainly a mistake.
def test_wire_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        Circuit().add_wire(N1, N1)


# Wires inside a net must form a tree; otherwise the currents in parallel ideal
# wires have no unique answer. Both a direct parallel wire and a ring must be
# refused.
def test_wire_loops_are_rejected() -> None:
    c = Circuit()
    c.add_wire(N1, N2)
    with pytest.raises(ValueError):
        c.add_wire(N2, N1)
    c.add_wire(N2, N3)
    with pytest.raises(ValueError):
        c.add_wire(N3, N1)


# After a delete the nets are stale and still contain the deleted wire's merge.
# The loop check must rebuild them first, or it would wrongly refuse this wire.
def test_a_deleted_wire_can_be_added_again() -> None:
    c = Circuit()
    w = c.add_wire(N1, N2)
    c.delete_wire(w.id)
    c.add_wire(N1, N2)
    assert c.are_connected("n1", "n2")


# Deleting a node must also delete everything attached to it, or wires and
# elements would point at a node that no longer exists.
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


# The basic case: two existing, couplable elements can be coupled.
def test_add_coupling() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    assert list(c.couplings) == ["K1"]


# A coupling must point at elements that exist, or its stamp would write into
# rows that aren't in the matrix. It must also refuse sources, whose branch row
# isn't an impedance equation.
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


# Two couplings on the same pair would silently add their impedances together.
# The pair check must work in either order (L1-L2 and L2-L1). A duplicate
# coupling id is also refused.
def test_a_pair_can_only_be_coupled_once() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    with pytest.raises(ValueError):
        c.add_coupling(MagneticCoupling("K1", "L2", "L3", m=0.5))
    with pytest.raises(ValueError):
        c.add_coupling(MagneticCoupling("K2", "L2", "L1", m=0.5))


# A coupling to a deleted element would point at nothing. Deleting an element
# must delete all of its couplings and leave the others alone.
def test_deleting_an_element_deletes_its_couplings() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K12", "L1", "L2", m=0.5))
    c.add_coupling(MagneticCoupling("K13", "L1", "L3", m=0.5))
    c.add_coupling(MagneticCoupling("K23", "L2", "L3", m=0.5))
    c.delete_element("L1")
    assert list(c.couplings) == ["K23"]


# Deleting a node deletes its elements indirectly. Their couplings must go too,
# through this second path.
def test_deleting_a_node_deletes_the_couplings_of_its_elements() -> None:
    c = circuit_with_inductors()
    c.add_coupling(MagneticCoupling("K23", "L2", "L3", m=0.5))
    c.delete_node("n3")
    assert c.couplings == {}


# A coupling can be removed on its own, and removing it twice is an error rather
# than a silent no-op.
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


# A CCVS that loses its control can't be solved. Deleting the control, directly
# or through its node, must be refused and leave the circuit unchanged.
def test_a_ccvs_control_cannot_be_deleted_before_the_ccvs() -> None:
    c = circuit_with_ccvs()
    with pytest.raises(ValueError):
        c.delete_element("R1")
    with pytest.raises(ValueError):
        c.delete_node("n1")
    assert set(c.elements) == {"V1", "R1", "H1"}


# The guard must not get in the way of the correct order: CCVS first, then its
# control.
def test_a_ccvs_control_can_be_deleted_after_the_ccvs() -> None:
    c = circuit_with_ccvs()
    c.delete_element("H1")
    c.delete_element("R1")
    assert list(c.elements) == ["V1"]


# When a node deletion removes both the CCVS and its control, nothing is left
# dangling, so it must be allowed.
def test_a_ccvs_and_its_control_on_the_same_node_are_deleted_together() -> None:
    c = Circuit()
    c.add_element(Resistor("R1", N1, GROUND_NODE, 1.0))
    c.add_element(CCVS("H1", N1, N2, 2.0, "R1"))
    c.delete_node("n1")
    assert c.elements == {}
