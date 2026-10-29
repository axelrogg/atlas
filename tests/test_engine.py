import math

import numpy as np
import pytest

from atlas.mna.engine import Engine
from atlas.model.circuit import Circuit
from atlas.model.elements import (
    CCVS,
    Capacitor,
    CurrentSource,
    ElementGroup,
    Impedance,
    Inductor,
    Resistor,
    VoltageSource,
)
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.model.node import GROUND_NODE, Node

N1, N2, N3 = Node("n1", ""), Node("n2", ""), Node("n3", "")
# The frequency at which ω = 1 rad/s, so jωL is just jL and results are easy to
# check by hand.
UNIT_OMEGA = 1 / (2 * math.pi)


# 10 V source, R1 = 2 Ω from n1 to n2, a wire n2-n3, R2 = 3 Ω from n3 to ground.
def divider(r1_group: ElementGroup = ElementGroup.G1) -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Resistor("R1", N1, N2, 2.0, group=r1_group))
    c.add_wire(N2, N3)
    c.add_element(Resistor("R2", N3, GROUND_NODE, 3.0))
    return c


# The whole pipeline (groups, layout, stamps, solve, results) on the simplest
# useful circuit: 6 V at the middle and 2 A everywhere. Running it with R1 in
# both groups checks that the choice of group doesn't change the answer.
@pytest.mark.parametrize("group", [ElementGroup.G1, ElementGroup.G2])
def test_voltage_divider(group: ElementGroup) -> None:
    sol = Engine(divider(group), 50).solve()
    assert sol.voltages["n1"] == pytest.approx(10)
    assert sol.voltages["n2"] == sol.voltages["n3"] == pytest.approx(6)
    assert sol.voltages["0"] == 0
    assert sol.element_currents == pytest.approx({"V1": -2, "R1": 2, "R2": 2})


# A circuit driven only by a current source: 2 A into 5 Ω gives 10 V. Checks
# the rhs path and the source's own current.
def test_current_source_into_a_resistor() -> None:
    c = Circuit()
    c.add_element(CurrentSource("I1", GROUND_NODE, N1, 2))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 5.0))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n1"] == pytest.approx(10)
    assert sol.element_currents == pytest.approx({"I1": 2, "R1": 2})


# G1 and G2 are two ways of writing the same physics. For L, C and Z, solving
# with either group must give the same voltage.
@pytest.mark.parametrize(
    "make",
    [
        lambda g: Inductor("X", N1, N2, 0.01, group=g),
        lambda g: Capacitor("X", N1, N2, 1e-4, group=g),
        lambda g: Impedance("X", N1, N2, 3 + 4j, group=g),
    ],
)
def test_g1_and_g2_stamps_give_the_same_answer(make) -> None:
    results = []
    for g in (ElementGroup.G1, ElementGroup.G2):
        c = Circuit()
        c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
        c.add_element(make(g))
        c.add_element(Resistor("R2", N2, GROUND_NODE, 3.0))
        results.append(Engine(c, 50).solve().voltages["n2"])
    assert results[0] == pytest.approx(results[1])


# A two-winding transformer: L1 = L2 = 1 H, M = 0.5 H, 1 Ω load, at ω = 1.
def transformer(l2_plus: Node, l2_minus: Node) -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Inductor("L1", N1, GROUND_NODE, 1.0))
    c.add_element(Inductor("L2", l2_plus, l2_minus, 1.0))
    c.add_element(Resistor("R", N2, GROUND_NODE, 1.0))
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    return c


# The load voltage 3.2 - 2.4j comes from solving the transformer equations by
# hand. This is the main check that coupling works end to end.
def test_coupled_inductors() -> None:
    sol = Engine(transformer(N2, GROUND_NODE), UNIT_OMEGA).solve()
    assert sol.voltages["n2"] == pytest.approx(3.2 - 2.4j)


# The dot is at each inductor's plus terminal. Reversing L2 reverses its
# winding, so the induced voltage must flip sign. This pins the dot convention.
def test_flipping_a_coupled_inductor_flips_the_mutual_voltage() -> None:
    sol = Engine(transformer(GROUND_NODE, N2), UNIT_OMEGA).solve()
    assert sol.voltages["n2"] == pytest.approx(-3.2 + 2.4j)


# A coupling given as a complex mutual impedance (the cable case) between two
# impedances, compared against the same two equations solved directly.
def test_mutual_impedance_coupling() -> None:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Impedance("Z1", N1, GROUND_NODE, 0.2 + 1j))
    c.add_element(Impedance("Z2", N2, GROUND_NODE, 0.3 + 1j))
    c.add_element(Resistor("R", N2, GROUND_NODE, 1.0))
    c.add_coupling(MagneticCoupling("K1", "Z1", "Z2", z=0.1 + 0.4j))
    sol = Engine(c, 50).solve()

    # Z1 I1 + Zm I2 = 10 and Zm I1 + (Z2 + R) I2 = 0, with V(n2) = -R I2.
    i1, i2 = np.linalg.solve([[0.2 + 1j, 0.1 + 0.4j], [0.1 + 0.4j, 1.3 + 1j]], [10, 0])
    assert sol.voltages["n2"] == pytest.approx(-i2)
    assert sol.element_currents["Z1"] == pytest.approx(i1)


# R1 carries 5 A, so a CCVS with gain 4 must produce 20 V. R1 starts as G1, so
# this also checks that the engine promotes it to G2 by itself.
def test_ccvs() -> None:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 2.0))
    c.add_element(CCVS("H1", N2, GROUND_NODE, 4.0, "R1"))
    c.add_element(Resistor("R3", N2, GROUND_NODE, 1.0))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n2"] == pytest.approx(20)
    assert sol.element_currents["R1"] == pytest.approx(5)


# Users never have to set groups. The engine must promote exactly the elements
# that need it (here the coupled inductors) and leave the rest in G1.
def test_solve_promotes_coupled_elements_and_ccvs_controls_to_g2() -> None:
    c = transformer(N2, GROUND_NODE)
    Engine(c, 50).solve()
    assert c.elements["L1"].group is ElementGroup.G2
    assert c.elements["L2"].group is ElementGroup.G2
    assert c.elements["R"].group is ElementGroup.G1


# A circuit with only ground gives a 0x0 matrix, which numpy would "solve"
# silently. It must be reported instead.
def test_a_circuit_without_unknowns_is_rejected() -> None:
    with pytest.raises(ValueError, match="no unknowns"):
        Engine(Circuit(), 50).solve()


# A node with nothing attached has an undefined voltage. Until there is proper
# validation, this must at least come out as the clear "singular" error.
def test_a_floating_node_gives_a_singular_matrix() -> None:
    c = divider()
    c.add_node(Node("n4", ""))
    with pytest.raises(ValueError, match="singular"):
        Engine(c, 50).solve()


# Two different voltage sources across the same nodes contradict each other.
# This is the other common cause of a singular matrix.
def test_parallel_voltage_sources_give_a_singular_matrix() -> None:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(VoltageSource("V2", N1, GROUND_NODE, 5))
    with pytest.raises(ValueError, match="singular"):
        Engine(c, 50).solve()


# The frequency is checked when the engine is built, before any work is done.
def test_engine_rejects_an_invalid_frequency() -> None:
    with pytest.raises(ValueError):
        Engine(divider(), 0)
