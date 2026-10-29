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
UNIT_OMEGA = 1 / (2 * math.pi)


def divider(r1_group: ElementGroup = ElementGroup.G1) -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Resistor("R1", N1, N2, 2.0, group=r1_group))
    c.add_wire(N2, N3)
    c.add_element(Resistor("R2", N3, GROUND_NODE, 3.0))
    return c


@pytest.mark.parametrize("group", [ElementGroup.G1, ElementGroup.G2])
def test_voltage_divider(group: ElementGroup) -> None:
    sol = Engine(divider(group), 50).solve()
    assert sol.voltages["n1"] == pytest.approx(10)
    assert sol.voltages["n2"] == sol.voltages["n3"] == pytest.approx(6)
    assert sol.voltages["0"] == 0
    assert sol.element_currents == pytest.approx({"V1": -2, "R1": 2, "R2": 2})


def test_current_source_into_a_resistor() -> None:
    c = Circuit()
    c.add_element(CurrentSource("I1", GROUND_NODE, N1, 2))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 5.0))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n1"] == pytest.approx(10)
    assert sol.element_currents == pytest.approx({"I1": 2, "R1": 2})


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


def transformer(l2_plus: Node, l2_minus: Node) -> Circuit:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Inductor("L1", N1, GROUND_NODE, 1.0))
    c.add_element(Inductor("L2", l2_plus, l2_minus, 1.0))
    c.add_element(Resistor("R", N2, GROUND_NODE, 1.0))
    c.add_coupling(MagneticCoupling("K1", "L1", "L2", m=0.5))
    return c


def test_coupled_inductors() -> None:
    sol = Engine(transformer(N2, GROUND_NODE), UNIT_OMEGA).solve()
    assert sol.voltages["n2"] == pytest.approx(3.2 - 2.4j)


def test_flipping_a_coupled_inductor_flips_the_mutual_voltage() -> None:
    sol = Engine(transformer(GROUND_NODE, N2), UNIT_OMEGA).solve()
    assert sol.voltages["n2"] == pytest.approx(-3.2 + 2.4j)


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


def test_ccvs() -> None:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 2.0))
    c.add_element(CCVS("H1", N2, GROUND_NODE, 4.0, "R1"))
    c.add_element(Resistor("R3", N2, GROUND_NODE, 1.0))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n2"] == pytest.approx(20)
    assert sol.element_currents["R1"] == pytest.approx(5)


def test_solve_promotes_coupled_elements_and_ccvs_controls_to_g2() -> None:
    c = transformer(N2, GROUND_NODE)
    Engine(c, 50).solve()
    assert c.elements["L1"].group is ElementGroup.G2
    assert c.elements["L2"].group is ElementGroup.G2
    assert c.elements["R"].group is ElementGroup.G1


def test_a_circuit_without_unknowns_is_rejected() -> None:
    with pytest.raises(ValueError, match="no unknowns"):
        Engine(Circuit(), 50).solve()


def test_a_floating_node_gives_a_singular_matrix() -> None:
    c = divider()
    c.add_node(Node("n4", ""))
    with pytest.raises(ValueError, match="singular"):
        Engine(c, 50).solve()


def test_parallel_voltage_sources_give_a_singular_matrix() -> None:
    c = Circuit()
    c.add_element(VoltageSource("V1", N1, GROUND_NODE, 10))
    c.add_element(VoltageSource("V2", N1, GROUND_NODE, 5))
    with pytest.raises(ValueError, match="singular"):
        Engine(c, 50).solve()


def test_engine_rejects_an_invalid_frequency() -> None:
    with pytest.raises(ValueError):
        Engine(divider(), 0)
