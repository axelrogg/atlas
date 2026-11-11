import math

import numpy as np
import pytest

from atlas.core.mna.engine import (
    Engine,
    MatrixLayout,
    assemble,
    build_layout,
    check_fixators_and_norators,
    element_current,
    node_voltages,
    required_g2,
    solve_system,
    validate_frequency,
    wire_currents,
)
from atlas.core.mna.labels import KCL, Branch, I, V
from atlas.core.model.circuit import Circuit, Wire
from atlas.core.model.elements import (
    CCVS,
    Capacitor,
    CurrentSource,
    ElementGroup,
    Fixator,
    Impedance,
    Inductor,
    Norator,
    Resistor,
    VoltageSource,
)
from atlas.core.model.magnetic_coupling import MagneticCoupling
from atlas.core.model.node import GROUND_NODE, Node

N1, N2, N3 = Node("n1", ""), Node("n2", ""), Node("n3", "")
# Ground on its own, n1 on its own, and n2 and n3 wired into one net named n2.
BLOCKS = [["0"], ["n1"], ["n2", "n3"]]
# The frequency at which ω = 1 rad/s, so jωL is just jL and results are easy to
# check by hand.
UNIT_OMEGA = 1 / (2 * math.pi)


# Zero (DC), negative, NaN and infinite frequencies aren't valid for this
# steady-state solver and must be refused before anything is built.
@pytest.mark.parametrize("f", [0, -1, math.nan, math.inf])
def test_invalid_frequencies_are_rejected(f: float) -> None:
    with pytest.raises(ValueError):
        validate_frequency(f)


# The check must not reject an ordinary frequency.
def test_valid_frequency_passes() -> None:
    validate_frequency(50)


# Coupling stamps write into Branch(a)/I(b), and a CCVS writes into
# I(control). All of those elements must be found and made G2, or their rows
# and columns won't exist in the matrix.
def test_required_g2_collects_coupled_elements_and_ccvs_controls() -> None:
    elements = {
        "R1": Resistor("R1", N1, GROUND_NODE, 1.0),
        "H1": CCVS("H1", N2, GROUND_NODE, 1.0, "R1"),
    }
    couplings = [MagneticCoupling("K1", "L1", "L2", m=0.1)]
    assert required_g2(elements, couplings) == {"R1", "L1", "L2"}


# A fixator's current is known and has no column, so a CCVS can't refer to it.
# This must be caught with a clear message, not as a missing label later.
def test_required_g2_rejects_a_fixator_as_ccvs_control() -> None:
    elements = {
        "FX1": Fixator("FX1", N1, GROUND_NODE, voltage=1, current=0),
        "H1": CCVS("H1", N2, GROUND_NODE, 1.0, "FX1"),
    }
    with pytest.raises(ValueError, match="fixator"):
        required_g2(elements, [])


# A CCVS added directly to a circuit isn't checked when it is added, so this is
# where a control that doesn't exist has to be caught, with a clear message.
def test_required_g2_rejects_an_unknown_ccvs_control() -> None:
    elements = {"H1": CCVS("H1", N2, GROUND_NODE, 1.0, "R9")}
    with pytest.raises(ValueError):
        required_g2(elements, [])


# The layout is the contract between stamps and matrix positions. Non-ground
# nets come first (KCL rows, V columns), then G2 elements (Branch rows, I
# columns), in pairs. Ground gets nothing, and every node maps to its net.
def test_build_layout_gives_kcl_and_v_per_non_ground_net_then_g2_elements() -> None:
    elements = [
        VoltageSource("V1", N1, GROUND_NODE, 10),
        Resistor("R1", N1, N2, 2.0),
        Resistor("R2", N3, GROUND_NODE, 3.0, group=ElementGroup.G2),
    ]
    layout = build_layout(BLOCKS, elements)
    assert layout.rows == {KCL("n1"): 0, KCL("n2"): 1, Branch("V1"): 2, Branch("R2"): 3}
    assert layout.cols == {V("n1"): 0, V("n2"): 1, I("V1"): 2, I("R2"): 3}
    assert layout.net_of_node == {"0": "0", "n1": "n1", "n2": "n2", "n3": "n2"}


# A fixator gets a branch row but no current column, and a norator a current
# column but no branch row. Together they keep the matrix square.
def test_build_layout_gives_a_fixator_a_row_and_a_norator_a_column() -> None:
    elements = [Fixator("FX1", N1, GROUND_NODE, voltage=10, current=-2), Norator("NR1", N2, GROUND_NODE)]
    layout = build_layout(BLOCKS, elements)
    assert Branch("FX1") in layout.rows and I("FX1") not in layout.cols
    assert I("NR1") in layout.cols and Branch("NR1") not in layout.rows
    assert len(layout.rows) == len(layout.cols)


# Each fixator brings one equation too many and each norator one unknown too
# many. If they don't come in equal numbers the system can't be solved, and the
# user must be told why instead of getting a non-square matrix.
def test_fixators_and_norators_must_balance() -> None:
    fx = Fixator("FX1", N1, GROUND_NODE, voltage=10, current=-2)
    nr = Norator("NR1", N2, GROUND_NODE)
    check_fixators_and_norators([fx, nr])
    check_fixators_and_norators([])
    with pytest.raises(ValueError, match="1 fixator"):
        check_fixators_and_norators([fx])
    with pytest.raises(ValueError, match="1 norator"):
        check_fixators_and_norators([nr])


# Stamps refer to nodes, not nets. A label on n3 must land on n2's row, a label
# on ground must be dropped (None), and Branch/I labels pass straight through.
def test_layout_maps_node_labels_to_their_net_and_drops_ground() -> None:
    layout = build_layout(BLOCKS, [VoltageSource("V1", N1, GROUND_NODE, 10)])
    assert layout.row(KCL("n3")) == layout.row(KCL("n2")) == 1
    assert layout.col(V("n3")) == 1
    assert layout.row(KCL("0")) is None
    assert layout.col(V("0")) is None
    assert layout.row(Branch("V1")) == 2
    assert layout.col(I("V1")) == 2


# MNA needs a square system. A layout with more rows than columns can only come
# from a bug in the engine, so it fails immediately with RuntimeError.
def test_layout_must_be_square() -> None:
    with pytest.raises(RuntimeError):
        MatrixLayout(rows={KCL("n1"): 0}, cols={}, net_of_node={})


# R1 between n1 and n2 and R2 between n3 (part of net n2) and ground. R2's
# ground entries must be dropped and its n3 entries must land on n2.
def test_assemble_adds_stamps_and_drops_ground_entries() -> None:
    r1 = Resistor("R1", N1, N2, 2.0)
    r2 = Resistor("R2", N3, GROUND_NODE, 4.0)
    layout = build_layout(BLOCKS, [r1, r2])
    mat, rhs = assemble(layout, [r1.stamp(50), r2.stamp(50)])
    np.testing.assert_allclose(mat, [[0.5, -0.5], [-0.5, 0.75]])
    np.testing.assert_allclose(rhs, [0, 0])


# Two elements on the same node write into the same matrix position. Entries
# must be added (+=), not overwritten: two 2 Ω in parallel give 1 S.
def test_assemble_accumulates_parallel_elements() -> None:
    blocks = [["0"], ["n1"]]
    ra, rb = Resistor("Ra", N1, GROUND_NODE, 2.0), Resistor("Rb", N1, GROUND_NODE, 2.0)
    mat, _ = assemble(build_layout(blocks, [ra, rb]), [ra.stamp(50), rb.stamp(50)])
    np.testing.assert_allclose(mat, [[1.0]])


# A 2 A source pushing current from ground into n1 must show up as +2 on n1's
# rhs. A sign error here would reverse every current source.
def test_assemble_puts_current_sources_on_the_rhs() -> None:
    blocks = [["0"], ["n1"]]
    j = CurrentSource("I1", GROUND_NODE, N1, 2)
    _, rhs = assemble(build_layout(blocks, [j]), [j.stamp(50)])
    np.testing.assert_allclose(rhs, [2])


# A singular matrix is how most circuit mistakes show up for now. It must come
# out as a ValueError that mentions the likely causes, not numpy's LinAlgError.
def test_solve_system_reports_a_singular_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        solve_system(np.zeros((2, 2), dtype=complex), np.zeros(2, dtype=complex))


# Every node, including ground and nodes merged into another net, must get a
# voltage: ground is 0, and n3 takes the voltage of its net n2.
def test_node_voltages_take_their_net_voltage() -> None:
    layout = build_layout(BLOCKS, [])
    v = node_voltages(layout, np.array([10, 6], dtype=complex), ["0", "n1", "n2", "n3"])
    assert v == {"0": 0, "n1": 10, "n2": 6, "n3": 6}


# Results should all be the same plain Python type, whatever numpy returned and
# whether or not the node is on ground, so users can print, compare and
# serialize them without surprises (np.complex128 or an int 0 for ground).
def test_node_voltages_are_plain_complex_numbers() -> None:
    layout = build_layout(BLOCKS, [])
    v = node_voltages(layout, np.array([10, 6], dtype=complex), ["0", "n1", "n3"])
    assert all(type(x) is complex for x in v.values())


# For a G1 element, evaluating its own KCL(plus) entries must give Ohm's law:
# (10 - 6) / 2 = 2 A.
def test_element_current_of_a_g1_element_is_its_admittance_times_its_voltage() -> None:
    r1 = Resistor("R1", N1, N2, 2.0)
    layout = build_layout(BLOCKS, [r1])
    assert element_current(r1, r1.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)


# The minus terminal is on ground, which has no column: that term must count as
# 0 V instead of failing. R2 sits on n3, which is part of net n2: 6 / 3 = 2 A.
def test_element_current_with_a_terminal_on_ground() -> None:
    r2 = Resistor("R2", N3, GROUND_NODE, 3.0)
    layout = build_layout(BLOCKS, [r2])
    assert element_current(r2, r2.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)


# A G2 element's KCL entry sits in its own I column, so its current must be read
# straight from the solution (here -2 A).
def test_element_current_of_a_g2_element_is_read_from_its_column() -> None:
    v1 = VoltageSource("V1", N1, GROUND_NODE, 10)
    layout = build_layout(BLOCKS, [v1])
    assert element_current(v1, v1.stamp(50), layout, np.array([10, 6, -2])) == pytest.approx(-2)


# A G1 current source has no matrix entries, only rhs entries. Its current must
# come from undoing the sign flip of the rhs and equal its value. Its plus
# terminal is ground, which also checks that ground rows are still read from the
# stamp.
def test_element_current_of_a_g1_current_source_is_its_value() -> None:
    j = CurrentSource("I1", GROUND_NODE, N1, 2)
    layout = build_layout(BLOCKS, [j])
    assert element_current(j, j.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)


# A wire carries on whatever current arrives at its end. A source feeds 2 A
# into n1, R1 delivers them into n2, so the wire n2 -> n3 carries 2 A on to R2.
# Written the other way round (n3 -> n2), the same current comes out negative.
def test_a_single_wire_carries_the_current_arriving_at_it() -> None:
    elements = [
        CurrentSource("I1", GROUND_NODE, N1, 2),
        Resistor("R1", N1, N2, 1.0),
        Resistor("R2", N3, GROUND_NODE, 1.0),
    ]
    currents = {"I1": 2, "R1": 2, "R2": 2}
    assert wire_currents([Wire("w", "n2", "n3")], elements, currents) == {"w": 2}
    assert wire_currents([Wire("w", "n3", "n2")], elements, currents) == {"w": -2}


A, B, C = Node("a", ""), Node("b", ""), Node("c", "")


# A chain a - b - c with current taken out along the way: 3 A come in at a,
# 1 A leaves at b, 2 A leave at c. The first wire carries 3 A, the second only
# what is left (2 A). Peeling must start at the far leaf and work inwards.
def test_a_chain_of_wires_passes_on_what_is_left() -> None:
    elements = [
        CurrentSource("I1", GROUND_NODE, A, 3),
        Resistor("Rb", B, GROUND_NODE, 1.0),
        Resistor("Rc", C, GROUND_NODE, 1.0),
    ]
    currents = {"I1": 3, "Rb": 1, "Rc": 2}
    wires = [Wire("ab", "a", "b"), Wire("bc", "b", "c")]
    assert wire_currents(wires, elements, currents) == pytest.approx({"ab": 3, "bc": 2})


# A star: 6 A come into the center and split into three branches. Each wire
# must carry exactly what its own branch draws, which only works if every leaf
# is peeled independently.
def test_a_star_of_wires_splits_the_current() -> None:
    s, x, y, z = Node("s", ""), Node("x", ""), Node("y", ""), Node("z", "")
    elements = [
        CurrentSource("I1", GROUND_NODE, s, 6),
        Resistor("Rx", x, GROUND_NODE, 1.0),
        Resistor("Ry", y, GROUND_NODE, 1.0),
        Resistor("Rz", z, GROUND_NODE, 1.0),
    ]
    currents = {"I1": 6, "Rx": 1, "Ry": 2, "Rz": 3}
    wires = [Wire("sx", "s", "x"), Wire("sy", "s", "y"), Wire("sz", "s", "z")]
    assert wire_currents(wires, elements, currents) == pytest.approx({"sx": 1, "sy": 2, "sz": 3})


# Ground has no KCL row in the matrix, but KCL still holds there. A wire to
# ground must get its current like any other: the 2 A pushed into a return to
# ground through the wire.
def test_a_wire_to_ground() -> None:
    elements = [CurrentSource("I1", GROUND_NODE, A, 2)]
    assert wire_currents([Wire("w", "a", "0")], elements, {"I1": 2}) == pytest.approx({"w": 2})


# No wires, nothing to report. KCL still has to hold at every node, so a
# consistent set of element currents must pass the check.
def test_no_wires() -> None:
    elements = [CurrentSource("I1", GROUND_NODE, A, 2), Resistor("R", A, GROUND_NODE, 1.0)]
    assert wire_currents([], elements, {"I1": 2, "R": 2}) == {}


# The sanity check: if the element currents broke KCL (which can only happen
# through a bug), the leftover current must be reported instead of being
# silently pushed into some wire.
def test_wire_currents_reject_currents_that_break_kcl() -> None:
    elements = [
        CurrentSource("I1", GROUND_NODE, N1, 2),
        Resistor("R1", N1, N2, 1.0),
        Resistor("R2", N3, GROUND_NODE, 1.0),
    ]
    with pytest.raises(RuntimeError, match="KCL"):
        wire_currents([Wire("w", "n2", "n3")], elements, {"I1": 2, "R1": 2, "R2": 5})


# Peeling can't get through a loop. The circuit never allows one, but if a loop
# ever got through, it must be reported instead of leaving wires without a
# current.
def test_wire_currents_reject_a_loop() -> None:
    elements = [Resistor("R", A, B, 1.0)]
    wires = [Wire("w1", "a", "b"), Wire("w2", "b", "a")]
    with pytest.raises(RuntimeError, match="loop"):
        wire_currents(wires, elements, {"R": 0})


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


# The wire between R1 and R2 carries the same 2 A as both resistors, from n2 to
# n3. This is the solver filling in wire currents after the matrix solve.
def test_solve_fills_in_wire_currents() -> None:
    c = divider()
    (w,) = c.wires.values()
    sol = Engine(c, 50).solve()
    assert (w.n1, w.n2) == ("n2", "n3")
    assert sol.wire_currents == pytest.approx({w.id: 2})


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


# 10 V and 2 A known at n1 (the 2 A flow into n1, so the fixator's own
# plus -> minus current is -2), R1 = 2 ohm to n2, and nothing known at n2. The
# solver must find V(n2) = 10 - 2 * 2 = 6 V and 2 A leaving through the norator.
def test_fixator_and_norator_solve_an_input_output_case() -> None:
    c = Circuit()
    c.add_element(Fixator("FX1", N1, GROUND_NODE, voltage=10, current=-2))
    c.add_element(Resistor("R1", N1, N2, 2.0))
    c.add_element(Norator("NR1", N2, GROUND_NODE))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n1"] == pytest.approx(10)
    assert sol.voltages["n2"] == pytest.approx(6)
    assert sol.element_currents == pytest.approx({"FX1": -2, "R1": 2, "NR1": 2})


# With a load at the output, the current known at the input splits between the
# load and the norator: 6 V across 6 ohm takes 1 A, so 1 A is left for the
# norator.
def test_fixator_and_norator_with_a_load_at_the_output() -> None:
    c = Circuit()
    c.add_element(Fixator("FX1", N1, GROUND_NODE, voltage=10, current=-2))
    c.add_element(Resistor("R1", N1, N2, 2.0))
    c.add_element(Resistor("R2", N2, GROUND_NODE, 6.0))
    c.add_element(Norator("NR1", N2, GROUND_NODE))
    sol = Engine(c, 50).solve()
    assert sol.voltages["n2"] == pytest.approx(6)
    assert sol.element_currents["R2"] == pytest.approx(1)
    assert sol.element_currents["NR1"] == pytest.approx(1)


# A fixator without a norator can't be solved; the engine must say so before it
# builds the matrix.
def test_engine_rejects_an_unbalanced_fixator() -> None:
    c = Circuit()
    c.add_element(Fixator("FX1", N1, GROUND_NODE, voltage=10, current=-2))
    c.add_element(Resistor("R1", N1, GROUND_NODE, 2.0))
    with pytest.raises(ValueError, match="norator"):
        Engine(c, 50).solve()


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
