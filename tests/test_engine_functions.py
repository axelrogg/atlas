import math

import numpy as np
import pytest

from atlas.mna.engine import (
    MatrixLayout,
    assemble,
    build_layout,
    element_current,
    node_voltages,
    required_g2,
    solve_system,
    validate_frequency,
)
from atlas.mna.labels import KCL, Branch, I, V
from atlas.model.elements import CCVS, CurrentSource, ElementGroup, Resistor, VoltageSource
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.model.node import GROUND_NODE, Node

N1, N2, N3 = Node("n1", ""), Node("n2", ""), Node("n3", "")
# Ground on its own, n1 on its own, and n2 and n3 wired into one net named n2.
BLOCKS = [["0"], ["n1"], ["n2", "n3"]]


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
