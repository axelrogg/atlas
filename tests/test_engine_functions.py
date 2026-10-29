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
BLOCKS = [["0"], ["n1"], ["n2", "n3"]]


@pytest.mark.parametrize("f", [0, -1, math.nan, math.inf])
def test_invalid_frequencies_are_rejected(f: float) -> None:
    with pytest.raises(ValueError):
        validate_frequency(f)


def test_valid_frequency_passes() -> None:
    validate_frequency(50)


def test_required_g2_collects_coupled_elements_and_ccvs_controls() -> None:
    elements = {
        "R1": Resistor("R1", N1, GROUND_NODE, 1.0),
        "H1": CCVS("H1", N2, GROUND_NODE, 1.0, "R1"),
    }
    couplings = [MagneticCoupling("K1", "L1", "L2", m=0.1)]
    assert required_g2(elements, couplings) == {"R1", "L1", "L2"}


def test_required_g2_rejects_an_unknown_ccvs_control() -> None:
    elements = {"H1": CCVS("H1", N2, GROUND_NODE, 1.0, "R9")}
    with pytest.raises(ValueError):
        required_g2(elements, [])


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


def test_layout_maps_node_labels_to_their_net_and_drops_ground() -> None:
    layout = build_layout(BLOCKS, [VoltageSource("V1", N1, GROUND_NODE, 10)])
    assert layout.row(KCL("n3")) == layout.row(KCL("n2")) == 1
    assert layout.col(V("n3")) == 1
    assert layout.row(KCL("0")) is None
    assert layout.col(V("0")) is None
    assert layout.row(Branch("V1")) == 2
    assert layout.col(I("V1")) == 2


def test_layout_must_be_square() -> None:
    with pytest.raises(RuntimeError):
        MatrixLayout(rows={KCL("n1"): 0}, cols={}, net_of_node={})


def test_assemble_adds_stamps_and_drops_ground_entries() -> None:
    r1 = Resistor("R1", N1, N2, 2.0)
    r2 = Resistor("R2", N3, GROUND_NODE, 4.0)
    layout = build_layout(BLOCKS, [r1, r2])
    mat, rhs = assemble(layout, [r1.stamp(50), r2.stamp(50)])
    np.testing.assert_allclose(mat, [[0.5, -0.5], [-0.5, 0.75]])
    np.testing.assert_allclose(rhs, [0, 0])


def test_assemble_accumulates_parallel_elements() -> None:
    blocks = [["0"], ["n1"]]
    ra, rb = Resistor("Ra", N1, GROUND_NODE, 2.0), Resistor("Rb", N1, GROUND_NODE, 2.0)
    mat, _ = assemble(build_layout(blocks, [ra, rb]), [ra.stamp(50), rb.stamp(50)])
    np.testing.assert_allclose(mat, [[1.0]])


def test_assemble_puts_current_sources_on_the_rhs() -> None:
    blocks = [["0"], ["n1"]]
    j = CurrentSource("I1", GROUND_NODE, N1, 2)
    _, rhs = assemble(build_layout(blocks, [j]), [j.stamp(50)])
    np.testing.assert_allclose(rhs, [2])


def test_solve_system_reports_a_singular_matrix() -> None:
    with pytest.raises(ValueError, match="singular"):
        solve_system(np.zeros((2, 2), dtype=complex), np.zeros(2, dtype=complex))


def test_node_voltages_take_their_net_voltage() -> None:
    layout = build_layout(BLOCKS, [])
    v = node_voltages(layout, np.array([10, 6], dtype=complex), ["0", "n1", "n2", "n3"])
    assert v == {"0": 0, "n1": 10, "n2": 6, "n3": 6}


def test_element_current_of_a_g1_element_is_its_admittance_times_its_voltage() -> None:
    r1 = Resistor("R1", N1, N2, 2.0)
    layout = build_layout(BLOCKS, [r1])
    assert element_current(r1, r1.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)


def test_element_current_with_a_terminal_on_ground() -> None:
    r2 = Resistor("R2", N3, GROUND_NODE, 3.0)
    layout = build_layout(BLOCKS, [r2])
    assert element_current(r2, r2.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)


def test_element_current_of_a_g2_element_is_read_from_its_column() -> None:
    v1 = VoltageSource("V1", N1, GROUND_NODE, 10)
    layout = build_layout(BLOCKS, [v1])
    assert element_current(v1, v1.stamp(50), layout, np.array([10, 6, -2])) == pytest.approx(-2)


def test_element_current_of_a_g1_current_source_is_its_value() -> None:
    j = CurrentSource("I1", GROUND_NODE, N1, 2)
    layout = build_layout(BLOCKS, [j])
    assert element_current(j, j.stamp(50), layout, np.array([10, 6])) == pytest.approx(2)
