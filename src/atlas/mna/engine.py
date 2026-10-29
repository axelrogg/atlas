from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from atlas.mna.labels import KCL, V, I, Branch, Col, Row
from atlas.mna.stamp import Stamp
from atlas.model.circuit import Circuit, Wire
from atlas.model.elements import CCVS, Element, ElementGroup
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.model.node import GROUND_NODE
from atlas.utils import is_number_valid


@dataclass(frozen=True)
class MatrixLayout:
    rows: dict[Row, int]
    cols: dict[Col, int]
    net_of_node: dict[str, str]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.cols):
            raise RuntimeError(f"Index has {len(self.rows)} rows but {len(self.cols)} columns; the system must be square.")

    def row(self, label: Row) -> int | None:
        if isinstance(label, KCL):
            net = self.net_of_node[label.node]
            return None if GROUND_NODE.id == net else self.rows[KCL(net)]
        return self.rows[label]

    def col(self, label: Col) -> int | None:
        if isinstance(label, V):
            net = self.net_of_node[label.node]
            return None if GROUND_NODE.id == net else self.cols[V(net)]
        return self.cols[label]


@dataclass(frozen=True)
class Solution:
    voltages: dict[str, complex]
    element_currents: dict[str, complex]
    wire_currents: dict[str, complex]


def validate_frequency(f: float) -> None:
    if not is_number_valid(f) or f <= 0:
        raise ValueError(f"Frequency must be positive, got {f} Hz.")


def required_g2(elements: Mapping[str, Element], couplings: Iterable[MagneticCoupling]) -> set[str]:
    needs_g2: set[str] = set()

    for k in couplings:
        needs_g2.update((k.a_id, k.b_id))

    for e in elements.values():
        if isinstance(e, CCVS):
            if e.control_id not in elements:
                raise ValueError(f"CCVS {e.id} refers to unknown control element: {e.control_id}")
            needs_g2.add(e.control_id)

    return needs_g2


def build_layout(net_blocks: Iterable[list[str]], elements: Iterable[Element]) -> MatrixLayout:
    net_of_node: dict[str, str] = {}
    rows: dict[Row, int] = {}
    cols: dict[Col, int] = {}

    # First we assign a net to each node, and then we assign rows and
    # columns to the KCL equations and voltage variables.
    for block in net_blocks:
        net = block[0]
        for node_id in block:
            net_of_node[node_id] = net
        if GROUND_NODE.id not in block:
            rows[KCL(net)] = len(rows)
            cols[V(net)] = len(cols)

    # Now we assign rows and columns to the branch currents of the
    # elements only if they are in group G2 (i.e., they have a branch current variable).
    for e in elements:
        if e.group is ElementGroup.G2:
            rows[Branch(e.id)] = len(rows)
            cols[I(e.id)] = len(cols)

    return MatrixLayout(rows=rows, cols=cols, net_of_node=net_of_node)


def assemble(layout: MatrixLayout, stamps: Iterable[Stamp]) -> tuple[np.ndarray, np.ndarray]:
    n = len(layout.rows)
    mat = np.zeros((n, n), dtype=complex)
    rhs = np.zeros(n, dtype=complex)

    for s in stamps:
        for entry in s.entries:
            row, col = layout.row(entry.row), layout.col(entry.col)
            if row is not None and col is not None:  # if the given stamp has nothing to do with ground node
                mat[row, col] += entry.value

        for entry in s.rhs:
            row = layout.row(entry.row)
            if row is not None:
                rhs[row] += entry.value

    return mat, rhs


def solve_system(mat: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        raise ValueError("System matrix is singular: check for floating nodes, loops of voltage sources, or elements only connected through capacitors at low frequency") from None


def node_voltages(layout: MatrixLayout, solution: np.ndarray, node_ids: Iterable[str]) -> dict[str, complex]:
    voltages: dict[str, complex] = {}

    for nid in node_ids:
        i = layout.col(V(nid))
        # Plain Python complex for every node, ground included, like the currents.
        voltages[nid] = 0j if i is None else complex(solution[i])

    return voltages


def element_current(e: Element, stamp: Stamp, layout: MatrixLayout, solution: np.ndarray) -> complex:
    # The current through an element (plus -> minus) is what its own stamp
    # draws out of its plus node: the KCL(plus) entries evaluated at the
    # solution, minus the KCL(plus) rhs entries.
    plus = KCL(e.plus.id)
    current: complex = 0

    # Unknown-dependent part (the left-hand side of the KCL(plus) row).
    for entry in stamp.entries:
        # Only the plus row: the minus row holds the same current with
        # the opposite sign, and Branch rows are voltage equations.
        if entry.row != plus:
            continue
        # Each entry is weighted by its own column's value: V(plus) and
        # V(minus) for G1 (giving y * (V(p) - V(m))), or the element's
        # own I(e) for G2 (giving I(e)). A KCL row never refers to
        # another element's I column.
        j = layout.col(entry.col)
        # A V column on the ground net has no matrix column: V = 0, so
        # the term adds nothing.
        if j is not None:
            current += entry.value * solution[j]

    # Known part (the right-hand side of the KCL(plus) row). Known
    # currents were moved to the rhs with a flipped sign, so they are
    # subtracted to flip it back. Only a G1 current source writes here;
    # every other rhs entry lives in a Branch row and is skipped.
    for entry in stamp.rhs:
        if entry.row == plus:
            current -= entry.value

    # np.complex128 -> plain Python complex.
    return complex(current)


def wire_currents(
    wires: Iterable[Wire],
    elements: Iterable[Element],
    element_currents: Mapping[str, complex],
) -> dict[str, complex]:
    wires = list(wires)

    # residual[n] is the current that still has to leave node n through its
    # wires. It starts as what the elements push into n: an element's current
    # arrives at its minus node and leaves its plus node.
    residual: dict[str, complex] = {}
    for e in elements:
        i = element_currents[e.id]
        residual[e.plus.id] = residual.get(e.plus.id, 0) - i
        residual[e.minus.id] = residual.get(e.minus.id, 0) + i

    # The wires at each node whose current isn't known yet.
    pending: dict[str, list[Wire]] = {}
    for w in wires:
        pending.setdefault(w.n1, []).append(w)
        pending.setdefault(w.n2, []).append(w)

    # Wire loops are rejected by the circuit, so the wires of every net form a
    # tree. A leaf (a node with a single pending wire) has no other way out, so
    # that wire must carry the leaf's whole residual. Assigning it moves the
    # current to the other end, which may become a leaf in turn.
    currents: dict[str, complex] = {}
    leaves = deque(n for n, ws in pending.items() if len(ws) == 1)
    while leaves:
        n = leaves.popleft()
        # n joined the queue with one pending wire, but a wire has two ends and
        # either can assign it. If the node at the other end came out of the
        # queue first, it already assigned this wire and removed it from n's
        # list too, so n has nothing left to do. Example: a -ab- b -bc- c.
        # Popping a assigns ab, and b joins the queue with [bc]. Popping c
        # assigns bc and empties b's list. When b comes out, it is skipped.
        if not len(pending[n]):
            continue
        w = pending[n][0]
        r = residual.get(n, 0)
        # Positive current flows n1 -> n2.
        i = r if n == w.n1 else -r
        currents[w.id] = complex(i)
        residual[w.n1] = residual.get(w.n1, 0) - i
        residual[w.n2] = residual.get(w.n2, 0) + i
        for end in (w.n1, w.n2):
            pending[end].remove(w)
            if len(pending[end]) == 1:
                leaves.append(end)

    # Sanity check. Every wire must have been reached (no loops), and KCL must
    # now hold at every node: nothing left to carry away. A failure here is a
    # bug in the engine, not a mistake in the circuit.
    if len(currents) != len(wires):
        raise RuntimeError(f"Assigned {len(currents)} of {len(wires)} wire currents; the wires contain a loop.")
    scale = max((abs(i) for i in element_currents.values()), default=0.0)
    tolerance = 1e-9 * max(1.0, scale)
    for n, r in residual.items():
        if abs(r) > tolerance:
            raise RuntimeError(f"KCL does not hold at node {n} after assigning wire currents: {r} A left over.")

    return currents


class Engine:

    def __init__(self, circuit: Circuit, f: float) -> None:
        validate_frequency(f)
        self.circuit: Circuit = circuit
        self.f: float = f

    def _resolve_element_groups(self) -> None:
        needs_g2 = required_g2(self.circuit.elements, self.circuit.couplings.values())
        for e in self.circuit.elements.values():
            if e.id in needs_g2 and e.group is ElementGroup.G1:
                e.regroup(ElementGroup.G2)

    def solve(self) -> Solution:
        # TODO(validation): floating subcircuits. Report every node with no path
        # to ground (graph BFS from ground) instead of failing later with a
        # singular matrix.
        self._resolve_element_groups()
        layout = build_layout(self.circuit.netblocks(), self.circuit.elements.values())
        if len(layout.rows) == 0:
            raise ValueError("Circuit has no unknowns to solve for.")

        # TODO(validation): shorted elements. An element whose plus and minus
        # end up in the same net (net_of_node) after wires merge them: a G1
        # stamp cancels to zero silently, a voltage source makes the matrix
        # singular.
        # TODO(validation): voltage source loops. A loop made only of voltage
        # sources (and CCVSs) over-constrains the system and makes the matrix
        # singular; report the elements in the loop.

        element_stamps = [(e, e.stamp(self.f)) for e in self.circuit.elements.values()]
        coupling_stamps = [k.stamp(self.f) for k in self.circuit.couplings.values()]

        mat, rhs = assemble(layout, [s for _, s in element_stamps] + coupling_stamps)
        solution = solve_system(mat, rhs)

        currents = {e.id: element_current(e, s, layout, solution) for e, s in element_stamps}

        return Solution(
            voltages=node_voltages(layout, solution, self.circuit.nodes),
            element_currents=currents,
            wire_currents=wire_currents(self.circuit.wires.values(), self.circuit.elements.values(), currents),
        )
