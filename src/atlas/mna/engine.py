import numpy as np
from dataclasses import dataclass

from atlas.mna.labels import KCL, V, I, Branch, Col, Row
from atlas.model.circuit import Circuit
from atlas.model.node import GROUND_NODE
from atlas.model.elements import ElementGroup, CCVS
from atlas.utils import is_number_valid
from atlas.mna.stamp import Stamp


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


class Engine:

    def __init__(self, circuit: Circuit, f: float) -> None:
        self.circuit = circuit
        self._validate_f(f)
        self.f = f

    def _validate_f(self, f: float) -> None:
        if not is_number_valid(f) or f <= 0:
            raise ValueError(f"Frequency must be positive, got {f} Hz.")

    def _layout(self) -> MatrixLayout:
        net_of_node: dict[str, str] = {}
        rows: dict[Row, int] = {}
        cols: dict[Col, int] = {}

        # First we assign a net to each node, and then we assign rows and
        # columns to the KCL equations and voltage variables.
        for block in self.circuit.netblocks():
            net = block[0]
            for node_id in block:
                net_of_node[node_id] = net
            if GROUND_NODE.id not in block:
                rows[KCL(net)] = len(rows)
                cols[V(net)] = len(cols)

        # Now we assign rows and columns to the branch currents of the
        # elements only if they are in group G2 (i.e., they have a branch current variable).
        for e in self.circuit.elements.values():
            if e.group is ElementGroup.G2:
                rows[Branch(e.id)] = len(rows)
                cols[I(e.id)] = len(cols)

        return MatrixLayout(rows=rows, cols=cols, net_of_node=net_of_node)

    def _resolve_element_groups(self) -> None:
        needs_g2: set[str] = set()

        for k in self.circuit.couplings.values():
            needs_g2.update((k.a_id, k.b_id))

        for e in self.circuit.elements.values():
            if isinstance(e, CCVS):
                if e.control_id not in self.circuit.elements:
                    raise ValueError(f"CCVS {e.id} refers to unknown control element: {e.control_id}")
                needs_g2.add(e.control_id)

        for e in self.circuit.elements.values():
            if e.id in needs_g2 and e.group is ElementGroup.G1:
                e.regroup(ElementGroup.G2)

    def _stamps(self) -> list[Stamp]:
        stamps = [e.stamp(self.f) for e in self.circuit.elements.values()]
        stamps += [k.stamp(self.f) for k in self.circuit.couplings.values()]
        return stamps

    def _assemble(self, layout: MatrixLayout, stamps: list[Stamp]) -> tuple[np.ndarray, np.ndarray]:
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

    def solve(self) -> Solution:
        # TODO(validation): floating subcircuits. Report every node with no path
        # to ground (graph BFS from ground) instead of failing later with a
        # singular matrix.
        self._resolve_element_groups()
        mat_layout = self._layout()
        if len(mat_layout.rows) == 0:
            raise ValueError("Circuit has no unknowns to solve for.")

        # TODO(validation): shorted elements. An element whose plus and minus
        # end up in the same net (net_of_node) after wires merge them: a G1
        # stamp cancels to zero silently, a voltage source makes the matrix
        # singular.
        # TODO(validation): voltage source loops. A loop made only of voltage
        # sources (and CCVSs) over-constrains the system and makes the matrix
        # singular; report the elements in the loop.

        stamps = self._stamps()

        mat, rhs = self._assemble(mat_layout, stamps)
        try:
            solution = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            raise ValueError("System matrix is singular: check for floating nodes, loops of voltage sources, or elements only connected through capacitors at low frequency") from None

        return self._complete_solution(mat_layout, solution, stamps)

    def _complete_solution(self, layout: MatrixLayout, solution: np.ndarray, stamps: list[Stamp]) -> Solution:
        voltages: dict[str, complex] = {}
        currents: dict[str, complex] = {}

        for nid in self.circuit.nodes:
            i = layout.col(V(nid))
            voltages[nid] = 0 if i is None else solution[i]

        # The current through an element (plus -> minus) is what its own stamp
        # draws out of its plus node: the KCL(plus) entries evaluated at the
        # solution, minus the KCL(plus) rhs entries. _stamps() lists element
        # stamps first, in the same order as circuit.elements.
        for e, s in zip(self.circuit.elements.values(), stamps):
            plus = KCL(e.plus.id)
            current: complex = 0

            # Unknown-dependent part (the left-hand side of the KCL(plus) row).
            for entry in s.entries:
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
            for entry in s.rhs:
                if entry.row == plus:
                    current -= entry.value

            # np.complex128 -> plain Python complex.
            currents[e.id] = complex(current)

        return Solution(voltages=voltages, element_currents=currents)

