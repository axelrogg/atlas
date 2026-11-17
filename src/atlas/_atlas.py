from collections.abc import Container

from atlas.core.mna.engine import Engine, Solution, validate_frequency
from atlas.core.model.circuit import Circuit, Wire
from atlas.core.model.elements import (
    CCVS,
    Capacitor,
    CurrentSource,
    Element,
    ElementKind,
    Fixator,
    Impedance,
    Inductor,
    Norator,
    Resistor,
    VoltageSource,
)
from atlas.core.model.magnetic_coupling import MagneticCoupling
from atlas.core.model.node import Node


def next_id(prefix: str, last: int, taken: Container[str]) -> tuple[str, int]:
    n = last
    while True:
        n += 1
        candidate = f"{prefix}{n}"
        if candidate not in taken:
            return candidate, n


def required_value(value: complex | None, kind: ElementKind) -> complex:
    if value is None:
        raise ValueError(f"Element of kind {kind} needs a value")
    return value


def to_real_value(value: complex, kind: ElementKind) -> float:
    if isinstance(value, complex):
        if value.imag != 0:
            raise ValueError(f"Element of kind {kind} needs a real value, got {value}")
        return value.real
    return value


class Simulator:

    def __init__(self, f: float) -> None:
        validate_frequency(f)
        self._f: float = f
        self.circuit: Circuit = Circuit()
        self._id_counts: dict[str, int] = {}

    @property
    def f(self) -> float:
        return self._f

    @property
    def ground(self) -> Node:
        return self.circuit.ground

    def _new_id(self, prefix: str, taken: Container[str]) -> str:
        new_id, n = next_id(prefix, self._id_counts.get(prefix, 0), taken)
        self._id_counts[prefix] = n
        return new_id

    def _require_node_in_circuit(self, n: Node) -> None:
        if self.circuit.nodes.get(n.id) is not n:
            raise ValueError(f"Node {n.id} does not belong to this circuit")

    def _require_element_in_circuit(self, e: Element) -> None:
        if self.circuit.elements.get(e.id) is not e:
            raise ValueError(f"Element {e.id} does not belong to this circuit")

    def add_element(
        self,
        kind: ElementKind,
        value: complex | None = None,
        plus: Node | None = None,
        minus: Node | None = None,
        control: Element | None = None,
        name: str = "",
        voltage: complex | None = None,
        current: complex | None = None
    ) -> Element:
        # Refuse arguments the kind doesn't use instead of silently ignoring them.
        # These checks come before the id is generated, so a rejected call
        # doesn't use up an id.
        if kind in (ElementKind.FIXATOR, ElementKind.NORATOR):
            if value is not None:
                raise ValueError(f"Element of kind {kind} takes no value, got {value}")
        elif value is None:
            raise ValueError(f"Element of kind {kind} needs a value")
        if kind is not ElementKind.FIXATOR and (voltage is not None or current is not None):
            raise ValueError(f"Only a fixator takes a voltage and a current, got kind {kind}")
        if control is not None and kind is not ElementKind.CCVS:
            raise ValueError(f"Only a CCVS takes a control element, got kind {kind}")

        eid = self._new_id(kind.value, self.circuit.elements)

        # A terminal that isn't connected at creation gets its own fresh node.
        # Node ids are plain integers ("1", "2", ...): the empty prefix keeps
        # them apart from element ids, and taken ids (ground is "0") are skipped.
        if plus is None:
            plus = Node(self._new_id("", self.circuit.nodes), "")
        else:
            self._require_node_in_circuit(plus)
        if minus is None:
            minus = Node(self._new_id("", self.circuit.nodes), "")
        else:
            self._require_node_in_circuit(minus)

        e: Element
        match kind:
            case ElementKind.RESISTOR:
                e = Resistor(eid, plus, minus, to_real_value(required_value(value, kind), kind), name=name)
            case ElementKind.INDUCTOR:
                e = Inductor(eid, plus, minus, to_real_value(required_value(value, kind), kind), name=name)
            case ElementKind.CAPACITOR:
                e = Capacitor(eid, plus, minus, to_real_value(required_value(value, kind), kind), name=name)
            case ElementKind.IMPEDANCE:
                e = Impedance(eid, plus, minus, required_value(value, kind), name=name)
            case ElementKind.VOLTAGE_SOURCE:
                e = VoltageSource(eid, plus, minus, required_value(value, kind), name=name)
            case ElementKind.CURRENT_SOURCE:
                e = CurrentSource(eid, plus, minus, required_value(value, kind), name=name)
            case ElementKind.FIXATOR:
                if voltage is None:
                    raise ValueError("A Fixator needs a voltage value")
                if current is None:
                    raise ValueError("A Fixator needs a current value")
                e = Fixator(eid, plus, minus, voltage=voltage, current=current, name=name)
            case ElementKind.NORATOR:
                e = Norator(eid, plus, minus, name=name)
            case ElementKind.CCVS:
                if control is None:
                    raise ValueError("A CCVS needs a control element")
                self._require_element_in_circuit(control)
                e = CCVS(eid, plus, minus, to_real_value(required_value(value, kind), kind), control.id, name=name)

        self.circuit.add_element(e)
        return e

    def connect(self, a: Node, b: Node) -> Wire | None:
        self._require_node_in_circuit(a)
        self._require_node_in_circuit(b)

        # Connecting two points that are already the same point changes nothing.
        if a.id == b.id or self.circuit.are_connected(a.id, b.id):
            return None
        return self.circuit.add_wire(a, b)

    def delete_element(self, e: Element) -> None:
        self._require_element_in_circuit(e)

        # The circuit deletes the element's couplings with it, and refuses if
        # the element is the control of a CCVS.
        self.circuit.delete_element(e.id)
        self._delete_orphan_nodes(e.plus.id, e.minus.id)

    def delete_wire(self, w: Wire) -> None:
        if self.circuit.wires.get(w.id) is not w:
            raise ValueError(f"Wire {w.id} does not belong to this circuit")

        self.circuit.delete_wire(w.id)
        self._delete_orphan_nodes(w.n1, w.n2)

    def _delete_orphan_nodes(self, *node_ids: str) -> None:
        # Atlas creates nodes only as element terminals, so a node with no
        # element or wire left on it would just float and make the matrix
        # singular.
        for nid in node_ids:
            if nid != self.ground.id and nid in self.circuit.nodes and not self.circuit.graph.neighbors(nid):
                self.circuit.delete_node(nid)

    def couple_elements(
        self,
        a: Element,
        b: Element,
        m: float | None = None,
        z: complex | None = None,
    ) -> MagneticCoupling:
        self._require_element_in_circuit(a)
        self._require_element_in_circuit(b)

        kid = self._new_id(MagneticCoupling.id_prefix, self.circuit.couplings)
        k = MagneticCoupling(kid, a.id, b.id, m=m, z=z)
        self.circuit.add_coupling(k)
        return k

    def solve(self) -> Solution:
        return Engine(self.circuit, self.f).solve()

