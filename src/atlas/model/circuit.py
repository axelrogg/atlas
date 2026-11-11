from dataclasses import dataclass
from enum import Enum, auto

from atlas.model.elements import CCVS, Element
from atlas.model.magnetic_coupling import MagneticCoupling
from atlas.structs.diset import DisjointSet
from atlas.structs.graph import DiGraph, Edge
from atlas.model.node import GROUND_NODE, Node

# For when i decide to rebuild the net lazily
class CircuitNetsState(Enum):
    OUT_OF_SYNC = auto()
    IN_SYNC = auto()


@dataclass(frozen=True)
class Wire:
    id: str
    n1: str
    n2: str


class Circuit:

    def __init__(self) -> None:
        self.graph: DiGraph = DiGraph()
        self.nets: DisjointSet = DisjointSet()
        self.elements: dict[str, Element] = {}
        #self.wires: list[tuple[str, str]] = []
        self.wires: dict[str, Wire] = {}
        self._wire_to_edge: dict[str, Edge] = {}
        self._element_to_edge: dict[str, Edge] = {}
        self.nodes: dict[str, Node] = {}
        self.couplings: dict[str, MagneticCoupling] = {}
        self._couplings_of: dict[str, set[str]] = {}

        self.edge_count: int = 0
        self.wire_count: int = 0

        self.netstate: CircuitNetsState = CircuitNetsState.IN_SYNC
        self._register_ground_node()

    @property
    def ground(self) -> Node:
        return GROUND_NODE

    @property
    def is_connected(self) -> bool:
        return self.graph.is_connected()

    def _register_ground_node(self) -> None:
        self.nodes[GROUND_NODE.id] = GROUND_NODE
        self.graph.add_vertex(GROUND_NODE.id)
        self.nets.add(GROUND_NODE.id)

    def _rebuild_nets(self):
        new_nets = DisjointSet()
        for n in self.nodes.values():
            new_nets.add(n.id)

        for w in self.wires.values():
            new_nets.union(new_nets.add(w.n1), new_nets.add(w.n2))

        self.nets = new_nets
        self.netstate = CircuitNetsState.IN_SYNC

    def _create_edge_key(self) -> str:
        self.edge_count += 1
        return str(self.edge_count)

    def _create_wire_key(self) -> str:
        self.wire_count += 1
        return str(self.wire_count)

    def _element_exists(self, e: Element) -> bool:
        return self.elements.get(e.id) is not None

    def are_connected(self, xid: str, yid: str) -> bool:
        if self.netstate == CircuitNetsState.OUT_OF_SYNC:
            self._rebuild_nets()

        xm, ym = self.nets.find_member(xid), self.nets.find_member(yid)
        return xm is not None and ym is not None and self.nets.is_connected(xm, ym)

    def add_wire(self, x: Node, y: Node) -> Wire:
        if x.id == y.id:
            raise ValueError(f"Self-loops are not allowed: wire would connect node {x.id} to itself")

        if self.are_connected(x.id, y.id):
            raise ValueError(f"Wire loops are not allowed: nodes {x.id} and {y.id} are already connected by wires")

        for n in (x, y):
            if n.id not in self.nodes:
                self.nodes[n.id] = n

        wire_id = self._create_wire_key()
        edge = Edge(key=f"{self._create_edge_key()}", tail=x.id, head=y.id)
        self.graph.add_edge(edge)

        self.nets.union(self.nets.add(x.id), self.nets.add(y.id))

        wire = Wire(id=wire_id, n1=x.id, n2=y.id)
        self.wires[wire_id] = wire
        self._wire_to_edge[wire_id] = edge
        return wire

    def add_element(self, e: Element) -> None:
        if self._element_exists(e):
            raise ValueError(f"Duplicate element with id: {e.id}")

        if e.plus.id == e.minus.id:
            raise ValueError(f"Self-loops are not allowed: element {e.id} has the same plus and minus node {e.plus.id}")

        for n in (e.plus, e.minus):
            if n.id not in self.nodes:
                self.nodes[n.id] = n
           
        edge = Edge(key=f"{self._create_edge_key()}", tail=e.plus.id, head=e.minus.id)
        self.graph.add_edge(edge)

        self.nets.add(e.plus.id)
        self.nets.add(e.minus.id)
        self.elements[e.id] = e
        self._element_to_edge[e.id] = edge


    def add_node(self, n: Node) -> None:
        if n.id == GROUND_NODE.id:
            raise ValueError(f"Node id {n.id!r} is reserved for ground, which every circuit already has (use circuit.ground)")
        if self.nodes.get(n.id) is not None:
            raise ValueError(f"Duplicate node with id: {n.id}")

        self.nodes[n.id] = n
        self.graph.add_vertex(n.id)
        self.nets.add(n.id)

    def delete_wire(self, wk: str) -> None:
        if self.wires.get(wk) is None:
            raise ValueError(f"No wire with id: {wk}")

        # remove from diset and edge
        self.graph.delete_edge(self._wire_to_edge[wk].key)
        del self.wires[wk]
        del self._wire_to_edge[wk]

        self.netstate = CircuitNetsState.OUT_OF_SYNC

    def delete_element(self, ek: str) -> None:
        if self.elements.get(ek) is None:
            raise ValueError(f"No element with id: {ek}")

        self._require_not_controlling({ek})

        self._delete_couplings_of(ek)
        self.graph.delete_edge(self._element_to_edge[ek].key)
        del self.elements[ek]
        del self._element_to_edge[ek]

        self.netstate = CircuitNetsState.OUT_OF_SYNC

    def delete_node(self, nk: str) -> None:
        if GROUND_NODE.id == nk:
            raise ValueError(f"Node id {nk} is reserved for ground and cannot be removed")

        if self.nodes.get(nk) is None:
            raise ValueError(f"No node with id: {nk}")

        wire_keys = [wk for wk, w in self.wires.items() if nk in (w.n1, w.n2)]
        element_keys = [ek for ek, e in self.elements.items() if nk in (e.plus.id, e.minus.id)]

        self._require_not_controlling(set(element_keys))

        for wk in wire_keys:
            del self.wires[wk]
            del self._wire_to_edge[wk]

        for ek in element_keys:
            self._delete_couplings_of(ek)
            del self.elements[ek]
            del self._element_to_edge[ek]

        del self.nodes[nk]
        self.graph.delete_vertex(nk)
        self.netstate = CircuitNetsState.OUT_OF_SYNC

    def _require_not_controlling(self, eks: set[str]) -> None:
        # A CCVS that stays in the circuit must not lose its control element.
        # One that is deleted together with its control is fine.
        for e in self.elements.values():
            if isinstance(e, CCVS) and e.id not in eks and e.control_id in eks:
                raise ValueError(f"Element {e.control_id} is the control of {e.id}; delete {e.id} first")

    def add_coupling(self, k: MagneticCoupling) -> None:
        if k.id in self.couplings:
            raise ValueError(f"Duplicate coupling with id: {k.id}")

        # Here we check that the elements being coupled exist and are couplable.
        for eid in (k.a_id, k.b_id):
            e = self.elements.get(eid)
            if e is None:
                raise ValueError(f"Coupling {k.id} refers to unknown element: {eid}")
            if not e.is_couplable:
                raise ValueError(f"Coupling {k.id} cannot couple element {eid} of kind {e.kind}")

        # TODO(validation): physical coupling limit. For m: |M| <= sqrt(L1 * L2).
        # For a complex z: a passivity condition on the real parts.

        # Is there an intersection between the sets of couplings for the two elements? If so, they are already coupled.
        shared = self._couplings_of.get(k.a_id, set()) & self._couplings_of.get(k.b_id, set())
        if shared:
            raise ValueError(f"Elements {k.a_id} and {k.b_id} are already coupled by: {shared.pop()}")

        self.couplings[k.id] = k
        self._couplings_of.setdefault(k.a_id, set()).add(k.id)
        self._couplings_of.setdefault(k.b_id, set()).add(k.id)

    def delete_coupling(self, kid: str) -> None:
        k = self.couplings.get(kid)
        if k is None:
            raise ValueError(f"No coupling with id: {kid}")

        self._couplings_of[k.a_id].discard(kid)
        self._couplings_of[k.b_id].discard(kid)
        del self.couplings[kid]

    def _delete_couplings_of(self, ek: str) -> None:
        for kid in list(self._couplings_of.get(ek, ())):
            self.delete_coupling(kid)
        self._couplings_of.pop(ek, None)

    def netblocks(self) -> list[list[str]]:
        if self.netstate == CircuitNetsState.OUT_OF_SYNC:
            self._rebuild_nets()
        return self.nets.blocks()

    def net_ids(self) -> list[str]:
        return [i[0] for i in self.netblocks()]

    def node_net(self, nid: str) -> str:
        net = [i[0] for i in self.netblocks() if nid in i]
        if len(net) == 0:
            raise ValueError(f"Node {nid} is not connected to any net")
        return net[0]
