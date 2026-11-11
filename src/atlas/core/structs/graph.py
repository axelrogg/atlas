from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    key: str
    tail: str  # out
    head: str  # in


class DiGraph:

    def __init__(self) -> None:
        self.inn: dict[str, list[Edge]] = {}
        self.out: dict[str, list[Edge]] = {}
        self.edges: dict[str, Edge] = {}

    def add_vertex(self, key: str) -> None:
        if key not in self.out:
            self.inn[key] = []
            self.out[key] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.key in self.edges:
            raise ValueError(f"Duplicate edge key: {edge.key!r}")

        # ===========================
        # =  No self-loops allowed  =
        # ===========================
        if edge.head == edge.tail:
            raise ValueError(f"Self-loop on {edge.tail!r}")

        self.add_vertex(edge.head)
        self.add_vertex(edge.tail)

        self.inn[edge.head].append(edge)
        self.out[edge.tail].append(edge)

        self.edges[edge.key] = edge

    def neighbors(self, v: str) -> list[str]:
        if self.inn.get(v) is None:
            raise ValueError(f"Unknown vertex: {v!r}")

        nbors = [e.tail for e in self.inn[v]]
        nbors.extend([e.head for e in self.out[v]])
        return list(set(nbors))

    def edge(self, key: str) -> Edge:
        e = self.edges.get(key)
        if e is None:
            raise ValueError(f"Unknown edge: {key!r}")

        return e

    def delete_vertex(self, v: str) -> None:
        if v not in self.out:
            raise ValueError(f"Unknown vertex: {v}")

        for e in self.out.pop(v):
            del self.edges[e.key]
            # we already removed from _out but we need to remove it from _in now,
            # since an edge also lives in its other endpoint's list and that copy has to go too.
            self.inn[e.head].remove(e)
        for e in self.inn.pop(v):
            del self.edges[e.key]
            # Same as above, we already removed from _in but we need to remove it from _in now,
            # since an edge also lives in its other endpoint's list and that copy has to go too.
            self.out[e.tail].remove(e)

    def delete_edge(self, key: str) -> None:
        e = self.edges.get(key)
        if e is None:
            raise ValueError(f"Unknown edge: {key!r}")

        self.inn[e.head].remove(e)
        self.out[e.tail].remove(e)
        del self.edges[key]

    @property
    def vertices(self) -> Iterator[str]:
        yield from self.out

    def __len__(self) -> int:  # Number of vertices
        return len(self.inn)

    def is_connected(self) -> bool:

        # We need to implement breadth-first search here.
        # We just need to know if every vertex is reachable from a source (any vertex can count as such).
        if len(self) == 0:
            return False

        start = list(self.inn.keys())[0]
        visited: set[str] = {start}
        q = deque([start])

        while len(q) != 0:
            v_lookup = q.popleft()
            for v in self.neighbors(v_lookup):
                if v not in visited:
                    visited.add(v)
                    q.append(v)

        return len(visited) == len(self)



