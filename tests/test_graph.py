import pytest

from atlas.structs.graph import DiGraph, Edge


# Adding an element or wire adds an edge, and its terminals may not exist as
# vertices yet. Both ends must be created, and each must see the other as a
# neighbor regardless of the edge's direction (connectivity ignores direction).
def test_add_edge_creates_both_vertices() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    assert set(g.vertices) == {"a", "b"}
    assert g.neighbors("a") == ["b"]
    assert g.neighbors("b") == ["a"]


# Edge keys identify elements and wires in the graph. A duplicate would make one
# of them impossible to delete or look up.
def test_duplicate_edge_key_is_rejected() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    with pytest.raises(ValueError):
        g.add_edge(Edge("e1", "b", "c"))


# An edge from a vertex to itself is an element shorted onto one node. It has no
# meaning in the circuit, so the graph refuses it as a last line of defense.
def test_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        DiGraph().add_edge(Edge("e1", "a", "a"))


# Asking about something that isn't there must fail loudly with ValueError, not
# with a KeyError from inside the graph or, worse, silently.
def test_unknown_vertex_and_edge_raise() -> None:
    g = DiGraph()
    with pytest.raises(ValueError):
        g.neighbors("x")
    with pytest.raises(ValueError):
        g.edge("x")
    with pytest.raises(ValueError):
        g.delete_edge("x")
    with pytest.raises(ValueError):
        g.delete_vertex("x")


# Deleting an element must leave its terminals in place so something else can be
# connected to them. The edge has to disappear from both ends' adjacency lists,
# or a deleted element would still count as a connection.
def test_delete_edge_keeps_orphan_vertices() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    g.delete_edge("e1")
    assert set(g.vertices) == {"a", "b"}
    assert g.neighbors("a") == []
    assert "e1" not in g.edges


# An edge lives in two adjacency lists (its tail's and its head's). Deleting a
# vertex must remove its edges from the other ends too, otherwise those
# vertices keep "ghost" neighbors that no longer exist.
def test_delete_vertex_removes_its_edges_from_the_other_end() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    g.add_edge(Edge("e2", "c", "a"))
    g.delete_vertex("a")
    assert set(g.vertices) == {"b", "c"}
    assert g.neighbors("b") == []
    assert g.neighbors("c") == []
    assert g.edges == {}


# Connectivity is how a floating part of the circuit will be detected. It must
# follow edges in both directions and report an isolated vertex as disconnected.
# An empty graph has nothing to connect, so it counts as not connected.
def test_is_connected() -> None:
    g = DiGraph()
    assert not g.is_connected()
    g.add_edge(Edge("e1", "a", "b"))
    g.add_edge(Edge("e2", "c", "b"))
    assert g.is_connected()
    g.add_vertex("d")
    assert not g.is_connected()
