import pytest

from atlas.structs.graph import DiGraph, Edge


def test_add_edge_creates_both_vertices() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    assert set(g.vertices) == {"a", "b"}
    assert g.neighbors("a") == ["b"]
    assert g.neighbors("b") == ["a"]


def test_duplicate_edge_key_is_rejected() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    with pytest.raises(ValueError):
        g.add_edge(Edge("e1", "b", "c"))


def test_self_loop_is_rejected() -> None:
    with pytest.raises(ValueError):
        DiGraph().add_edge(Edge("e1", "a", "a"))


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


def test_delete_edge_keeps_orphan_vertices() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    g.delete_edge("e1")
    assert set(g.vertices) == {"a", "b"}
    assert g.neighbors("a") == []
    assert "e1" not in g.edges


def test_delete_vertex_removes_its_edges_from_the_other_end() -> None:
    g = DiGraph()
    g.add_edge(Edge("e1", "a", "b"))
    g.add_edge(Edge("e2", "c", "a"))
    g.delete_vertex("a")
    assert set(g.vertices) == {"b", "c"}
    assert g.neighbors("b") == []
    assert g.neighbors("c") == []
    assert g.edges == {}


def test_is_connected() -> None:
    g = DiGraph()
    assert not g.is_connected()
    g.add_edge(Edge("e1", "a", "b"))
    g.add_edge(Edge("e2", "c", "b"))
    assert g.is_connected()
    g.add_vertex("d")
    assert not g.is_connected()
