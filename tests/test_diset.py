from atlas.structs.diset import DisjointSet


def test_add_returns_the_same_member_for_the_same_key() -> None:
    ds = DisjointSet()
    assert ds.add("a") is ds.add("a")
    assert ds.find_member("a") is ds.add("a")
    assert ds.find_member("missing") is None


def test_new_members_are_their_own_block() -> None:
    ds = DisjointSet()
    a, b = ds.add("a"), ds.add("b")
    assert not ds.is_connected(a, b)
    assert ds.blocks() == [["a"], ["b"]]


def test_union_merges_blocks_transitively() -> None:
    ds = DisjointSet()
    a, b, c, d = (ds.add(k) for k in "abcd")
    ds.union(a, b)
    ds.union(b, c)
    assert ds.is_connected(a, c)
    assert not ds.is_connected(a, d)


def test_blocks_keep_registration_order() -> None:
    ds = DisjointSet()
    a, b, c = (ds.add(k) for k in "abc")
    ds.union(c, a)
    assert ds.blocks() == [["a", "c"], ["b"]]


def test_union_of_the_same_block_changes_nothing() -> None:
    ds = DisjointSet()
    a, b = ds.add("a"), ds.add("b")
    ds.union(a, b)
    root = ds.find_root(a)
    ds.union(b, a)
    assert ds.find_root(a) is root
    assert ds.find_root(b) is root
