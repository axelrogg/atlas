from atlas.structs.diset import DisjointSet


# The circuit calls add() for a node every time it shows up (in a wire, in an
# element). It must hand back the existing member, or the same node would end
# up as two separate members and wires would stop merging it.
def test_add_returns_the_same_member_for_the_same_key() -> None:
    ds = DisjointSet()
    assert ds.add("a") is ds.add("a")
    assert ds.find_member("a") is ds.add("a")
    assert ds.find_member("missing") is None


# Before any wire, every node is its own net. If two fresh members looked
# connected, unrelated nodes would share a KCL row.
def test_new_members_are_their_own_block() -> None:
    ds = DisjointSet()
    a, b = ds.add("a"), ds.add("b")
    assert not ds.is_connected(a, b)
    assert ds.blocks() == [["a"], ["b"]]


# A chain of wires a-b-c makes a and c one net even though no wire joins them
# directly. Nets depend on this transitivity.
def test_union_merges_blocks_transitively() -> None:
    ds = DisjointSet()
    a, b, c, d = (ds.add(k) for k in "abcd")
    ds.union(a, b)
    ds.union(b, c)
    assert ds.is_connected(a, c)
    assert not ds.is_connected(a, d)


# A net is named after the first node in its block. Blocks must keep
# registration order (not root order) so net names, and therefore matrix
# labels, stay the same however the unions happened.
def test_blocks_keep_registration_order() -> None:
    ds = DisjointSet()
    a, b, c = (ds.add(k) for k in "abc")
    ds.union(c, a)
    assert ds.blocks() == [["a", "c"], ["b"]]


# Joining two members that are already in the same block must not re-link the
# roots; doing so could break the tree or the rank bookkeeping.
def test_union_of_the_same_block_changes_nothing() -> None:
    ds = DisjointSet()
    a, b = ds.add("a"), ds.add("b")
    ds.union(a, b)
    root = ds.find_root(a)
    ds.union(b, a)
    assert ds.find_root(a) is root
    assert ds.find_root(b) is root
