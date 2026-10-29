class DisjointSetMember:
    
    def __init__(self, key: str, rank: int = 0, name: str = "") -> None:
        self.p: "DisjointSetMember" = self
        self.key: str = key
        self.rank: int = rank
        self.name: str = name or key

    def find_root(self) -> "DisjointSetMember":
        current = self
        while current.p != current:
            current = current.p

        root, current = current, self
        while current.p != root:
            y = current
            current = current.p
            y.p = root

        return root

    #def __iter__(self) -> 

    
class DisjointSet:

    def __init__(self) -> None:
        self.members: dict[str, DisjointSetMember] = {}
        
    def find_member(self, key: str) -> DisjointSetMember | None:
        return self.members.get(key)

    def blocks(self) -> list[list[str]]:
        groups: dict[DisjointSetMember, list[str]] = {}
        for k, v in self.members.items():
            groups.setdefault(v.find_root(), []).append(k)
        return list(groups.values())

    def add(self, key: str) -> DisjointSetMember:
        node = self.members.get(key)
        if node is None:
            node = DisjointSetMember(key)
            self.members[key] = node
        return node

    def is_connected(self, x: DisjointSetMember, y: DisjointSetMember) -> bool:
        return self.find_root(x) == self.find_root(y)

    def find_root(self, x: DisjointSetMember) -> DisjointSetMember:
        return x.find_root()

    def link(self, x: DisjointSetMember, y: DisjointSetMember):
        if x.rank > y.rank:
            y.p = x
        else:
            x.p = y
        if x.rank == y.rank:
            # in this case `y` absorbed `x`, so `y.rank` has to increment in 1.
            y.rank += 1

    def union(self, x: DisjointSetMember, y: DisjointSetMember) -> None:
        rx, ry = self.find_root(x), self.find_root(y)
        if rx is not ry:
            return self.link(rx, ry)


