from dataclasses import dataclass, field

@dataclass(frozen=True)
class Node:
    id: str
    name: str = field(compare=False)

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.id)


GROUND_NODE = Node(id="0", name="Ground")