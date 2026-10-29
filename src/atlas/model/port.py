from dataclasses import dataclass, field

from atlas.utils import is_number_valid

from atlas.model.elements import Element, VoltageSource
from atlas.model.node import Node, GROUND_NODE


@dataclass(frozen=True)
class Port:
    id: str
    node: Node
    name: str = field(compare=False, default="")
    voltage: complex | None = None
    current: complex | None = None

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.id)

        if self.voltage is not None and not is_number_valid(self.voltage):
            raise ValueError(f"Port {self.id} has an invalid voltage value: {self.voltage}")

        if self.current is not None and not is_number_valid(self.current):
            raise ValueError(f"Port {self.id} has an invalid current value: {self.current}")

        if self.node.id == GROUND_NODE.id:
            raise ValueError(f"Port {self.id} cannot be connected to the ground node.")

    def rename(self, name: str) -> None:
        object.__setattr__(self, "name", name)

    def to_element(self) -> Element:
        if self.voltage is not None and self.current is None:
            # If we know the voltage but not the current, we can model this as a voltage source with the given voltage and the ground node as the minus terminal.
            return VoltageSource(id=self.id, name=self.name, value=self.voltage, plus=self.node, minus=GROUND_NODE)
        if self.voltage is not None and self.current is not None:
            # If we know both the voltage and the current, we can model this as
            # an Independent Voltage Source with its own edge current known.
            return VoltageSource(id=self.id, name=self.name, value=self.voltage, plus=self.node, minus=GROUND_NODE)
        if self.voltage is None and self.current is None:
            # If we don't know either the voltage or the current, i dont know yet how we can model it.
            raise NotImplementedError("Port with unknown voltage and current is not yet supported.")

        raise NotImplementedError("Port with known current but unknown voltage is not yet supported.")
        
