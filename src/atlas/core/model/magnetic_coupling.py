from dataclasses import dataclass
from typing import ClassVar

from atlas.core.mna.labels import Branch, I
from atlas.core.mna.stamp import Stamp, StampEntry
from atlas.core.utils import is_number_valid, omega


@dataclass(frozen=True)
class MagneticCoupling:
    id_prefix: ClassVar[str] = "K"
    id: str
    a_id: str
    b_id: str
    m: float | None = None
    z: complex | None = None

    def __post_init__(self):
        if self.a_id == self.b_id:
            raise ValueError(f"Coupling {self.id} cannot couple element {self.a_id} with itself.")
        if (self.m is None) == (self.z is None):
            raise ValueError(f"Coupling {self.id} must define exactly one of m or z.")
        if self.m is not None and not is_number_valid(self.m):
            raise ValueError(f"Coupling {self.id} must have a finite mutual inductance, got {self.m}.")
        if self.z is not None and not is_number_valid(self.z):
            raise ValueError(f"Coupling {self.id} must have a finite mutual impedance, got {self.z}.")

    def impedance(self, f: float) -> complex:
        if self.m is not None:
            return 1j * omega(f) * self.m
        assert self.z is not None, f"Coupling {self.id} must define exactly one of m or z."
        return self.z

    def stamp(self, f: float) -> Stamp:
        z = self.impedance(f)
        return Stamp(
            entries=[
                StampEntry(Branch(self.a_id), I(self.b_id), - z),
                StampEntry(Branch(self.b_id), I(self.a_id), - z),
            ]
        )
