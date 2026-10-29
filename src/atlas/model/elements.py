from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import ClassVar

from atlas.mna.labels import KCL, Branch, V, I
from atlas.mna.stamp import Stamp, StampEntry, StampRHSEntry
from atlas.model.node import Node
from atlas.utils import is_number_valid, omega


# Each value is the id prefix for that kind (SPICE letters). Values must stay
# unique: a repeated value would silently make two kinds the same member.
@unique
class ElementKind(Enum):
    RESISTOR = "R"
    INDUCTOR = "L"
    CAPACITOR = "C"
    IMPEDANCE = "Z"
    VOLTAGE_SOURCE = "V"
    CURRENT_SOURCE = "I"
    CCVS = "H"
    #VCVS = "E"
    #VCCS = "G"
    #CCCS = "F"


class ElementGroup(Enum):
    G1 = auto()
    G2 = auto()


@dataclass(frozen=True)
class Element(ABC):
    id: str
    kind: ClassVar[ElementKind]
    is_couplable: ClassVar[bool] = False
    plus: Node
    minus: Node
    name: str = field(default="", kw_only=True, compare=False)
    group: ElementGroup = field(default=ElementGroup.G1, kw_only=True, compare=False)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "kind" not in vars(cls):
            raise TypeError(f"{cls.__name__} must define kind.")

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.id)
        if self.plus == self.minus:
            raise ValueError(f"Element {self.name} has the same plus and minus nodes: {self.plus}. No self-loops allowed.")

    def regroup(self, g: ElementGroup) -> None:
        if self.kind in {ElementKind.VOLTAGE_SOURCE, ElementKind.CCVS}:
            raise ValueError(f"Element {self.name} of kind {self.kind} cannot be regrouped; it is always in group G2.")
        object.__setattr__(self, "group", g)

    def rename(self, name: str) -> None:
        object.__setattr__(self, "name", name or self.id)

    @abstractmethod
    def stamp(self, f: float) -> Stamp: ...


@dataclass(frozen=True)
class Resistor(Element):
    kind: ClassVar[ElementKind] = ElementKind.RESISTOR
    is_couplable: ClassVar[bool] = True
    r: float

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.r) or self.r < 0:
            raise ValueError(f"Resistor {self.name} must have a finite non-negative resistance, got {self.r}.")

    @property
    def impedance(self) -> complex:
        return self.r

    @property
    def admittance(self) -> complex:
        if self.impedance == 0:
            raise ValueError(f"Resistor {self.name} has zero resistance; admittance is infinite.")
        return 1 / self.r

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        if self.group is ElementGroup.G1:
            y = self.admittance
            return Stamp(
                entries=[
                    StampEntry(KCL(p), V(p), y),
                    StampEntry(KCL(p), V(m), - y),
                    StampEntry(KCL(m), V(p), - y),
                    StampEntry(KCL(m), V(m), y),
                ]
            )
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
                StampEntry(Branch(self.id), I(self.id), - self.r),
            ]
        )

@dataclass(frozen=True)
class Inductor(Element):
    kind: ClassVar[ElementKind] = ElementKind.INDUCTOR
    is_couplable: ClassVar[bool] = True
    l: float

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.l) or self.l < 0:
            raise ValueError(f"Inductor {self.name} must have a finite non-negative inductance, got {self.l}.")

    def impedance(self, f: float) -> complex:
        return 1j * omega(f) * self.l

    def admittance(self, f: float) -> complex:
        if self.impedance(f) == 0:
            raise ValueError(f"Inductor {self.name} has zero impedance at f = {f} Hz; admittance is infinite.")
        return 1 / self.impedance(f)

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        if self.group is ElementGroup.G1:
            y = self.admittance(f)
            return Stamp(
                entries=[
                    StampEntry(KCL(p), V(p), y),
                    StampEntry(KCL(p), V(m), - y),
                    StampEntry(KCL(m), V(p), - y),
                    StampEntry(KCL(m), V(m), y),
                ]
            )
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
                StampEntry(Branch(self.id), I(self.id), - self.impedance(f)),
            ]
        )


@dataclass(frozen=True)
class Capacitor(Element):
    kind: ClassVar[ElementKind] = ElementKind.CAPACITOR
    is_couplable: ClassVar[bool] = True
    c: float

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.c) or self.c < 0:
            raise ValueError(f"Capacitor {self.name} must have a finite non-negative capacitance, got {self.c}.")

    def admittance(self, f: float) -> complex:
        return 1j * omega(f) * self.c

    def impedance(self, f: float) -> complex:
        if self.admittance(f) == 0:
            raise ValueError(f"Capacitor {self.name} has zero admittance at f = {f} Hz; impedance is infinite.")
        return 1 / self.admittance(f)

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        if self.group is ElementGroup.G1:
            y = self.admittance(f)
            return Stamp(
                entries=[
                    StampEntry(KCL(p), V(p), y),
                    StampEntry(KCL(p), V(m), - y),
                    StampEntry(KCL(m), V(p), - y),
                    StampEntry(KCL(m), V(m), y),
                ]
            )
        
        if self.admittance(f) == 0:
            raise ValueError(f"Capacitor {self.name} has zero admittance at f = {f} Hz; cannot stamp in group G2.")
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
                StampEntry(Branch(self.id), I(self.id), - self.impedance(f)),
            ]
        )


@dataclass(frozen=True)
class Impedance(Element):
    kind: ClassVar[ElementKind] = ElementKind.IMPEDANCE
    is_couplable: ClassVar[bool] = True
    z: complex

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.z) or self.z.real < 0:
            raise ValueError(f"Impedance {self.name} must have a finite impedance with non-negative real part, got {self.z}.")

    @property
    def impedance(self) -> complex:
        return self.z

    @property
    def admittance(self) -> complex:
        if self.impedance == 0:
            raise ValueError(f"Impedance {self.name} has zero impedance; admittance is infinite.")
        return 1 / self.z

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        if self.group is ElementGroup.G1:
            y = self.admittance
            return Stamp(
                entries=[
                    StampEntry(KCL(p), V(p), y),
                    StampEntry(KCL(p), V(m), - y),
                    StampEntry(KCL(m), V(p), - y),
                    StampEntry(KCL(m), V(m), y),
                ]
            )
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
                StampEntry(Branch(self.id), I(self.id), - self.z),
            ]
        )


@dataclass(frozen=True)
class VoltageSource(Element):
    kind: ClassVar[ElementKind] = ElementKind.VOLTAGE_SOURCE
    value: complex
    group: ElementGroup = field(default=ElementGroup.G2, kw_only=True, compare=False)

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.value):
            raise ValueError(f"Voltage source {self.name} must have a finite voltage, got {self.value}.")

        if self.group is not ElementGroup.G2:
            raise ValueError(f"Voltage source {self.name} must be in group G2; it cannot be regrouped to G1.")

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
            ],
            rhs=[
                StampRHSEntry(Branch(self.id), self.value),
            ]
        )


@dataclass(frozen=True)
class CurrentSource(Element):
    kind: ClassVar[ElementKind] = ElementKind.CURRENT_SOURCE
    value: complex

    def __post_init__(self):
        super().__post_init__()
        if not is_number_valid(self.value):
            raise ValueError(f"Current source {self.name} must have a finite current, got {self.value}.")

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        if self.group is ElementGroup.G1:
            return Stamp(
                rhs=[
                    StampRHSEntry(KCL(p), - self.value),
                    StampRHSEntry(KCL(m), self.value),
                ]
            )
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), I(self.id), 1),
            ],
            rhs=[
                StampRHSEntry(Branch(self.id), self.value),
            ]
        )


@dataclass(frozen=True)
class CCVS(Element):
    kind: ClassVar[ElementKind] = ElementKind.CCVS
    gain: float
    control_id: str
    group: ElementGroup = field(default=ElementGroup.G2, kw_only=True, compare=False)

    def __post_init__(self):
        super().__post_init__()

        if not is_number_valid(self.gain):
            raise ValueError(f"CCVS {self.name} must have a finite gain, got {self.gain}.")
        if not self.control_id:
            raise ValueError(f"CCVS {self.name} must have a non-empty control_id.")

        if self.control_id == self.id:
            raise ValueError(f"CCVS {self.name} cannot control itself; control_id must be different from its own id.")

        if self.group is not ElementGroup.G2:
            raise ValueError(f"CCVS {self.name} must be in group G2; it cannot be regrouped to G1.")

    def stamp(self, f: float) -> Stamp:
        p, m = self.plus.id, self.minus.id
        return Stamp(
            entries=[
                StampEntry(KCL(p), I(self.id), 1),
                StampEntry(KCL(m), I(self.id), - 1),
                StampEntry(Branch(self.id), V(p), 1),
                StampEntry(Branch(self.id), V(m), - 1),
                StampEntry(Branch(self.id), I(self.control_id), - self.gain),
            ]
        )
