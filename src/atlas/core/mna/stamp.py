from dataclasses import dataclass, field

from atlas.core.mna.labels import Col, Row


@dataclass(frozen=True)
class StampEntry:
    row: Row
    col: Col
    value: complex


@dataclass(frozen=True)
class StampRHSEntry:
    row: Row
    value: complex


@dataclass(frozen=True)
class Stamp:
    entries: list[StampEntry] = field(default_factory=list)
    rhs: list[StampRHSEntry] = field(default_factory=list)
