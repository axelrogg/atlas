from dataclasses import dataclass


@dataclass(frozen=True)
class KCL:
    node: str


@dataclass(frozen=True)
class Branch:
    element: str


@dataclass(frozen=True)
class V:
    node: str


@dataclass(frozen=True)
class I:
    element: str


type Row = KCL | Branch
type Col = V | I
