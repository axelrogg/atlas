from abc import ABC, abstractmethod
from dataclasses import dataclass

from atlas.lines.cable_parts import BareConductor, Insulation


# Any cable: a named design with a conductor and the insulation around it.
# What lies outside the insulation (screens, armour, jackets) and what the
# physics needs from it depends on the kind of cable, so each kind is a
# subclass that adds its own layers.
@dataclass(frozen=True, kw_only=True)
class Cable(ABC):
    name: str
    conductor: BareConductor
    insulation: Insulation

    def __post_init__(self) -> None:
        # Layers grow from the inside out. Only the diameters that are given
        # can be checked.
        given = [(what, d) for what, d in self.layers() if d is not None]
        for (inner, d_inner), (outer, d_outer) in zip(given, given[1:]):
            if d_inner >= d_outer:
                raise ValueError(f"Cable {self.name}: {inner} ({d_inner} m) must be smaller than {outer} ({d_outer} m).")

    # Every layer's diameter, from the inside out, for the check above. A
    # subclass extends it with its own layers.
    def layers(self) -> list[tuple[str, float | None]]:
        return [
            ("conductor diameter", self.conductor.diameter),
            ("insulation diameter_over", self.insulation.diameter_over),
        ]

    # m, over the whole cable: what touches the next cable.
    @property
    @abstractmethod
    def outer_diameter(self) -> float: ...

    # Missing data fails here, when it is used, not when the cable is created,
    # so incomplete catalog entries can still be described. The conductor,
    # once its diameter is known to be given, so the error names the cable.
    def _measured_conductor(self) -> BareConductor:
        if self.conductor.diameter is None:
            raise ValueError(f"Cable {self.name}: the conductor diameter is needed but was not given.")
        return self.conductor

    def conductor_radius(self) -> float:
        return self._measured_conductor().radius()

    def conductor_gmr(self) -> float:
        return self._measured_conductor().gmr()

    def conductor_rdc(self, temperature: float) -> float:
        return self.conductor.rdc(temperature)
