import math
from dataclasses import dataclass

from atlas.lines.materials import Material, dc_resistance
from atlas.core.utils import is_number_positive


def _require_conductor(what: str, material: Material) -> None:
    if not material.is_conductor:
        raise ValueError(f"{what}: material {material.name} is not a conductor.")


def _rdc20(material: Material, area: float, rdc20: float | None) -> float:
    # A stated rdc20 accounts for stranding and compaction; resistivity/area
    # doesn't, so it is only the fallback.
    if rdc20 is not None:
        return rdc20
    assert material.resistivity is not None
    return material.resistivity / area


# A conductor: the core of a cable, or one on its own in the ground (such as an
# earth continuity conductor), where it has no capacitance in the model.
@dataclass(frozen=True, kw_only=True)
class BareConductor:
    material: Material
    area: float                    # m²
    diameter: float | None = None  # m; some catalogs don't give it
    rdc20: float | None = None     # Ω/m at REFERENCE_TEMPERATURE
    # IEC 60287-1-1 Table 2 coefficients. 1.0 is the value for round stranded
    # conductors; other constructions (e.g. segmental) need their own.
    ks: float = 1.0
    kp: float = 1.0

    def __post_init__(self) -> None:
        _require_conductor("BareConductor", self.material)
        if not is_number_positive(self.area):
            raise ValueError(f"BareConductor area must be a positive finite number, got {self.area!r}.")
        if self.diameter is not None and not is_number_positive(self.diameter):
            raise ValueError(f"BareConductor diameter must be a positive finite number, got {self.diameter!r}.")
        if self.rdc20 is not None and not is_number_positive(self.rdc20):
            raise ValueError(f"BareConductor rdc20 must be a positive finite number, got {self.rdc20!r}.")
        if not is_number_positive(self.ks):
            raise ValueError(f"BareConductor ks must be a positive finite number, got {self.ks!r}.")
        if not is_number_positive(self.kp):
            raise ValueError(f"BareConductor kp must be a positive finite number, got {self.kp!r}.")

    def rdc(self, temperature: float) -> float:
        assert self.material.alpha is not None
        return dc_resistance(_rdc20(self.material, self.area, self.rdc20), self.material.alpha, temperature)

    # Missing data fails here, when it is used, not when the conductor is
    # created.
    def radius(self) -> float:
        if self.diameter is None:
            raise ValueError("BareConductor: the diameter is needed but was not given.")
        return self.diameter / 2

    # The GMR of a solid round conductor, as in the reference script.
    def gmr(self) -> float:
        return self.radius() * math.exp(-1 / 4)


@dataclass(frozen=True, kw_only=True)
class Insulation:
    material: Material
    diameter_over: float  # m, over the insulation as catalogs state it (outer semicon included)

    def __post_init__(self) -> None:
        if not self.material.is_dielectric:
            raise ValueError(f"Insulation: material {self.material.name} is not a dielectric.")
        if not is_number_positive(self.diameter_over):
            raise ValueError(f"Insulation diameter_over must be a positive finite number, got {self.diameter_over!r}.")


@dataclass(frozen=True, kw_only=True)
class Screen:
    material: Material
    area: float                         # m²
    diameter_over: float | None = None  # m, over the screen; some datasheets don't give it
    rdc20: float | None = None          # Ω/m at REFERENCE_TEMPERATURE

    def __post_init__(self) -> None:
        _require_conductor("Screen", self.material)
        if not is_number_positive(self.area):
            raise ValueError(f"Screen area must be a positive finite number, got {self.area!r}.")
        if self.diameter_over is not None and not is_number_positive(self.diameter_over):
            raise ValueError(f"Screen diameter_over must be a positive finite number, got {self.diameter_over!r}.")
        if self.rdc20 is not None and not is_number_positive(self.rdc20):
            raise ValueError(f"Screen rdc20 must be a positive finite number, got {self.rdc20!r}.")

    def rdc(self, temperature: float) -> float:
        assert self.material.alpha is not None
        return dc_resistance(_rdc20(self.material, self.area, self.rdc20), self.material.alpha, temperature)


@dataclass(frozen=True, kw_only=True)
class Jacket:
    diameter_over: float  # m, the cable's outer diameter

    def __post_init__(self) -> None:
        if not is_number_positive(self.diameter_over):
            raise ValueError(f"Jacket diameter_over must be a positive finite number, got {self.diameter_over!r}.")
