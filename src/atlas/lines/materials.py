from dataclasses import dataclass

from atlas.core.utils import is_number_positive, is_real_number_valid


EPSILON_0 = 8.8541878128e-12  # F/m, vacuum permittivity

# The temperature resistivities and catalog rdc20 values are quoted at, in °C.
REFERENCE_TEMPERATURE = 20.0


@dataclass(frozen=True, kw_only=True)
class Material:
    name: str
    resistivity: float | None = None   # Ω·m at REFERENCE_TEMPERATURE; None = not a conductor
    alpha: float | None = None         # 1/K, temperature coefficient of resistance
    permittivity: float | None = None  # relative; None = not a dielectric

    def __post_init__(self) -> None:
        if self.resistivity is not None:
            if not is_number_positive(self.resistivity):
                raise ValueError(f"Material {self.name}: resistivity must be a positive finite number, got {self.resistivity!r}.")
            if self.alpha is None:
                raise ValueError(f"Material {self.name}: a conductor needs alpha to correct its resistance for temperature.")
        # alpha can be zero or negative for some alloys, so it only has to be finite.
        if self.alpha is not None:
            if self.resistivity is None:
                raise ValueError(f"Material {self.name}: alpha is only used with a resistivity.")
            if not is_real_number_valid(self.alpha):
                raise ValueError(f"Material {self.name}: alpha must be a finite real number, got {self.alpha!r}.")
        if self.permittivity is not None:
            if not is_number_positive(self.permittivity):
                raise ValueError(f"Material {self.name}: permittivity must be a positive finite number, got {self.permittivity!r}.")
            if self.permittivity < 1:
                raise ValueError(f"Material {self.name}: relative permittivity can't be below 1, got {self.permittivity}.")

    @property
    def is_conductor(self) -> bool:
        return self.resistivity is not None

    @property
    def is_dielectric(self) -> bool:
        return self.permittivity is not None


# Resistance at `temperature` (°C) from the resistance at REFERENCE_TEMPERATURE,
# with the linear coefficient alpha (1/K): any conductor.
def dc_resistance(rdc20: float, alpha: float, temperature: float) -> float:
    if not is_number_positive(rdc20):
        raise ValueError(f"rdc20 must be a positive finite number, got {rdc20!r}.")
    if not is_real_number_valid(alpha):
        raise ValueError(f"alpha must be a finite real number, got {alpha!r}.")
    if not is_real_number_valid(temperature):
        raise ValueError(f"temperature must be a finite real number, got {temperature!r}.")
    return rdc20 * (1 + alpha * (temperature - REFERENCE_TEMPERATURE))


# IEC 60287-1-1, Table 1 (resistivity, alpha) and Table 3 (permittivity).
COPPER = Material(name="copper", resistivity=1.7241e-8, alpha=3.93e-3)
ALUMINIUM = Material(name="aluminium", resistivity=2.8264e-8, alpha=4.03e-3)
XLPE = Material(name="xlpe", permittivity=2.5)
HEPR = Material(name="hepr", permittivity=3.0)
