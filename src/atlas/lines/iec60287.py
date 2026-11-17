# AC resistance of cable conductors, IEC 60287-1-1. Empirical fits for cable
# constructions and layouts; not the general physics of skin and proximity
# effects, so they don't apply to overhead conductors.
import math

from atlas.core.utils import is_number_positive


# IEC 60287-1-1, 2.1.2. rdc is in Ω/m at the operating temperature, so the
# standard's 1e-7 factor applies as written.
def skin_effect_factor(rdc: float, f: float, ks: float = 1.0) -> float:
    if not is_number_positive(rdc):
        raise ValueError(f"rdc must be a positive finite number, got {rdc!r}.")
    if not is_number_positive(f):
        raise ValueError(f"f must be a positive finite number, got {f!r}.")
    xs2 = 8 * math.pi * f * 1e-7 * ks / rdc
    xs = math.sqrt(xs2)
    if xs <= 2.8:
        return xs2**2 / (192 + 0.8 * xs2**2)
    if xs <= 3.8:
        return -0.136 - 0.0177 * xs + 0.0563 * xs2
    return 0.354 * xs - 0.733


# TODO (todo.txt, Cables): IEC assumes balanced currents; generalize later.
# IEC 60287-1-1, 2.1.4. diameter and spacing (axis to axis) only matter as a
# ratio. The standard gives formulas for two-core cables (or two single-core
# cables) and for three-core cables (or three single-core cables) only, and
# only for xp <= 2.8.
def proximity_effect_factor(rdc: float, f: float, diameter: float, spacing: float, conductors: int, kp: float = 1.0) -> float:
    if not is_number_positive(rdc):
        raise ValueError(f"rdc must be a positive finite number, got {rdc!r}.")
    if not is_number_positive(f):
        raise ValueError(f"f must be a positive finite number, got {f!r}.")
    if not is_number_positive(diameter):
        raise ValueError(f"diameter must be a positive finite number, got {diameter!r}.")
    if not is_number_positive(spacing):
        raise ValueError(f"spacing must be a positive finite number, got {spacing!r}.")
    if conductors not in (2, 3):
        raise ValueError(f"IEC 60287 gives the proximity effect for 2 or 3 conductors only, got {conductors}.")
    xp2 = 8 * math.pi * f * 1e-7 * kp / rdc
    fxp = xp2**2 / (192 + 0.8 * xp2**2)
    ratio2 = (diameter / spacing) ** 2
    if conductors == 2:
        return 2.9 * fxp * ratio2
    return fxp * ratio2 * (0.312 * ratio2 + 1.18 / (fxp + 0.27))


def ac_resistance(rdc: float, ys: float, yp: float) -> float:
    return rdc * (1 + ys + yp)
