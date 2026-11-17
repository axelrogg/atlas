# atlas

A steady-state (phasor) circuit simulator based on Modified Nodal Analysis, built
as the core for simulating shielded underground power cables.

You pick a frequency, describe a circuit by placing elements and connecting
their terminals, and get every node voltage and every element current back as
complex phasors.

> **Status:** early. The solver works for the elements listed below, and
> `atlas.lines` has the physics of cables (materials, resistances, Carson's
> earth-return impedance). Turning a cable line into a circuit, and most
> validation, are still to come (see [Limitations](#limitations)).

## Install

Requires Python 3.14 and [Poetry](https://python-poetry.org/).

```sh
poetry install
```

## Quickstart

A 10 V source driving two resistors in series:

```python
from atlas import ElementKind, Simulator

sim = Simulator(f=50)

v1 = sim.add_element(ElementKind.VOLTAGE_SOURCE, 10)
r1 = sim.add_element(ElementKind.RESISTOR, 2.0, plus=v1.plus)
r2 = sim.add_element(ElementKind.RESISTOR, 3.0, minus=sim.ground)

sim.connect(r1.minus, r2.plus)
sim.connect(v1.minus, sim.ground)

sol = sim.solve()

sol.voltages[r2.plus.id]       # 6 V
sol.element_currents[r1.id]    # 2 A
sol.element_currents[v1.id]    # -2 A (the source pushes current out of its plus terminal)
```

Values are complex numbers and carry the usual floating-point rounding, so
`6.000000000000001` is 6.

## How it works

### Elements and terminals

`add_element(kind, value=None, plus=None, minus=None, control=None, name="",
voltage=None, current=None)` creates an element and returns it. Every element
has two terminals, `plus` and `minus`.

- A terminal you don't pass gets its own new point, with an integer id (`1`, `2`, ...;
  ground is `0`).
- A terminal you do pass is shared: `plus=v1.plus` puts the new element on the
  same point as `v1`'s plus terminal. `sim.ground` is the reference point (0 V).

You never create nodes or ids yourself. Ids are generated per kind:

| Kind | `ElementKind` | `value` | Id prefix |
|---|---|---|---|
| Resistor | `RESISTOR` | resistance (Ω, real) | `R` |
| Inductor | `INDUCTOR` | inductance (H, real) | `L` |
| Capacitor | `CAPACITOR` | capacitance (F, real) | `C` |
| Impedance | `IMPEDANCE` | impedance (Ω, complex) | `Z` |
| Voltage source | `VOLTAGE_SOURCE` | voltage (V, complex) | `V` |
| Current source | `CURRENT_SOURCE` | current (A, complex) | `I` |
| Current-controlled voltage source | `CCVS` | gain (Ω, real), plus `control=<element>` | `H` |
| Fixator | `FIXATOR` | none; `voltage=` (V) and `current=` (A) instead | `FX` |
| Norator | `NORATOR` | none | `NR` |

`name` is an optional display label; it doesn't affect the element's identity.
Arguments a kind doesn't use are refused rather than ignored: a `value` for a
fixator or norator, `voltage`/`current` for anything but a fixator, `control`
for anything but a CCVS.

### Fixators and norators

They describe what is *known* where the circuit meets the outside world:

- A **fixator** imposes both the voltage across it and the current through it
  (for example "10 V and 2 A at the input").
- A **norator** imposes nothing; its voltage and current are whatever the circuit
  makes them (for example "the output, where nothing is known").

They always come in pairs: a fixator adds one condition too many and a norator
one too few, so `solve` refuses a circuit with more of one than the other.

```python
fx = sim.add_element(ElementKind.FIXATOR, minus=sim.ground, voltage=10, current=-2)
r = sim.add_element(ElementKind.RESISTOR, 2.0, plus=fx.plus)
nr = sim.add_element(ElementKind.NORATOR, plus=r.minus, minus=sim.ground)
# V at the output = 10 - 2 * 2 = 6 V, and 2 A leave through the norator.
```

Like every element, a fixator's `current` is positive from `plus` to `minus`, so
current fed *into* the circuit from ground is negative.

### Connecting

`sim.connect(a, b)` makes two terminals the same point and returns the wire it
created. Connecting two points that are already connected does nothing and
returns `None`, so you can connect freely without creating loops of wires.

### Mutual coupling

`sim.couple_elements(a, b, m=...)` couples two elements by a mutual inductance
`m` (H), or `z=...` for a complex mutual impedance (Ω). Resistors, inductors,
capacitors and impedances can be coupled.

The dot of each element is at its `plus` terminal, so the sign of the coupling
follows the elements' orientation: reversing one element's `plus` and `minus`
reverses the mutual voltage.

### Removing

`sim.delete_element(e)` and `sim.delete_wire(w)` remove things again.
Deleting an element also deletes its couplings. Points left with nothing attached
are removed. An element that controls a CCVS can't be deleted until the CCVS is.

### Results

`Simulator(f)` fixes the frequency `f` (Hz, must be positive) for everything
built in it, and `sim.solve()` solves at that frequency and returns a
`Solution`:

- `sol.voltages[node_id]`: the voltage of every point, relative to ground.
- `sol.element_currents[element_id]`: the current through every element.
  A positive current flows through the element from `plus` to `minus`.
- `sol.wire_currents[wire.id]`: the current through every wire returned by
  `connect`. A positive current flows from the wire's first point to its second
  (`connect(a, b)` means from `a` to `b`).

To solve at another frequency, build the circuit in another `Simulator`. The
frequency is fixed when the simulator is created because element values worked
out from physical data, such as a cable's per-metre impedance, depend on it.

## Limitations

- **Errors in a circuit** (a floating point, a shorted voltage source, a loop of
  voltage sources) are reported as "System matrix is singular" without saying
  which part is at fault.
- **Cable lines** can't be placed in a circuit yet. `atlas.lines` computes their
  physics, but the layer that turns a line into circuit elements is being
  redesigned.
- **Steady state only.** There is no transient (time-domain) analysis, and no DC
  analysis (`f` must be greater than 0).

## Project layout

```
src/atlas/
  _atlas.py        Simulator, the public entry point
  core/
    model/         circuit, elements, nodes, couplings
    mna/           matrix layout, stamps and the solver engine
    structs/       disjoint set and graph
    utils.py
  lines/           cable physics
    materials.py   materials and DC resistance at temperature
    cable_parts.py conductor, insulation, screen and jacket
    cable.py       the Cable base class
    iec60287.py    IEC 60287 skin and proximity effect
    carson.py      Carson–Clem series impedance with earth return
    geometry.py    positions, trefoil and flat formations, overlap check
```

## Development

```sh
poetry install
poetry run pre-commit install   # install the git hooks
poetry run pytest
```

The hooks block a commit if a test fails or if the commit message doesn't follow
[Conventional Commits](https://www.conventionalcommits.org/) (`feat: ...`,
`fix: ...`, `test: ...`, `docs: ...`, `refactor: ...`, `build: ...`).

### Releasing

The version lives in `pyproject.toml` (and is available as `atlas.__version__`).
To release, run:

```sh
poetry run cz bump
```

It picks the next version from the commits since the last tag (`fix` → patch,
`feat` → minor, breaking change → major; below 1.0 a breaking change bumps the
minor version), updates `pyproject.toml` and `CHANGELOG.md`, commits, and creates
a signed `vX.Y.Z` tag.
