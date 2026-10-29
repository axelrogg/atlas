# atlas

A steady-state (phasor) circuit simulator based on Modified Nodal Analysis, built
as the core for simulating shielded underground power cables.

You describe a circuit by placing elements and connecting their terminals, pick a
frequency, and get every node voltage and every element current back as complex
phasors.

> **Status:** early. The solver works for the elements listed below, but cable
> models, ports, wire currents and most validation are still to come (see
> [Limitations](#limitations)).

## Install

Requires Python 3.14 and [Poetry](https://python-poetry.org/).

```sh
poetry install
```

## Quickstart

A 10 V source driving two resistors in series:

```python
from atlas import Atlas, ElementKind

atlas = Atlas()

v1 = atlas.add_element(ElementKind.VOLTAGE_SOURCE, 10)
r1 = atlas.add_element(ElementKind.RESISTOR, 2.0, plus=v1.plus)
r2 = atlas.add_element(ElementKind.RESISTOR, 3.0, minus=atlas.ground)

atlas.connect(r1.minus, r2.plus)
atlas.connect(v1.minus, atlas.ground)

sol = atlas.solve(f=50)

sol.voltages[r2.plus.id]       # 6 V
sol.element_currents[r1.id]    # 2 A
sol.element_currents[v1.id]    # -2 A (the source pushes current out of its plus terminal)
```

Values are complex numbers and carry the usual floating-point rounding, so
`6.000000000000001` is 6.

## How it works

### Elements and terminals

`add_element(kind, value, plus=None, minus=None, control=None, name="")` creates
an element and returns it. Every element has two terminals, `plus` and `minus`.

- A terminal you don't pass gets its own new point (named `R1+`, `R1-`, ...).
- A terminal you do pass is shared: `plus=v1.plus` puts the new element on the
  same point as `v1`'s plus terminal. `atlas.ground` is the reference point (0 V).

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

`name` is an optional display label; it doesn't affect the element's identity.

### Connecting

`atlas.connect(a, b)` makes two terminals the same point and returns the wire it
created. Connecting two points that are already connected does nothing and
returns `None`, so you can connect freely without creating loops of wires.

### Mutual coupling

`atlas.couple_elements(a, b, m=...)` couples two elements by a mutual inductance
`m` (H), or `z=...` for a complex mutual impedance (Ω). Resistors, inductors,
capacitors and impedances can be coupled.

The dot of each element is at its `plus` terminal, so the sign of the coupling
follows the elements' orientation: reversing one element's `plus` and `minus`
reverses the mutual voltage.

### Removing

`atlas.delete_element(e)` and `atlas.delete_wire(w)` remove things again.
Deleting an element also deletes its couplings. Points left with nothing attached
are removed. An element that controls a CCVS can't be deleted until the CCVS is.

### Results

`atlas.solve(f)` solves at frequency `f` (Hz, must be positive) and returns a
`Solution`:

- `sol.voltages[node_id]`: the voltage of every point, relative to ground.
- `sol.element_currents[element_id]`: the current through every element.
  A positive current flows through the element from `plus` to `minus`.

Solving at another frequency is just another `solve` call.

## Limitations

- **Wire currents** are not computed yet, only element currents.
- **Errors in a circuit** (a floating point, a shorted voltage source, a loop of
  voltage sources) are reported as "System matrix is singular" without saying
  which part is at fault.
- **Ports** and **cable templates** are not implemented yet.
- **Steady state only.** There is no transient (time-domain) analysis, and no DC
  analysis (`f` must be greater than 0).

## Project layout

```
src/atlas/
  _atlas.py      Atlas, the public entry point
  model/         circuit, elements, nodes, couplings, ports
  mna/           matrix layout, stamps and the solver engine
  structs/       disjoint set and graph
  utils.py
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
