## v0.1.0 (2026-10-29)

### Feat

- Steady-state (phasor) MNA solver for R, L, C, Z, voltage and current sources, and CCVS
- Mutual coupling between elements, by mutual inductance or mutual impedance
- `Atlas` facade: add elements and connect their terminals without creating nodes or ids, delete elements and wires, solve at a frequency
- Results for every node voltage and every element current
- Test suite, run before every commit by a pre-commit hook
