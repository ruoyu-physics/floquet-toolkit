# Floquet Toolkit

`floquet_toolkit` is a reusable Python framework for numerical Floquet calculations in driven two-band Bloch systems, with built-in examples centered on driven Dirac and graphene-style models.

It currently provides tools to:
- build truncated extended space Floquet Hamiltonians from time-periodic models
- diagonalize Floquet Hamiltonians and reconstruct time-dependent Floquet states
- compute quasienergy spectra
- compute Berry curvature from static, Floquet, perturbative Floquet, and high-frequency effective descriptions
- compute current observables for several state constructions
- work with built-in driven Dirac and graphene-style model factories

## Design Goal

The package is designed to avoid rewriting model-specific or observable-specific code for each new study. New driven two-band systems can be introduced through the `DrivenBlochHamiltonian` abstraction, while new observables can be built on top of the existing builder and calculator classes. In practice, this means users can reuse the same numerical machinery across different models and extend the toolkit with new calculators instead of rewiring the full codebase.

## Project Layout

```text
floquet-toolkit/
  floquet_toolkit/
    builtin_models/
    builders/
    calculators/
    core/
    config.py
    managers/
  docs/
  benchmarks/
  tests/
  examples/
  pyproject.toml
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Install the package in editable mode:

```bash
pip install -e .
```

Install optional plotting and test dependencies:

```bash
pip install -e ".[plots,dev]"
```

Dependency groups:
- `plots`: installs `matplotlib`
- `dev`: installs `pytest`

## Main Components

### `FloquetLocalManager` and `FloquetTransportManager`

`FloquetLocalManager` handles local-in-momentum observables such as spectra,
Floquet states, Berry curvature, and velocities. `FloquetTransportManager`
collects loop/integrated transport-style observables such as Berry phases,
Berry-curvature integrals, and current integrals over explicit k-space
quadratures. `FloquetManager` remains available as a lightweight compatibility
wrapper that exposes `.local` and `.transport`.

Useful methods are grouped roughly as follows:

- State and spectrum
  - `diagonalize_floquet_hamiltonian`
  - `select_floquet_state`
  - `compute_floquet_spectrum`

- Berry curvature
  - `compute_static_berry_curvature`
  - `compute_instantaneous_berry_curvature`
  - `compute_perturbed_state_berry_curvature`

- Velocity observables
  - `compute_floquet_velocity`
  - `compute_static_velocity`
  - `compute_adiabatic_velocity`

- Integrated observables
  - `integrate_current`
  - `integrate_adaptive_current`
  - `integrate_berry_curvature_on_grid`

`KQuadrature` is the standard way to describe fixed k-space integration grids
for current calculations. Build one with `KQuadrature.cartesian(...)` or
`KQuadrature.polar(...)`, then pass it to `FloquetTransportManager.integrate_current(...)`.
Additional helper methods are also available for perturbative and high-frequency calculations, but the methods above are the most common starting points for users of the package.

### Built-in Models

The package currently includes:
- `DiracModel`
- `GrapheneModel`
- `RotatingFrameDiracModel`

Each built-in model class can be converted into a solver-facing
`DrivenBlochHamiltonian` with `.to_driven_hamiltonian()`. Convenience factory
helpers are also available:
- `driven_dirac_model(...)`
- `driven_graphene_model(...)`
- `rotating_frame_dirac_model(...)`

Users are not limited to the built-in models. You can define your own time-periodic two-band Hamiltonian with the `DrivenBlochHamiltonian` class and then reuse the same Floquet builders, calculators, and `FloquetManager` interface.

### Configuration Dataclasses

Shared configuration lives in `floquet_toolkit.config`:
- `UnitConvention`
- `DriveParameters`
- `FloquetParameters`

Model-specific physical parameters live in `floquet_toolkit.builtin_models`:
- `DiracParameters`
- `GrapheneParameters`

## Examples

The `examples/` folder contains lightweight plotting examples for the built-in models. Current examples include:

- `examples/plot_spectra.py` and `examples/plot_spectra.ipynb` for Dirac and graphene Floquet spectrum visualizations
- `examples/plot_dirac_curvature_vs_amplitude.py` and `examples/plot_dirac_curvature_vs_amplitude.ipynb` for scanning the time-averaged Dirac-model curvature at `k=(0,0)` versus circular-drive amplitude

Performance notes live in `docs/PERFORMANCE.md`, with runnable benchmark and profiling harnesses in `benchmarks/`.

## Running Tests

Run the current test suite with:

```bash
python3 -m pytest -q
```

The current tests cover:
- Floquet-builder convergence with respect to `n_trunc`, `n_harmonics`, and `n_time`
- matrix-size and eigensystem consistency
- `dk` convergence for instantaneous Berry curvature
- `dk` convergence for numerically computed static Berry curvature
- built-in model class construction and manager/cache behavior
- polar integration and `KQuadrature` current-integration behavior


## Quick Start

```python
from floquet_toolkit import DiracModel, FloquetLocalManager, UnitConvention
from floquet_toolkit.builtin_models import DiracParameters
from floquet_toolkit.config import (
    DriveParameters,
    FloquetParameters,
    MEV_TO_J,
)

si_units = UnitConvention.SI_UNITS()

dirac_params = DiracParameters(
    units=si_units,
    vf=1.0e6,
    mass=-40.0 * MEV_TO_J,
)

drive_params = DriveParameters(
    units=si_units,
    AL=3.0e-9,
    AR=0.0,
)

floquet_params = FloquetParameters(
    n_trunc=5,
    n_harmonics=2,
    n_time=61,
)

model = DiracModel(dirac_params, drive_params).to_driven_hamiltonian()
manager = FloquetLocalManager(model, floquet_params)

time = floquet_params.time_grid(model.period)
curvature = manager.compute_instantaneous_berry_curvature(
    time=time,
    kx=0.0,
    ky=0.0,
    band="conduction",
    dk=1.0e5,
)

print(curvature.shape)
```
