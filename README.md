# Floquet Toolkit

`floquet_toolkit` is a small Python package for numerical Floquet calculations in driven two-band Bloch models, with an emphasis on driven Dirac Hamiltonians.

It currently provides tools to:
- build truncated Floquet Hamiltonians from time-periodic models
- diagonalize Floquet Hamiltonians and reconstruct time-dependent Floquet states
- compute quasienergy spectra
- compute Berry curvature from static, instantaneous Floquet, perturbative Floquet, and high-frequency effective descriptions
- compute current observables for several state constructions
- work with built-in driven Dirac and graphene-style model factories

## Project Layout

```text
floquet-toolkit/
  floquet_toolkit/
    builders/
    calculators/
    core/
    config.py
    floquet_manager.py
    builtin_models.py
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

## Quick Start

```python
import numpy as np

from floquet_toolkit import FloquetManager
from floquet_toolkit.config import DriveParameters, FloquetParameters, PhysicsParameters
from floquet_toolkit.builtin_models import driven_dirac_model

physics_params = PhysicsParameters(
    vf=1.0e6,
    mass=40.0e-3 * 1.60218e-19,
)

drive_params = DriveParameters(
    omega=17.0 * 1.60218e-22 / 1.054571817e-34,
    AL=3.0e-9,
    AR=0.0,
)

floquet_params = FloquetParameters(
    n_trunc=5,
    n_harmonics=2,
    n_time=61,
)

model = driven_dirac_model(physics_params, drive_params)
manager = FloquetManager(model, floquet_params)

time = np.linspace(0.0, drive_params.period, floquet_params.n_time, endpoint=False)
curvature = manager.compute_instantaneous_berry_curvature(
    time=time,
    kx=0.0,
    ky=0.0,
    band="conduction",
    dk=1.0e5,
)

print(curvature.shape)
```

## Main Components

### `FloquetManager`

`FloquetManager` is the main user-facing entry point. It wraps the lower-level builders and calculators and provides methods such as:
- `diagonalize_floquet_hamiltonian`
- `select_floquet_state`
- `compute_static_berry_curvature`
- `compute_instantaneous_berry_curvature`
- `compute_perturbed_state_berry_curvature`
- `compute_hfe_berry_curvature`
- `compute_floquet_current`
- `compute_static_current`
- `compute_instantaneous_current`
- `compute_hfe_current`

### Built-in Models

The package currently includes:
- `driven_dirac_model(...)`
- `driven_graphene_model(...)`
- `rotating_frame_dirac_model(...)`

These factories return a `DrivenBlochHamiltonian` object that can be passed into `FloquetManager`.

### Configuration Dataclasses

Shared configuration lives in `floquet_toolkit.config`:
- `PhysicsParameters`
- `DriveParameters`
- `FloquetParameters`

## Running Tests

Run the current test suite with:

```bash
pytest -q tests/test_floquet_builder.py tests/test_curvature_convergence.py
```

The current tests cover:
- Floquet-builder convergence with respect to `n_trunc`, `n_harmonics`, and `n_time`
- matrix-size and eigensystem consistency
- `dk` convergence for instantaneous Berry curvature
- `dk` convergence for numerically computed static Berry curvature

## Examples

The `examples/` folder contains lightweight plotting examples for the built-in models. In particular, `examples/plot_spectra.py` and `examples/plot_spectra.ipynb` provide separate Dirac and graphene Floquet spectrum visualizations.

## Status

This project is currently best described as a research code package under active cleanup. The core package is installable, example plots are included, and a basic automated test suite is in place, but the public API may still evolve.

