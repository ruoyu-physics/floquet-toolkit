"""Shared configuration for Floquet Dirac calculations.

This module centralizes physical constants, model parameters, and numerical
settings so both the analytic and numerical Floquet codes use the same values.
"""

from dataclasses import dataclass
import numpy as np


HBAR = 1.054571817e-34
"""Reduced Planck constant in joule-seconds."""

E_CHARGE = 1.60217663e-19
"""Elementary charge in coulombs."""

EPSILON0 = 8.85418782e-12
"""Vacuum permittivity in C^2 / (J m)."""

MEV_TO_J = 1.60218e-22
"""Conversion factor from meV to joules."""


@dataclass(frozen=True)
class PhysicsParameters:
    """Material and static model parameters.

    Attributes:
        vf: Fermi velocity in m/s.
        mass: Dirac mass gap in joules.
        e_fermi: Fermi energy in joules.
    """

    vf: float = 1.0e6
    mass: float = 40.0 * MEV_TO_J
    e_fermi: float = 65.0 * MEV_TO_J

@dataclass(frozen=True)
class DriveParameters:
    """External drive parameters for time-periodic model construction.

    Attributes:
        omega: Drive angular frequency in rad/s.
        AL: Left-circular drive component amplitude in meters.
        AR: Right-circular drive component amplitude in meters.
    """

    omega: float = 17.0 * MEV_TO_J / HBAR
    AL: float = 3.0e-9
    AR: float = 0.0

    @property
    def period(self) -> float:
        """Return the drive period T = 2π / ω in seconds."""
        return 2.0 * np.pi / self.omega


@dataclass(frozen=True)
class FloquetParameters:
    """Numerical parameters for Floquet calculations.

    Attributes:
        n_trunc: Floquet sideband cutoff, keeping m in [-n_trunc, n_trunc].
        n_harmonics: Number of Fourier harmonics retained on each side.
        n_time: Number of time samples used for numerical Fourier integration
            or time-grid-based observables.
        dk: Momentum step used in plaquette Berry-curvature calculations.
        band: Default band label for state selection.
    """

    n_trunc: int = 4
    n_harmonics: int = 1
    n_time: int = 64
    dk: float = 1e5

    @property
    def n_blocks(self) -> int:
        """Return the total number of Floquet blocks, 2*n_trunc + 1."""
        return 2 * self.n_trunc + 1


@dataclass(frozen=True)
class PlotParameters:
    """Sampling settings used for plotting and scans.

    Attributes:
        n_time_plot: Number of time points for time-dependent plots.
        n_kx: Number of momentum samples along kx.
        n_ky: Number of momentum samples along ky.
    """

    n_time_plot: int = 100
    n_kx: int = 91
    n_ky: int = 21


PHYSICS_PARAMS = PhysicsParameters()
"""Default physical parameter set."""

DRIVE_PARAMS = DriveParameters()
"""Default drive parameter set."""

FLOQUET_PARAMS = FloquetParameters()
"""Default Floquet numerical parameter set."""

PLOTS = PlotParameters()
"""Default plotting and scan parameter set."""
