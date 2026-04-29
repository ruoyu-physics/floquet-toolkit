"""Shared configuration for Floquet calculations.

This module groups together:
- physical constants used by the built-in SI-based models
- unit-convention metadata
- convenience parameter dataclasses for built-in models
- numerical solver settings
"""

from dataclasses import dataclass, field
import numpy as np


HBAR = 1.054571817e-34
"""Reduced Planck constant in joule-seconds."""

E_CHARGE = 1.60217663e-19
"""Elementary charge in coulombs."""

EPSILON0 = 8.85418782e-12
"""Vacuum permittivity in C^2 / (J m)."""

MEV_TO_J = 1.60218e-22
"""Conversion factor from meV to joules."""

GRAPHENE_BOND_LENGTH = 1.42e-10
"""Nearest-neighbor carbon-carbon bond length in meters."""

@dataclass(frozen=True)
class UnitConvention:
    """Describe how model inputs and outputs should be interpreted.

    The numerical solvers operate on plain floats; this dataclass records the
    convention attached to those numbers, including the values of
    constants such as ``hbar`` and ``e_charge`` and human-readable unit labels.

    The built-in ``UNITLESS()`` preset is not a fully generic dimensionless
    system. It represents one specific nondimensionalization commonly used for
    Dirac- or lattice-style models:
    - energy in units of ``delta``
    - length in units of ``a``
    - momentum in units of ``1/a``
    - time in units of ``1/delta`` with ``hbar = 1``
    - charge in units of ``e``

    Attributes:
        name: Name of the convention.
        hbar: Numerical value of the reduced Planck constant in the chosen
            energy and time units.
        e_charge: Numerical value of the elementary charge in the chosen
            charge unit.
        dimensionless: Whether quantities are represented as rescaled,
            dimensionless numbers relative to named reference scales rather
            than as direct SI-valued inputs.
    """
    name: str = "si"
    hbar: float = HBAR
    e_charge: float = E_CHARGE
    dimensionless: bool = False

    time_unit: str = "s"
    length_unit: str = "m"
    charge_unit: str = "C"
    energy_unit: str = "J"
    momentum_unit: str = "1/m"
    hbar_unit: str = "J s"
    
    @classmethod
    def SI_UNITS(cls)-> "UnitConvention":
        """Return the standard SI convention."""
        return cls()
    
    @classmethod
    def UNITLESS(cls) -> "UnitConvention":
        """Return the built-in nondimensionalized model convention.

        This preset uses ``hbar = 1`` and ``e = 1`` with labels corresponding
        to a specific reference-scale choice: energy in ``delta``, length in
        ``a``, momentum in ``1/a``, and time in ``1/delta``.
        """
        return cls(
            name="unitless",
            hbar=1.0,
            e_charge=1.0,
            dimensionless=True,
            time_unit="1/delta",
            length_unit="a",
            charge_unit="e",
            energy_unit="delta",
            momentum_unit="1/a",
            hbar_unit="1"
        )

@dataclass(frozen=True)
class PhysicsParameters:
    """Material and static model parameters for built-in models.

    The values are interpreted in the attached ``units`` convention. The
    default numerical values correspond to the SI preset.

    Attributes:
        units: Unit convention used to interpret the remaining fields.
        vf: Characteristic Dirac velocity in the convention's length/time
            units.
        mass: Dirac mass gap in the convention's energy units.
        e_fermi: Fermi energy in the convention's energy units.
        lattice_spacing: Reference nearest-neighbor spacing used by lattice
            built-in models, expressed in the convention's length units.
    """

    units: UnitConvention = field(default_factory=UnitConvention.SI_UNITS)
    vf: float = 1.0e6
    mass: float = -40.0 * MEV_TO_J
    e_fermi: float = 65.0 * MEV_TO_J
    lattice_spacing: float = GRAPHENE_BOND_LENGTH

@dataclass(frozen=True)
class DriveParameters:
    """Drive parameters for built-in model construction.

    The values are interpreted in the attached ``units`` convention. The
    default numerical values correspond to the SI preset.

    Attributes:
        units: Unit convention used to interpret the remaining fields.
        omega: Drive angular frequency in inverse time units.
        AL: Left-circular drive component amplitude in the convention's
            length-like drive units.
        AR: Right-circular drive component amplitude in the convention's
            length-like drive units.
    """

    units: UnitConvention = field(default_factory=UnitConvention.SI_UNITS)
    omega: float = 17.0 * MEV_TO_J / HBAR
    AL: float = 3.0e-9
    AR: float = 0.0

    @property
    def period(self) -> float:
        """Return the drive period ``T = 2π / ω`` in the same time units."""
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

PHYSICS_PARAMS = PhysicsParameters()
"""Default physical parameter set."""

DRIVE_PARAMS = DriveParameters()
"""Default drive parameter set."""

FLOQUET_PARAMS = FloquetParameters()
"""Default Floquet numerical parameter set."""
