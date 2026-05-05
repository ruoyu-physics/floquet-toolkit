"""Built-in massive Dirac model."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import DriveParameters, MEV_TO_J, UnitConvention
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils import vector_potential_components
from .base_model import BuiltinDrivenModelSpec

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass(frozen=True)
class DiracParameters:
    """Physical parameters for built-in Dirac-family continuum models."""

    units: UnitConvention = field(default_factory=UnitConvention.SI_UNITS)
    vf: float = 1.0e6
    mass: float = -40.0 * MEV_TO_J
    e_fermi: float = 65.0 * MEV_TO_J


class DiracModel(BuiltinDrivenModelSpec):
    """Lab-frame massive Dirac model with circular drive components."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = self.model_params.mass
        self.vf = self.model_params.vf
        self.e_fermi = self.model_params.e_fermi

    def H_static(self, kx: float, ky: float) -> np.ndarray:
        """Return the static Dirac Hamiltonian."""
        return self.mass * SIGMA_Z + self.hbar * self.vf * (
            kx * SIGMA_X + ky * SIGMA_Y
        )

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full time-dependent Dirac Hamiltonian with k - qA/hbar."""
        ax, ay = vector_potential_components(
            t,
            self.omega,
            self.AL,
            self.AR,
            self.polarization_axis,
        )
        shifted_kx = kx - self.e_charge / self.hbar * ax
        shifted_ky = ky - self.e_charge / self.hbar * ay
        return self.H_static(shifted_kx, shifted_ky)

    def analytic_static_berry_curvature(self, kx, ky, band="conduction"):
        """Return the analytic static Berry curvature."""
        prefactor = 0.5 * self.mass * (self.hbar * self.vf) ** 2
        denominator = (
            (self.hbar * self.vf) ** 2 * (kx**2 + ky**2) + self.mass**2
        ) ** (3 / 2)
        if denominator == 0.0:
            raise ValueError("Berry curvature is undefined at the gapless Dirac point.")
        if band in ("conduction", 1):
            return -prefactor / denominator
        if band in ("valence", 0):
            return prefactor / denominator
        raise ValueError("Band must be 'conduction', 'valence', 0, or 1")

    def to_driven_hamiltonian(self) -> DrivenBlochHamiltonian:
        """Build the solver-facing driven Hamiltonian container."""
        return DrivenBlochHamiltonian(
            H_t=self.H_t,
            omega=self.omega,
            H_static=self.H_static,
            analytic_static_berry_curvature=self.analytic_static_berry_curvature,
            units=self.units,
        )


def driven_dirac_model(
    model_params: DiracParameters,
    drive_params: DriveParameters,
):
    """Create a built-in driven Dirac model."""
    return DiracModel(model_params, drive_params).to_driven_hamiltonian()
