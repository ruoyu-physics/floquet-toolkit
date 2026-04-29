"""Built-in massive Dirac model."""

from __future__ import annotations

import numpy as np

from ..config import DriveParameters, PhysicsParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .base import BuiltinDrivenModelSpec

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)


class DiracModel(BuiltinDrivenModelSpec):
    """Lab-frame massive Dirac model with circular drive components."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = self.physics_params.mass
        self.vf = self.physics_params.vf
        self.AL = self.drive_params.AL
        self.AR = self.drive_params.AR

    def H_static(self, kx: float, ky: float) -> np.ndarray:
        """Return the static Dirac Hamiltonian."""
        return self.mass * SIGMA_Z + self.hbar * self.vf * (
            kx * SIGMA_X + ky * SIGMA_Y
        )

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full time-dependent Dirac Hamiltonian."""
        ax = (self.AL + self.AR) / np.sqrt(2.0) * np.cos(self.omega * t)
        ay = (self.AL - self.AR) / np.sqrt(2.0) * np.sin(self.omega * t)
        shifted_kx = kx - self.e_charge / self.hbar * ax
        shifted_ky = ky - self.e_charge / self.hbar * ay
        return self.H_static(shifted_kx, shifted_ky)

    def analytic_static_berry_curvature(self, kx, ky, band="conduction"):
        """Return the analytic static Berry curvature."""
        prefactor = 0.5 * self.mass * (self.hbar * self.vf) ** 2
        denominator = (
            (self.hbar * self.vf) ** 2 * (kx**2 + ky**2) + self.mass**2
        ) ** (3 / 2)
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
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a built-in driven Dirac model."""
    return DiracModel(physics_params, drive_params).to_driven_hamiltonian()

