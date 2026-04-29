"""Built-in rotating-frame Dirac model."""

from __future__ import annotations

import numpy as np

from ..config import DriveParameters, PhysicsParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .base import BuiltinDrivenModelSpec
from .dirac import SIGMA_X, SIGMA_Y, SIGMA_Z


class RotatingFrameDiracModel(BuiltinDrivenModelSpec):
    """User-specified rotating-frame version of the Dirac model."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = self.physics_params.mass
        self.vf = self.physics_params.vf
        self.AL = self.drive_params.AL
        self.AR = self.drive_params.AR

    def H_static(self, kx: float, ky: float) -> np.ndarray:
        """Return the static rotating-frame Dirac Hamiltonian."""
        return (self.mass - self.hbar * self.omega / 2.0) * SIGMA_Z

    def H_periodic(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the time-periodic rotating-frame contribution."""
        rotation_matrix = np.array(
            [
                [np.cos(self.omega * t), -np.sin(self.omega * t)],
                [np.sin(self.omega * t), np.cos(self.omega * t)],
            ]
        )
        k_vector = np.array([kx, ky], dtype=float)
        ax = (self.AL + self.AR) / np.sqrt(2.0) * np.cos(self.omega * t)
        ay = (self.AL - self.AR) / np.sqrt(2.0) * np.sin(self.omega * t)
        a_vector = np.array([ax, ay], dtype=float)

        rotated_k = rotation_matrix @ (k_vector - self.e_charge * a_vector / self.hbar)
        return self.hbar * self.vf * (
            rotated_k[0] * SIGMA_X + rotated_k[1] * SIGMA_Y
        )

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full rotating-frame Hamiltonian."""
        return self.H_static(kx, ky) + self.H_periodic(t, kx, ky)

    def to_driven_hamiltonian(self) -> DrivenBlochHamiltonian:
        """Build the solver-facing driven Hamiltonian container."""
        return DrivenBlochHamiltonian(
            H_t=self.H_t,
            omega=self.omega,
            H_static=self.H_static,
            units=self.units,
        )


def rotating_frame_dirac_model(
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a built-in rotating-frame Dirac model."""
    return RotatingFrameDiracModel(
        physics_params,
        drive_params,
    ).to_driven_hamiltonian()

