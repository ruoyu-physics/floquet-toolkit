"""Built-in rotating-frame Dirac model."""

from __future__ import annotations

import numpy as np

from ..config import DriveParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils import vector_potential_components
from .base_model import BuiltinDrivenModelSpec
from .dirac import DiracParameters, SIGMA_X, SIGMA_Y, SIGMA_Z


class RotatingFrameDiracModel(BuiltinDrivenModelSpec):
    """User-specified rotating-frame version of the Dirac model."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = self.model_params.mass
        self.vf = self.model_params.vf
        self.e_fermi = self.model_params.e_fermi

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
        ax, ay = vector_potential_components(
            t,
            self.omega,
            self.AL,
            self.AR,
            self.polarization_axis,
        )
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
    model_params: DiracParameters,
    drive_params: DriveParameters,
):
    """Create a built-in rotating-frame Dirac model."""
    return RotatingFrameDiracModel(
        model_params,
        drive_params,
    ).to_driven_hamiltonian()
