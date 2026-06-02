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

    def H_periodic(self, t, kx, ky) -> np.ndarray:
        """Return the time-periodic rotating-frame contribution.

        Broadcasts over an array of times: scalar ``t`` yields a ``(2, 2)``
        matrix, an array of shape ``S`` yields ``S + (2, 2)``. The momentum is
        shifted by the vector potential, rotated by ``omega t``, then mapped
        onto the Pauli matrices.
        """
        t = np.asarray(t, dtype=float)
        cos_t = np.cos(self.omega * t)
        sin_t = np.sin(self.omega * t)
        ax, ay = vector_potential_components(
            t,
            self.omega,
            self.AL,
            self.AR,
            self.polarization_axis,
        )
        shifted_kx = kx - self.e_charge * ax / self.hbar
        shifted_ky = ky - self.e_charge * ay / self.hbar
        rotated_kx = cos_t * shifted_kx - sin_t * shifted_ky
        rotated_ky = sin_t * shifted_kx + cos_t * shifted_ky
        rotated_kx = np.asarray(rotated_kx)[..., None, None]
        rotated_ky = np.asarray(rotated_ky)[..., None, None]
        return self.hbar * self.vf * (rotated_kx * SIGMA_X + rotated_ky * SIGMA_Y)

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full rotating-frame Hamiltonian."""
        return self.H_static(kx, ky) + self.H_periodic(t, kx, ky)

    def to_driven_hamiltonian(self) -> DrivenBlochHamiltonian:
        """Build the solver-facing driven Hamiltonian container."""
        return DrivenBlochHamiltonian(
            H_t=self.H_t,
            omega=self.omega,
            H_static=self.H_static,
            supports_vectorized_time=True,
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
