"""Built-in graphene tight-binding model."""

from __future__ import annotations

import numpy as np

from ..config import DriveParameters, PhysicsParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .base import BuiltinDrivenModelSpec


class GrapheneModel(BuiltinDrivenModelSpec):
    """Nearest-neighbor graphene tight-binding model with circular drive."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = 0.0
        self.vf = self.physics_params.vf
        self.AL = self.drive_params.AL
        self.AR = self.drive_params.AR
        self.lattice_spacing = self.physics_params.lattice_spacing

        self.hopping = 2.0 * self.hbar * self.vf / (3.0 * self.lattice_spacing)
        delta_1 = self.lattice_spacing * np.array([0.0, 1.0])
        delta_2 = self.lattice_spacing * np.array([np.sqrt(3.0) / 2.0, -0.5])
        delta_3 = self.lattice_spacing * np.array([-np.sqrt(3.0) / 2.0, -0.5])
        self.neighbor_vectors = np.array([delta_1, delta_2, delta_3])

    def structure_factor(self, kx: float, ky: float) -> complex:
        """Return the nearest-neighbor graphene structure factor."""
        k_vector = np.array([kx, ky], dtype=float)
        phases = np.exp(1j * (self.neighbor_vectors @ k_vector))
        return np.sum(phases)

    def H_static(self, kx: float, ky: float) -> np.ndarray:
        """Return the static graphene Bloch Hamiltonian."""
        gamma_k = self.structure_factor(kx, ky)
        return np.array(
            [
                [self.mass, -self.hopping * gamma_k],
                [-self.hopping * np.conj(gamma_k), -self.mass],
            ],
            dtype=complex,
        )

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full time-dependent graphene Hamiltonian."""
        ax = (self.AL + self.AR) / np.sqrt(2.0) * np.cos(self.omega * t)
        ay = (self.AL - self.AR) / np.sqrt(2.0) * np.sin(self.omega * t)
        shifted_kx = kx - self.e_charge / self.hbar * ax
        shifted_ky = ky - self.e_charge / self.hbar * ay
        return self.H_static(shifted_kx, shifted_ky)

    def to_driven_hamiltonian(self) -> DrivenBlochHamiltonian:
        """Build the solver-facing driven Hamiltonian container."""
        return DrivenBlochHamiltonian(
            H_t=self.H_t,
            omega=self.omega,
            H_static=self.H_static,
            analytic_static_berry_curvature=None,
            units=self.units,
        )


def driven_graphene_model(
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a built-in driven graphene model."""
    return GrapheneModel(physics_params, drive_params).to_driven_hamiltonian()

