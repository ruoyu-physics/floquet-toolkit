"""Built-in graphene tight-binding model."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..config import DriveParameters, GRAPHENE_BOND_LENGTH, MEV_TO_J, UnitConvention
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils import vector_potential_components
from .base_model import BuiltinDrivenModelSpec


@dataclass(frozen=True)
class GrapheneParameters:
    """Physical parameters for the built-in graphene tight-binding model."""

    units: UnitConvention = field(default_factory=UnitConvention.SI_UNITS)
    vf: float = 1.0e6
    lattice_spacing: float = GRAPHENE_BOND_LENGTH
    e_fermi: float = 30.0 * MEV_TO_J


class GrapheneModel(BuiltinDrivenModelSpec):
    """Nearest-neighbor graphene tight-binding model with circular drive."""

    def __post_init__(self):
        super().__post_init__()
        self.mass = 0.0
        self.vf = self.model_params.vf
        self.e_fermi = self.model_params.e_fermi
        self.lattice_spacing = self.model_params.lattice_spacing

        self.hopping = 2.0 * self.hbar * self.vf / (3.0 * self.lattice_spacing)
        delta_1 = self.lattice_spacing * np.array([0.0, 1.0])
        delta_2 = self.lattice_spacing * np.array([np.sqrt(3.0) / 2.0, -0.5])
        delta_3 = self.lattice_spacing * np.array([-np.sqrt(3.0) / 2.0, -0.5])
        self.neighbor_vectors = np.array([delta_1, delta_2, delta_3])

    def structure_factor(self, kx, ky):
        """Return the nearest-neighbor graphene structure factor.

        Broadcasts over array-valued momenta: scalar ``(kx, ky)`` returns a
        scalar, arrays of shape ``S`` return shape ``S``.
        """
        k_vector = np.stack(
            [np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)],
            axis=-1,
        )  # (..., 2)
        phases = np.exp(1j * (k_vector @ self.neighbor_vectors.T))  # (..., 3)
        return np.sum(phases, axis=-1)

    def H_static(self, kx, ky) -> np.ndarray:
        """Return the static graphene Bloch Hamiltonian.

        Scalar ``(kx, ky)`` yields a ``(2, 2)`` matrix; arrays of shape ``S``
        yield ``S + (2, 2)``.
        """
        gamma_k = np.asarray(self.structure_factor(kx, ky), dtype=complex)
        hamiltonian = np.zeros(gamma_k.shape + (2, 2), dtype=complex)
        hamiltonian[..., 0, 0] = self.mass
        hamiltonian[..., 1, 1] = -self.mass
        hamiltonian[..., 0, 1] = -self.hopping * gamma_k
        hamiltonian[..., 1, 0] = -self.hopping * np.conj(gamma_k)
        return hamiltonian

    def H_t(self, t: float, kx: float, ky: float) -> np.ndarray:
        """Return the full time-dependent graphene Hamiltonian with k - qA/hbar."""
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

    def to_driven_hamiltonian(self) -> DrivenBlochHamiltonian:
        """Build the solver-facing driven Hamiltonian container."""
        return DrivenBlochHamiltonian(
            H_t=self.H_t,
            omega=self.omega,
            H_static=self.H_static,
            analytic_static_berry_curvature=None,
            supports_vectorized_time=True,
            units=self.units,
        )


def driven_graphene_model(
    model_params: GrapheneParameters,
    drive_params: DriveParameters,
):
    """Create a built-in driven graphene model."""
    return GrapheneModel(model_params, drive_params).to_driven_hamiltonian()
