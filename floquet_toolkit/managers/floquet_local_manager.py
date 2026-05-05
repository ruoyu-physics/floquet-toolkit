"""Manager for local Floquet observables and state-resolved quantities."""

from functools import partial
import numpy as np

from ..builders import FloquetBuilder
from ..calculators import (
    FloquetCurvatureCalculator,
    FloquetPerturbationCalculator,
    FloquetSpectrumCalculator,
    FloquetStateProvider,
    FloquetVelocityCalculator,
)
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian


class FloquetLocalManager:
    """Facade for local-in-k Floquet states, spectra, curvature, and velocity."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.hbar = driven_hamiltonian.units.hbar

        self.floquet_curvature_calculator = FloquetCurvatureCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.floquet_velocity_calculator = FloquetVelocityCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.floquet_perturbation_calculator = FloquetPerturbationCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.floquet_spectrum_calculator = FloquetSpectrumCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)

    def diagonalize_floquet_hamiltonian(self, kx, ky):
        """Return exact truncated Floquet eigenvalues and eigenvectors."""
        builder = FloquetBuilder(
            partial(self.floquet_curvature_calculator.Ht, kx=kx, ky=ky),
            self.floquet_curvature_calculator.omega,
            self.hbar,
            self.floquet_params,
        )
        floquet_hamiltonian = builder.compute_floquet_hamiltonian()
        quasi_energy, floquet_states = np.linalg.eigh(floquet_hamiltonian)
        return quasi_energy, floquet_states

    def select_floquet_state(self, kx, ky, band: str = "conduction", mode: str = "overlap"):
        """Return the selected Floquet eigenstate at one momentum."""
        return self.state_provider.select_floquet_state(kx, ky, band=band, mode=mode)

    def compute_static_berry_curvature(
        self,
        kx: float,
        ky: float,
        band="conduction",
        dk: float = 1e5,
        method: str = "auto",
    ):
        return self.floquet_curvature_calculator.compute_static_berry_curvature(
            kx,
            ky,
            band=band,
            dk=dk,
            method=method,
        )

    def compute_instantaneous_berry_curvature(
        self,
        time: float | np.ndarray,
        kx: float,
        ky: float,
        band="conduction",
        dk: float = 1e5,
    ):
        return self.floquet_curvature_calculator.compute_instantaneous_berry_curvature(
            time,
            kx,
            ky,
            band=band,
            dk=dk,
        )

    def compute_hfe_berry_curvature(
        self,
        kx: float,
        ky: float,
        band="conduction",
        dk: float = 1e5,
        order: int = 2,
    ):
        return self.floquet_curvature_calculator.compute_hfe_berry_curvature(
            kx,
            ky,
            band=band,
            dk=dk,
            order=order,
        )

    def compute_floquet_velocity(
        self,
        time,
        kx,
        ky,
        band="conduction",
        mode: str = "overlap",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        return self.floquet_velocity_calculator.compute_floquet_velocity(
            time,
            kx,
            ky,
            band=band,
            mode=mode,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_perturbed_velocity(
        self,
        time,
        kx,
        ky,
        band="conduction",
        order: int = 1,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        return self.floquet_velocity_calculator.compute_perturbed_velocity(
            time,
            kx,
            ky,
            band=band,
            order=order,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_static_velocity(
        self,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        return self.floquet_velocity_calculator.compute_static_velocity(
            kx,
            ky,
            band=band,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_adiabatic_velocity(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        return self.floquet_velocity_calculator.compute_adiabatic_velocity(
            time,
            kx,
            ky,
            band=band,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_hfe_velocity(
        self,
        kx,
        ky,
        band="conduction",
        order: int = 2,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        return self.floquet_velocity_calculator.compute_hfe_velocity(
            kx,
            ky,
            band=band,
            order=order,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_perturbed_state(
        self,
        kx: float,
        ky: float,
        time: float | np.ndarray,
        band="conduction",
        order: int = 2,
    ):
        _, perturbed_state = self.floquet_perturbation_calculator.compute_perturbed_state(
            kx,
            ky,
            band=band,
            order=order,
        )
        return self.state_provider.reconstruct_floquet_state(perturbed_state, time=time)

    def compute_perturbed_state_berry_curvature(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk=1e5,
    ):
        return self.floquet_curvature_calculator.compute_perturbed_state_berry_curvature(
            time,
            kx,
            ky,
            band=band,
            dk=dk,
        )

    def compute_quasi_energies(self, kx, ky, band="conduction", fold_to_zone: bool = True):
        return self.floquet_spectrum_calculator.compute_quasi_energies(
            kx,
            ky,
            band=band,
            fold_to_zone=fold_to_zone,
        )

    def compute_floquet_spectrum(
        self,
        kx_values,
        ky_values,
        band="conduction",
        fold_to_zone: bool = False,
    ):
        return self.floquet_spectrum_calculator.compute_floquet_spectrum(
            kx_values,
            ky_values,
            band=band,
            fold_to_zone=fold_to_zone,
        )
