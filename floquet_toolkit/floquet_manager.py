"""High-level user-facing interface for common Floquet workflows."""

from functools import partial
import numpy as np
from .core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .calculators import (
    FloquetCurrentCalculator,
    FloquetCurvatureCalculator,
    FloquetPerturbationCalculator,
    FloquetSpectrumCalculator,
)
from .calculators import FloquetStateProvider, FloquetBerryPhaseCalculator
from .config import FloquetParameters
from .builders import FloquetBuilder


class FloquetManager:
    """Thin facade over calculators used by scripts and notebooks.

    The manager keeps the public API compact while detailed construction, state
    selection, and observable logic live in specialized classes.
    """

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Create calculators for one driven Hamiltonian and Floquet setup."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.hbar = driven_hamiltonian.units.hbar

        self.floquet_curvature_calculator = FloquetCurvatureCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.floquet_current_calculator = FloquetCurrentCalculator(
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
        self.berry_phase_calculator = FloquetBerryPhaseCalculator(
            driven_hamiltonian,
            floquet_params,
        )

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
        """Return the selected Floquet eigenstate at one momentum.

        Returns:
            Tuple ``(state_index, static_state, floquet_state)`` from the state
            provider's overlap- or quasienergy-based selection rule.
        """
        return self.state_provider.select_floquet_state(kx, ky, band=band, mode=mode)

    def compute_static_berry_curvature(
        self,
        kx: float,
        ky: float,
        band="conduction",
        dk: float = 1e5,
        method: str = "auto",
    ):
        """Compute Berry curvature from the static Hamiltonian."""
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
        """Compute time-resolved Berry curvature from exact Floquet states."""
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
        """Compute Berry curvature from the high-frequency effective Hamiltonian."""
        return self.floquet_curvature_calculator.compute_hfe_berry_curvature(
            kx,
            ky,
            band=band,
            dk=dk,
            order=order,
        )

    def compute_floquet_current(
        self,
        time,
        kx,
        ky,
        band="conduction",
        mode: str = "overlap",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute current from the selected exact Floquet mode."""
        return self.floquet_current_calculator.compute_floquet_current(
            time,
            kx,
            ky,
            band=band,
            mode=mode,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_perturbed_current(
        self,
        time,
        kx,
        ky,
        band="conduction",
        order: int = 1,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute current from the perturbative Floquet state."""
        return self.floquet_current_calculator.compute_perturbed_current(
            time,
            kx,
            ky,
            band=band,
            order=order,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_static_current(
        self,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute current from the static Hamiltonian eigenstate."""
        return self.floquet_current_calculator.compute_static_current(
            kx,
            ky,
            band=band,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_instantaneous_current(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute current from the instantaneous Hamiltonian eigenstate."""
        return self.floquet_current_calculator.compute_instantaneous_current(
            time,
            kx,
            ky,
            band=band,
            dk=dk,
            include_charge=include_charge,
        )

    def compute_hfe_current(
        self,
        kx,
        ky,
        band="conduction",
        order: int = 2,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute current from the high-frequency effective Hamiltonian."""
        return self.floquet_current_calculator.compute_hfe_current(
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
        """Compute a first-order corrected Floquet state at one momentum."""
        perturbation_calculator = FloquetPerturbationCalculator(
            self.floquet_curvature_calculator.driven_hamiltonian,
            self.floquet_params,
        )
        _, perturbed_state = perturbation_calculator.compute_perturbed_state(
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
        """Compute Berry curvature from the first-order corrected Floquet state."""
        return self.floquet_curvature_calculator.compute_perturbed_state_berry_curvature(
            time,
            kx,
            ky,
            band=band,
            dk=dk,
        )

    def compute_quasi_energies(self, kx, ky, band="conduction", fold_to_zone: bool = True):
        """Return quasienergies, central-block weights, and band overlaps."""
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
        """Return quasienergy spectra, central-block weights, and band overlaps."""
        return self.floquet_spectrum_calculator.compute_floquet_spectrum(
            kx_values,
            ky_values,
            band=band,
            fold_to_zone=fold_to_zone,
        )

    def compute_floquet_berry_phase(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=100,
        band="conduction",
    ):
        """Compute the Floquet-mode Berry phase on a circular momentum-space loop."""
        return self.berry_phase_calculator.compute_floquet_berry_phase(
            time,
            k_radius,
            k_center,
            n_points,
            band=band,
        )

    def integrate_berry_curvature_on_grid(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=51,
        band="conduction",
    ):
        """Integrate instantaneous Floquet Berry curvature over a circular region."""
        return self.berry_phase_calculator.integrate_curvature_over_kgrid(
            time,
            k_radius,
            k_center,
            n_points,
            band,
        )
