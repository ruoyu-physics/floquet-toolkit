"""Berry-curvature calculators for static, Floquet, and HFE states."""

import numpy as np
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..builders import FloquetBuilder
from ..builders import HFEBuilder
from .floquet_state_provider import FloquetStateProvider
from functools import partial
from typing import Callable
from .floquet_perturbation_calculator import FloquetPerturbationCalculator


class FloquetCurvatureCalculator:
    """Compute Berry curvatures from several Floquet-related state sources.

    The calculator owns observable-level logic. Matrix construction is delegated
    to builders and band/state selection is delegated to ``FloquetStateProvider``.
    """
    
    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Initialize the calculator with a driven model and numerical parameters."""
        self.driven_hamiltonian = driven_hamiltonian
        self.perturbation_calculator = FloquetPerturbationCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)

        self.Ht = driven_hamiltonian.Ht
        self.omega = driven_hamiltonian.omega
        self.dimension = driven_hamiltonian.dimension
        self.period = 2.0 * np.pi / self.omega

        self.floquet_params = floquet_params
        self.n_trunc = floquet_params.n_trunc
        self.n_harmonics = floquet_params.n_harmonics
        self.n_time = floquet_params.n_time
        self.n_blocks = floquet_params.n_blocks

        self.hbar = driven_hamiltonian.units.hbar
        
    def _compute_link_variable(self, state_1, state_2, atol=1e-12):
        """Compute the normalized link variable between two states."""
        if state_1.shape != state_2.shape:
            raise ValueError("State shapes do not match for link variable computation.")
        
        if state_1.ndim == 1:
            # Static case: simple inner product
            increment = np.vdot(state_1, state_2)
        elif state_1.ndim == 2:
            # Floquet case: time-resolved link variable
            increment = np.einsum("ij,ij->i", np.conj(state_1), state_2)
        else:
            raise ValueError("State arrays must be 1D (static) or 2D (Floquet).")
        if np.any(abs(increment) < atol):
            raise ValueError(
                "Near-zero link variable encountered; check state continuity "
                "and dk choice."
            )
        return increment / abs(increment)
    
    def _compute_berry_curvature(
        self,
        bloch_state: Callable,
        kx: float,
        ky: float,
        band="conduction",
        dk=1e5,
    ):
        """Compute Berry curvature using the Fukui loop method."""
        v1 = bloch_state(kx - dk / 2, ky - dk / 2, band=band)
        v2 = bloch_state(kx + dk / 2, ky - dk / 2, band=band)
        v3 = bloch_state(kx - dk / 2, ky + dk / 2, band=band)
        v4 = bloch_state(kx + dk / 2, ky + dk / 2, band=band)

        Ux = self._compute_link_variable(v1, v2)
        Uy_dkx = self._compute_link_variable(v2, v4)
        Ux_dky = self._compute_link_variable(v3, v4)
        Uy = self._compute_link_variable(v1, v3)

        curvature = -np.angle(Ux * Uy_dkx / Ux_dky / Uy) / (dk) ** 2
        return curvature

    def compute_instantaneous_berry_curvature(
        self,
        time,
        kx: float,
        ky: float,
        band="conduction",
        dk=1e5,
    ):
        """Compute time-resolved Berry curvature from exact Floquet states.

        Args:
            time: Time or time grid at which to compute the curvature.
            kx: Momentum along x.
            ky: Momentum along y.
            band: Target band label or integer index.
            dk: Half-width of the Fukui plaquette for numerical evaluation.
        """

        def floquet_state(kx, ky, band="conduction"):
            """Select and reconstruct the time-dependent Floquet state."""
            _, __, f_state = self.state_provider.select_floquet_state(kx, ky, band=band)
            return self.state_provider.reconstruct_floquet_state(f_state, time=time)

        return self._compute_berry_curvature(floquet_state, kx, ky, dk=dk, band=band)

    def compute_perturbed_state_berry_curvature(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk=1e5,
    ):
        """Compute Berry curvature from the first-order corrected Floquet state."""

        def perturbed_state(kx, ky, band="conduction"):
            _, perturbed_state = self.perturbation_calculator.compute_perturbed_state(
                kx,
                ky,
                band=band,
                order=2,
            )
            return self.state_provider.reconstruct_floquet_state(perturbed_state, time=time)

        return self._compute_berry_curvature(perturbed_state, kx, ky, band=band, dk=dk)

    def compute_static_berry_curvature(
        self,
        kx: float,
        ky: float,
        band="conduction",
        dk=1e5,
        method="auto",
    ):
        """Compute Berry curvature for the model's static Hamiltonian."""
        if method == "analytic":
            if self.driven_hamiltonian.analytic_static_berry_curvature is None:
                raise ValueError("Analytic static Berry curvature is not available for this model.")
            return self.driven_hamiltonian.analytic_static_berry_curvature(
                kx,
                ky,
                band=band,
            )

        if method == "auto" and self.driven_hamiltonian.analytic_static_berry_curvature is not None:
            return self.driven_hamiltonian.analytic_static_berry_curvature(
                kx,
                ky,
                band=band,
            )

        def static_state(kx, ky, band="conduction"):
            energy, states = self.state_provider.diagonalize_static_hamiltonian(kx, ky)
            band_index = self.state_provider.resolve_band_index(energy, band)
            return states[:, band_index]

        return self._compute_berry_curvature(static_state, kx, ky, band=band, dk=dk)

    def compute_hfe_berry_curvature(
        self,
        kx,
        ky,
        band="conduction",
        dk=1e5,
        order: int = 2,
    ):
        """Compute Berry curvature from the high-frequency effective Hamiltonian.

        The effective Hamiltonian is treated as a static Hamiltonian at each
            plaquette corner, so this does not include micromotion dressing.
        """
        def eff_state(kx, ky, band="conduction"):
            floquet_builder = FloquetBuilder(
                partial(self.Ht, kx=kx, ky=ky),
                self.omega,
                self.hbar,
                self.floquet_params,
            )
            hfe_builder = HFEBuilder(floquet_builder)
            Heff = hfe_builder.compute_hfe_hamiltonian(order=order)
            eigvals, eigvecs = np.linalg.eigh(Heff)
            band_index = self.state_provider.resolve_band_index(eigvals, band)
            return eigvecs[:, band_index]

        return self._compute_berry_curvature(eff_state, kx, ky, band=band, dk=dk)
