"""Current-like expectation values for static and Floquet states."""

from functools import partial

import numpy as np

from ..builders import FloquetBuilder, HFEBuilder
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .floquet_perturbation_calculator import FloquetPerturbationCalculator
from .floquet_state_provider import FloquetStateProvider


class FloquetCurrentCalculator:
    """Compute velocity/current expectations from several state sources.

    The returned values are single-particle velocity expectation values by default.
    Set ``include_charge=True`` to multiply by ``-e`` and return charge-current
    expectations instead.
    """

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Initialize from one driven model and one Floquet parameter set."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)
        self.perturbation_calculator = FloquetPerturbationCalculator(
            driven_hamiltonian,
            floquet_params,
        )

        self.Ht = driven_hamiltonian.Ht
        self.H_static = driven_hamiltonian.H_static
        self.omega = driven_hamiltonian.omega
        self.e_charge = driven_hamiltonian.units.e_charge
        self.hbar = driven_hamiltonian.units.hbar

    def _resolve_band_state(self, eigvals, eigvecs, band):
        """Return the eigenvector associated with a band label or index."""
        band_index = self.state_provider.resolve_band_index(eigvals, band)
        return eigvecs[:, band_index]

    def _expectation_value(self, state, operator):
        """Compute <state|operator|state> for one state or a time series."""
        state = np.asarray(state)
        operator = np.asarray(operator)

        if state.ndim == 1 and operator.ndim == 2:
            return np.vdot(state, operator @ state)

        if state.ndim == 2 and operator.ndim == 2:
            return np.einsum("ti,ij,tj->t", np.conj(state), operator, state)

        if state.ndim == 2 and operator.ndim == 3:
            return np.einsum("ti,tij,tj->t", np.conj(state), operator, state)

        raise ValueError(
            "state/operator shapes are incompatible for expectation-value "
            "evaluation"
        )

    def _finalize_current(self, expectation, include_charge):
        """Convert a velocity expectation into a charge current if requested."""
        if include_charge:
            return -self.e_charge * expectation.real
        return expectation.real

    def _velocity_operator(
        self,
        hamiltonian,
        time,
        kx,
        ky,
        axis,
        dk,
    ):
        """Compute the velocity operator (1/hbar) dH/dk by finite difference."""
        if axis == "x":
            delta_kx, delta_ky = dk / 2.0, 0.0
        elif axis == "y":
            delta_kx, delta_ky = 0.0, dk / 2.0
        else:
            raise ValueError("axis must be 'x' or 'y'")

        time = np.asarray(time)
        if time.ndim == 0:
            h_plus = hamiltonian(time, kx + delta_kx, ky + delta_ky)
            h_minus = hamiltonian(time, kx - delta_kx, ky - delta_ky)
            return (h_plus - h_minus) / (self.hbar * dk)

        velocity = []
        for t in time:
            h_plus = hamiltonian(t, kx + delta_kx, ky + delta_ky)
            h_minus = hamiltonian(t, kx - delta_kx, ky - delta_ky)
            velocity.append((h_plus - h_minus) / (self.hbar * dk))
        return np.asarray(velocity)

    def _static_hamiltonian(self, _time, kx, ky):
        """Adapter exposing H_static with the same signature as H_t."""
        return self.H_static(kx, ky)

    def _expectation_components(
        self,
        state,
        hamiltonian,
        time,
        kx,
        ky,
        dk,
        include_charge,
    ):
        """Compute x and y current components from one state."""
        velocity_x = self._velocity_operator(hamiltonian, time, kx, ky, "x", dk)
        velocity_y = self._velocity_operator(hamiltonian, time, kx, ky, "y", dk)

        current_x = self._expectation_value(state, velocity_x)
        current_y = self._expectation_value(state, velocity_y)
        return (
            self._finalize_current(current_x, include_charge),
            self._finalize_current(current_y, include_charge),
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
        """Compute current from the selected exact Floquet mode.

        Args:
            time: Scalar time or 1D time grid for Floquet-state reconstruction.
            kx: Momentum along x.
            ky: Momentum along y.
            band: Target band label or integer index.
            mode: State-selection rule passed to ``FloquetStateProvider``.
            dk: Momentum increment used for the velocity-operator derivative.
            include_charge: If ``True``, multiply the velocity expectation by
                ``-e`` to return a charge current.
        """
        _, _, floquet_state = self.state_provider.select_floquet_state(
            kx,
            ky,
            band=band,
            mode=mode,
        )
        reconstructed_state = self.state_provider.reconstruct_floquet_state(
            floquet_state,
            time=time,
        )
        return self._expectation_components(
            reconstructed_state,
            self.Ht,
            time,
            kx,
            ky,
            dk,
            include_charge,
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
        _, perturbed_state = self.perturbation_calculator.compute_perturbed_state(
            kx,
            ky,
            band=band,
            order=order,
        )
        reconstructed_state = self.state_provider.reconstruct_floquet_state(
            perturbed_state,
            time=time,
        )
        return self._expectation_components(
            reconstructed_state,
            self.Ht,
            time,
            kx,
            ky,
            dk,
            include_charge,
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
        eigvals, eigvecs = self.state_provider.diagonalize_static_hamiltonian(kx, ky)
        state = self._resolve_band_state(eigvals, eigvecs, band)
        return self._expectation_components(
            state,
            self._static_hamiltonian,
            0.0,
            kx,
            ky,
            dk,
            include_charge,
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
        """Compute current from the instantaneous Hamiltonian H(t, k).

        This is a useful adiabatic benchmark: diagonalize the Hamiltonian at a
        fixed time and take the expectation value of the corresponding
        instantaneous-band state.
        """
        h_inst = self.Ht(time, kx, ky)
        eigvals, eigvecs = np.linalg.eigh(h_inst)
        state = self._resolve_band_state(eigvals, eigvecs, band)
        return self._expectation_components(
            state,
            self.Ht,
            time,
            kx,
            ky,
            dk,
            include_charge,
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
        floquet_builder = FloquetBuilder(
            partial(self.Ht, kx=kx, ky=ky),
            self.omega,
            self.hbar,
            self.floquet_params,
        )
        hfe_builder = HFEBuilder(floquet_builder)
        h_eff = hfe_builder.compute_hfe_hamiltonian(order=order)
        eigvals, eigvecs = np.linalg.eigh(h_eff)
        state = self._resolve_band_state(eigvals, eigvecs, band)

        def effective_hamiltonian(_time, kx_eval, ky_eval):
            builder = FloquetBuilder(
                partial(self.Ht, kx=kx_eval, ky=ky_eval),
                self.omega,
                self.hbar,
                self.floquet_params,
            )
            return HFEBuilder(builder).compute_hfe_hamiltonian(order=order)

        return self._expectation_components(
            state,
            effective_hamiltonian,
            0.0,
            kx,
            ky,
            dk,
            include_charge,
        )
