"""Velocity-like expectation values for static and Floquet states."""

from functools import partial

import numpy as np

from ..builders import FloquetBuilder, HFEBuilder
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .floquet_perturbation_calculator import FloquetPerturbationCalculator
from .states import FloquetStateCache, FloquetStateProvider


class FloquetVelocityCalculator:
    """Compute local velocity expectations from several state sources.

    The returned values are single-particle velocity expectation values by
    default. Set ``include_charge=True`` to multiply by the configured charge
    ``e_charge`` (from the unit convention) and return charge-current
    expectations instead.
    """

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        cache: FloquetStateCache | None = None,
    ):
        """Initialize from one driven model and one Floquet parameter set."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.state_provider = FloquetStateProvider(
            driven_hamiltonian,
            floquet_params,
            cache=cache,
        )
        self.perturbation_calculator = FloquetPerturbationCalculator(
            driven_hamiltonian,
            floquet_params,
        )

        self.Ht = driven_hamiltonian.Ht
        self.H_static = driven_hamiltonian.H_static
        self.omega = driven_hamiltonian.omega
        self.e_charge = driven_hamiltonian.units.e_charge
        self.hbar = driven_hamiltonian.units.hbar
        # Optional fast paths advertised by the model (default to the generic
        # finite-difference loop when absent, e.g. for custom Hamiltonians).
        self.analytic_velocity_operator = getattr(
            driven_hamiltonian, "analytic_velocity_operator", None
        )
        self.supports_vectorized_time = getattr(
            driven_hamiltonian, "supports_vectorized_time", False
        )

    def _resolve_band_state(self, eigvals, eigvecs, band):
        """Return the eigenvector associated with a band label or index."""
        band_index = self.state_provider.resolve_band_index(eigvals, band)
        return eigvecs[:, band_index]

    @staticmethod
    def _expectation(states, operator):
        """Return ``<psi|operator|psi>`` for one state or a batch of states.

        ``states`` has shape ``(..., dim)`` and ``operator`` broadcasts to
        ``(..., dim, dim)`` (a bare ``(dim, dim)`` constant operator is fine).
        The leading axes are preserved, so this single primitive serves both the
        single-state case (leading shape ``()`` -> scalar; ``(n_time,)`` ->
        ``(n_time,)``) and the batched grid case (leading shape ``G + (n_time,)``).
        """
        states = np.asarray(states)
        operator = np.asarray(operator)
        op_state = np.matmul(operator, states[..., None])[..., 0]
        return np.sum(np.conj(states) * op_state, axis=-1)

    def _finalize_velocity(self, expectation, include_charge):
        """Convert a velocity expectation into a charge current if requested."""
        if include_charge:
            return self.e_charge * expectation.real
        return expectation.real

    def _velocity_operators(
        self,
        hamiltonian,
        time,
        kx,
        ky,
        dk,
        velocity_operator_fn=None,
    ):
        """Return the velocity operators ``((1/hbar) dH/dkx, (1/hbar) dH/dky)``.

        The two Cartesian components are returned together because every caller
        needs both. Resolution order:

        1. If ``velocity_operator_fn`` is supplied (the model provides a closed
           form for ``dH_t/dk``), call it per axis — exact and loop-free.
        2. Otherwise finite-difference ``hamiltonian``. When the model supports
           vectorized time (or ``time`` is scalar), evaluate the whole time
           grid in one call, returning ``(n_time, dim, dim)`` operators.
        3. Otherwise fall back to the generic per-time Python loop, preserving
           behavior for models whose ``H_t`` is not array-aware.
        """
        if velocity_operator_fn is not None:
            return (
                velocity_operator_fn(time, kx, ky, "x"),
                velocity_operator_fn(time, kx, ky, "y"),
            )

        half = dk / 2.0
        time = np.asarray(time)
        if time.ndim == 0 or self.supports_vectorized_time:
            velocity_x = (
                hamiltonian(time, kx + half, ky) - hamiltonian(time, kx - half, ky)
            ) / (self.hbar * dk)
            velocity_y = (
                hamiltonian(time, kx, ky + half) - hamiltonian(time, kx, ky - half)
            ) / (self.hbar * dk)
            return velocity_x, velocity_y

        velocity_x, velocity_y = [], []
        for t in time:
            velocity_x.append(
                (hamiltonian(t, kx + half, ky) - hamiltonian(t, kx - half, ky))
                / (self.hbar * dk)
            )
            velocity_y.append(
                (hamiltonian(t, kx, ky + half) - hamiltonian(t, kx, ky - half))
                / (self.hbar * dk)
            )
        return np.asarray(velocity_x), np.asarray(velocity_y)

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
        velocity_operator_fn=None,
    ):
        """Compute x and y velocity components from one state."""
        velocity_x, velocity_y = self._velocity_operators(
            hamiltonian, time, kx, ky, dk, velocity_operator_fn
        )

        current_x = self._expectation(state, velocity_x)
        current_y = self._expectation(state, velocity_y)
        return (
            self._finalize_velocity(current_x, include_charge),
            self._finalize_velocity(current_y, include_charge),
        )

    def compute_floquet_velocity(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
    ):
        """Compute velocity from the selected exact Floquet mode.

        Args:
            time: Scalar time or 1D time grid for Floquet-state reconstruction.
            kx: Momentum along x.
            ky: Momentum along y.
            band: Target band label or integer index.
            dk: Momentum increment used for the velocity-operator derivative.
            include_charge: If ``True``, multiply the velocity expectation by
                the configured charge ``e_charge`` to return a charge current.
            band_selection_mode: State-selection rule passed to
                ``FloquetStateProvider``.
        """
        _, _, floquet_state = self.state_provider.select_floquet_state(
            kx,
            ky,
            band=band,
            band_selection_mode=band_selection_mode,
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
            velocity_operator_fn=self.analytic_velocity_operator,
        )

    def compute_floquet_velocity_from_state(
        self,
        time,
        kx,
        ky,
        floquet_state,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute velocity from a supplied Floquet eigenstate."""
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
            velocity_operator_fn=self.analytic_velocity_operator,
        )

    def _velocity_operators_batched(self, time, kx_flat, ky_flat, dk):
        """Build velocity operators for a flat batch of momenta over a time grid.

        Returns ``(vx, vy)``. When the model is momentum/time batchable (or
        provides an analytic operator) the operators broadcast to
        ``(n_points, n_time, dim, dim)`` in one call; otherwise they are built
        per momentum, reproducing the single-``k`` finite-difference path.
        """
        velocity_operator_fn = self.analytic_velocity_operator
        time = np.atleast_1d(np.asarray(time, dtype=float))
        kx_flat = np.asarray(kx_flat, dtype=float)
        ky_flat = np.asarray(ky_flat, dtype=float)

        if velocity_operator_fn is None and not self.supports_vectorized_time:
            velocity_x, velocity_y = [], []
            for kx, ky in zip(kx_flat, ky_flat):
                vx_op, vy_op = self._velocity_operators(self.Ht, time, kx, ky, dk)
                velocity_x.append(vx_op)
                velocity_y.append(vy_op)
            return np.asarray(velocity_x), np.asarray(velocity_y)

        kx_b = kx_flat[:, None]
        ky_b = ky_flat[:, None]
        t_b = time[None, :]
        return self._velocity_operators(
            self.Ht, t_b, kx_b, ky_b, dk, velocity_operator_fn
        )

    def compute_floquet_velocity_map_from_states(
        self,
        time,
        kx_grid,
        ky_grid,
        floquet_states,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Batched local-velocity map over a k-grid from pre-selected states.

        This is the vectorized counterpart of
        :meth:`compute_floquet_velocity_from_state` evaluated over a whole grid.
        State selection is assumed already done (e.g. by the branch tracker);
        this routine only reconstructs the states and evaluates the velocity
        expectation in batch.

        Args:
            time: 1D time grid.
            kx_grid, ky_grid: Momentum grids of identical shape ``G``.
            floquet_states: ``G + (n_blocks * dimension,)`` selected
                extended-space eigenvectors.
            dk: Momentum step for the velocity-operator derivative.
            include_charge: Multiply by the charge to return a current.

        Returns:
            ``(jx_map, jy_map)``, each of shape ``G + (n_time,)``.
        """
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        grid_shape = kx_grid.shape
        time = np.atleast_1d(np.asarray(time, dtype=float))

        n_states = self.state_provider.n_blocks * self.state_provider.dimension
        states_flat = np.asarray(floquet_states).reshape(-1, n_states)
        kx_flat = kx_grid.ravel()
        ky_flat = ky_grid.ravel()

        reconstructed = self.state_provider.reconstruct_floquet_states_batched(
            states_flat, time
        )
        velocity_x, velocity_y = self._velocity_operators_batched(
            time, kx_flat, ky_flat, dk
        )
        current_x = self._expectation(reconstructed, velocity_x)
        current_y = self._expectation(reconstructed, velocity_y)

        jx_map = self._finalize_velocity(current_x, include_charge)
        jy_map = self._finalize_velocity(current_y, include_charge)
        out_shape = grid_shape + (time.size,)
        return jx_map.reshape(out_shape), jy_map.reshape(out_shape)

    def compute_floquet_velocity_map(
        self,
        time,
        kx_grid,
        ky_grid,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
    ):
        """Batched local-velocity map with independent per-point selection.

        Vectorized counterpart of :meth:`compute_floquet_velocity` over a grid:
        the Floquet eigensystems are diagonalized in one batched call (when the
        model supports it), the band-connected state is selected independently
        at each momentum, and the velocity expectation is evaluated in batch.

        Args:
            time: 1D time grid.
            kx_grid, ky_grid: Momentum grids of identical shape ``G``.
            band: Target band label or integer index.
            dk: Momentum step for the velocity-operator derivative.
            include_charge: Multiply by the charge to return a current.
            band_selection_mode: State-selection rule passed to the provider.

        Returns:
            ``(jx_map, jy_map)``, each of shape ``G + (n_time,)``.
        """
        provider = self.state_provider
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        grid_shape = kx_grid.shape

        n_states = provider.n_blocks * provider.dimension
        selected = provider.select_floquet_states_on_grid(
            kx_grid.ravel(),
            ky_grid.ravel(),
            band=band,
            band_selection_mode=band_selection_mode,
        )

        selected = selected.reshape(grid_shape + (n_states,))
        return self.compute_floquet_velocity_map_from_states(
            time,
            kx_grid,
            ky_grid,
            selected,
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
        """Compute velocity from the perturbative Floquet state."""
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
            velocity_operator_fn=self.analytic_velocity_operator,
        )

    def compute_static_velocity(
        self,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute velocity from the static Hamiltonian eigenstate."""
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

    def compute_adiabatic_velocity(
        self,
        time,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute velocity from the adiabatic state of H(t, k).

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
            velocity_operator_fn=self.analytic_velocity_operator,
        )

    def compute_adiabatic_velocity_map(
        self,
        time,
        kx_grid,
        ky_grid,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Batched adiabatic-velocity map over a k-grid and time grid.

        Vectorized counterpart of :meth:`compute_adiabatic_velocity` evaluated
        over a whole grid: the instantaneous Hamiltonian is diagonalized at
        every ``(k, t)`` in one batched ``eigh``, the band-resolved eigenstate
        is selected, and the velocity expectation is evaluated in batch. Falls
        back to the per-point routine for non-batchable models.

        Args:
            time: 1D time grid.
            kx_grid, ky_grid: Momentum grids of identical shape ``G``.
            band: Target band label or integer index.
            dk: Momentum step for the velocity-operator derivative.
            include_charge: Multiply by the charge to return a current.

        Returns:
            ``(jx_map, jy_map)``, each of shape ``G + (n_time,)``.
        """
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        grid_shape = kx_grid.shape
        time = np.atleast_1d(np.asarray(time, dtype=float))
        kx_flat = kx_grid.ravel()
        ky_flat = ky_grid.ravel()
        out_shape = grid_shape + (time.size,)

        if not self.supports_vectorized_time:
            local_jx = np.zeros((kx_flat.size, time.size), dtype=float)
            local_jy = np.zeros((kx_flat.size, time.size), dtype=float)
            for index in range(kx_flat.size):
                jx_t, jy_t = zip(
                    *[
                        self.compute_adiabatic_velocity(
                            t,
                            kx_flat[index],
                            ky_flat[index],
                            band=band,
                            dk=dk,
                            include_charge=include_charge,
                        )
                        for t in time
                    ]
                )
                local_jx[index] = np.asarray(jx_t)
                local_jy[index] = np.asarray(jy_t)
            return local_jx.reshape(out_shape), local_jy.reshape(out_shape)

        kx_b = kx_flat[:, None]
        ky_b = ky_flat[:, None]
        t_b = time[None, :]
        # Instantaneous eigenstates at every (k, t): (n_k, n_time, dim, dim).
        h_inst = np.asarray(self.Ht(t_b, kx_b, ky_b), dtype=complex)
        eigvals, eigvecs = np.linalg.eigh(h_inst)
        band_index = self.state_provider.resolve_band_index(
            eigvals.reshape(-1, eigvals.shape[-1])[0], band
        )
        states = eigvecs[..., band_index]  # (n_k, n_time, dim)

        velocity_x, velocity_y = self._velocity_operators_batched(
            time, kx_flat, ky_flat, dk
        )
        current_x = self._expectation(states, velocity_x)
        current_y = self._expectation(states, velocity_y)

        jx_map = self._finalize_velocity(current_x, include_charge)
        jy_map = self._finalize_velocity(current_y, include_charge)
        return jx_map.reshape(out_shape), jy_map.reshape(out_shape)

    def compute_hfe_velocity(
        self,
        kx,
        ky,
        band="conduction",
        order: int = 2,
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute velocity from the high-frequency effective Hamiltonian."""
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

    def compute_velocity_from_quasi_energy_spectrum(
        self,
        kx,
        ky,
        band="conduction",
        dk: float = 1e5,
        include_charge: bool = False,
    ):
        """Compute velocity from the quasi-energy spectrum slope."""
        def quasi_energy_state(kx, ky, band="conduction"):
            all_quasi_energies, all_floquet_states = self.state_provider.diagonalize_floquet_hamiltonian(kx, ky)
            state_index = self.state_provider.resolve_band_index(all_quasi_energies, band)
            return all_quasi_energies[state_index]
        
        energy1 = quasi_energy_state(kx - dk/2, ky, band=band)
        energy2 = quasi_energy_state(kx, ky, band=band)
        energy3 = quasi_energy_state(kx + dk/2, ky, band=band)
        velocity_x = np.gradient([energy1, energy2, energy3], dk)/self.hbar

        energy1 = quasi_energy_state(kx, ky - dk/2, band=band)
        energy2 = quasi_energy_state(kx, ky, band=band)
        energy3 = quasi_energy_state(kx, ky + dk/2, band=band)
        velocity_y = np.gradient([energy1, energy2, energy3], dk)/self.hbar
        if include_charge:
            return self.e_charge * velocity_x[1], self.e_charge * velocity_y[1]
        return velocity_x[1], velocity_y[1]


        
