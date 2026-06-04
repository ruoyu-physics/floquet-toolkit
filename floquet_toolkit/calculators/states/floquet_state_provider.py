"""State-selection helpers shared by Floquet calculators."""

from functools import partial
import numpy as np
from ...core import DrivenBlochHamiltonian
from ...config import FloquetParameters
from ...builders import FloquetBuilder
from .floquet_state_cache import FloquetStateCache

# Available options:
# - "principal_zone": keep one Floquet-zone representative set in [-hbar*omega/2, hbar*omega/2)
# - "central_block_norm": keep the `dimension` eigenvectors with the largest m=0 block norm
CANDIDATE_SUBSPACE_MODE = "central_block_norm"


class FloquetStateProvider:
    """Select band-connected states from Floquet eigensystems.

    Selection compares a small candidate set of Floquet eigenstates against one
    target static-band state after reconstructing the Floquet states at ``t=0``.
    """

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        cache: FloquetStateCache | None = None,
    ):
        """Initialize block layout information from Floquet parameters."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.cache = cache

        self.dimension = driven_hamiltonian.dimension
        self.omega = driven_hamiltonian.omega

        self.n_trunc = floquet_params.n_trunc
        self.n_blocks = floquet_params.n_blocks
        self.hbar = driven_hamiltonian.units.hbar

    def _fold_quasi_energies_to_zone(self, quasi_energy):
        """Fold quasienergies into the principal Floquet Brillouin zone."""
        quasi_energy = np.asarray(quasi_energy, dtype=float)
        zone_width = self.hbar * self.omega
        folded = (
            (quasi_energy + 0.5 * zone_width) % zone_width
        ) - 0.5 * zone_width
        return folded

    def _principal_zone_candidate_indices(self, quasi_energy):
        """Return one Floquet-zone representative set of physical candidates."""
        quasi_energy = np.asarray(quasi_energy, dtype=float)
        zone_half_width = 0.5 * self.hbar * self.omega
        tolerance = max(1.0e-12 * abs(self.hbar * self.omega), 1.0e-30)

        in_zone = np.where(
            (quasi_energy >= -zone_half_width - tolerance)
            & (quasi_energy < zone_half_width + tolerance)
        )[0]
        if in_zone.size == self.dimension:
            return np.sort(in_zone)

        folded = self._fold_quasi_energies_to_zone(quasi_energy)
        zone_distance = np.abs(quasi_energy - folded)
        ranked = np.lexsort((np.abs(folded), zone_distance))
        return np.sort(ranked[: self.dimension])

    def _central_block_candidate_indices(self, floquet_states):
        """Return the candidates with the largest weight in the central Floquet block."""
        floquet_states = np.asarray(floquet_states)
        central_block = floquet_states[
            self.n_trunc * self.dimension : (self.n_trunc + 1) * self.dimension,
            :,
        ]
        central_block_norms = np.sum(np.abs(central_block) ** 2, axis=0)
        ranked = np.argsort(-central_block_norms, kind="stable")
        return np.sort(ranked[: self.dimension])

    def _zone_candidate_data(self, quasi_energy, floquet_states):
        """Return candidate indices and reconstructed states for state selection."""
        if CANDIDATE_SUBSPACE_MODE == "principal_zone":
            candidate_indices = self._principal_zone_candidate_indices(quasi_energy)
        elif CANDIDATE_SUBSPACE_MODE == "central_block_norm":
            candidate_indices = self._central_block_candidate_indices(floquet_states)
        else:
            raise ValueError(
                "CANDIDATE_SUBSPACE_MODE must be 'principal_zone' or "
                "'central_block_norm'."
            )
        candidate_vectors = floquet_states[:, candidate_indices]
        candidate_states_t0 = self.reconstruct_floquet_state(
            candidate_vectors,
            time=0.0,
        )
        return candidate_indices, candidate_vectors, candidate_states_t0

    def _align_extended_state_phase(
        self,
        reference_state_t0,
        candidate_state_t0,
        candidate_extended_state,
        atol=1e-14,
    ):
        """Fix the global phase of an extended Floquet state using t=0 overlap."""
        overlap = np.vdot(reference_state_t0, candidate_state_t0)
        if abs(overlap) < atol:
            return candidate_extended_state
        return candidate_extended_state * np.exp(-1j * np.angle(overlap))

    def diagonalize_static_hamiltonian(self, kx, ky):
        """Diagonalize ``H_static(kx, ky)``.

        Returns:
            Tuple ``(eigvals, eigvecs)`` from ``numpy.linalg.eigh``.
        """
        if self.cache is not None:
            cached = self.cache.get_static_eigensystem(kx, ky)
            if cached is not None:
                return cached

        H_static = self.driven_hamiltonian.H_static(kx, ky)
        if not np.allclose(H_static, H_static.conj().T):
            raise ValueError("Static Hamiltonian is not Hermitian")
        static_energy, static_state = np.linalg.eigh(H_static)
        if self.cache is not None:
            self.cache.store_static_eigensystem(
                kx,
                ky,
                static_energy,
                static_state,
            )
        return static_energy, static_state

    def diagonalize_floquet_hamiltonian(self, kx, ky):
        """Diagonalize the truncated Floquet Hamiltonian at one momentum.

        Returns:
            Tuple ``(quasi_energy, floquet_states)`` for the extended-space
            Floquet matrix.
        """
        if self.cache is not None:
            cached = self.cache.get_floquet_eigensystem(kx, ky)
            if cached is not None:
                return cached

        builder = FloquetBuilder(
            partial(self.driven_hamiltonian.Ht, kx=kx, ky=ky),
            self.omega,
            self.hbar,
            self.floquet_params,
        )
        hamiltonian = builder.compute_floquet_hamiltonian()
        quasi_energy, floquet_states = np.linalg.eigh(hamiltonian)
        if self.cache is not None:
            self.cache.store_floquet_eigensystem(
                kx,
                ky,
                quasi_energy,
                floquet_states,
            )
        return quasi_energy, floquet_states

    def diagonalize_floquet_hamiltonian_batched(self, kxs, kys):
        """Diagonalize the Floquet Hamiltonian at many momenta in one batch.

        Uses the vectorized construction (one ``(k, t)`` evaluation of ``Ht``)
        and a single batched ``eigh`` over the stacked matrices. Intended for
        grid/sweep workflows; single-momentum callers use
        :meth:`diagonalize_floquet_hamiltonian`. Requires the model's ``Ht`` to
        broadcast over a momentum axis (``supports_vectorized_time``).

        Returns:
            Tuple ``(quasi_energy, floquet_states)`` with shapes
            ``(n_k, D)`` and ``(n_k, D, D)``.
        """
        builder = FloquetBuilder(
            self.driven_hamiltonian.Ht,
            self.omega,
            self.hbar,
            self.floquet_params,
        )
        hamiltonians = builder.compute_floquet_hamiltonians_batched(
            np.asarray(kxs, dtype=float),
            np.asarray(kys, dtype=float),
        )
        return np.linalg.eigh(hamiltonians)

    def resolve_band_index(self, static_energy, band):
        """Convert a band label or integer into an eigenvector column index.

        Args:
            static_energy: Sorted eigenvalues associated with a static
                Hamiltonian.
            band: ``"valence"``, ``"conduction"``, or an integer index.

        Returns:
            Integer index into the eigenvector array.
        """
        if isinstance(band, str):
            if band == "valence":
                return 0
            if band == "conduction":
                if len(static_energy) < 2:
                    raise ValueError(
                        "Band 'conduction' requires at least two static bands"
                    )
                return 1
            raise ValueError(
                "Band must be 'conduction', 'valence', or an integer index"
            )

        band_index = int(band)
        if not 0 <= band_index < len(static_energy):
            raise IndexError("band index out of range")
        return band_index

    def select_floquet_state(
        self,
        kx,
        ky,
        band="conduction",
        H0=None,
        band_selection_mode: str = "overlap",
    ):
        """Select an extended-space eigenstate associated with a static band.

        Args:
            kx: Momentum along x.
            ky: Momentum along y.
            band: Target band selector, `conduction`, `valence`, or an index.
            H0: Optional extended-space Hamiltonian to diagonalize and select
                from. If omitted, use the exact Floquet Hamiltonian.
            band_selection_mode: State-selection rule. The only supported
                value is ``"overlap"``, which embeds the target static state
                in the central Floquet block and selects the eigenstate with
                maximal overlap at ``t=0``.

        Returns:
            Column index of the selected extended-space eigenstate, the target
            static state, and the selected eigenstate.
        """
        if H0 is None and self.cache is not None:
            cached = self.cache.get_selected_state(
                kx,
                ky,
                band=band,
                band_selection_mode=band_selection_mode,
            )
            if cached is not None:
                return cached

        if H0 is not None:
            quasi_energy, floquet_states = np.linalg.eigh(H0)
        else:
            quasi_energy, floquet_states = self.diagonalize_floquet_hamiltonian(
                kx,
                ky,
            )

        selected = self.select_floquet_state_from_eigensystem(
            kx,
            ky,
            quasi_energy,
            floquet_states,
            band=band,
            band_selection_mode=band_selection_mode,
        )
        if H0 is None and self.cache is not None:
            self.cache.store_selected_state(
                kx,
                ky,
                selected[0],
                selected[1],
                selected[2],
                band=band,
                band_selection_mode=band_selection_mode,
            )
        return selected

    def select_floquet_state_from_eigensystem(
        self,
        kx,
        ky,
        quasi_energy,
        floquet_states,
        band="conduction",
        band_selection_mode: str = "overlap",
    ):
        """Select a band-connected eigenstate from a precomputed eigensystem.

        Carries out the overlap-based selection for one momentum given an
        already-diagonalized Floquet eigensystem, so callers that diagonalize
        a whole grid in one batched call (see
        :meth:`diagonalize_floquet_hamiltonian_batched`) can reuse the result
        instead of re-diagonalizing per point. ``select_floquet_state`` is the
        single-momentum entry point built on top of this.

        Args:
            kx: Momentum along x.
            ky: Momentum along y.
            quasi_energy: Quasienergies from the Floquet eigensystem.
            floquet_states: Extended-space eigenvectors from the same
                eigensystem.
            band: Target band selector, `conduction`, `valence`, or an index.
            band_selection_mode: State-selection rule; only ``"overlap"`` is
                supported.

        Returns:
            Column index of the selected extended-space eigenstate, the target
            static state, and the selected eigenstate.
        """
        static_energy, static_states = self.diagonalize_static_hamiltonian(kx, ky)
        band_index = self.resolve_band_index(static_energy, band)
        target_state = static_states[:, band_index]
        candidate_indices, _, candidate_states_t0 = self._zone_candidate_data(
            quasi_energy,
            floquet_states,
        )

        if band_selection_mode != "overlap":
            raise ValueError(
                "band_selection_mode must be 'overlap'"
            )
        overlaps = np.abs(candidate_states_t0.conj().T @ target_state) ** 2
        local_index = int(np.argmax(overlaps))
        band_index = int(candidate_indices[local_index])
        selected_state = self._align_extended_state_phase(
            target_state,
            candidate_states_t0[:, local_index],
            floquet_states[:, band_index],
        )
        return band_index, target_state, selected_state

    def reconstruct_floquet_state(self, floquet_state, time):
        """Reconstruct time-dependent Floquet modes from sideband components.

        Args:
            floquet_state: Either a single Floquet eigenvector with shape
                ``(n_blocks * dimension,)`` or the full eigenvector matrix from
                ``numpy.linalg.eigh`` with shape
                ``(n_blocks * dimension, n_states)``.
            time: Scalar time or 1D time grid.

        Returns:
            For a single eigenvector:
                - ``(dimension,)`` for scalar time
                - ``(n_time, dimension)`` for a time grid
            For an eigenvector matrix:
                - ``(dimension, n_states)`` for scalar time
                - ``(n_time, dimension, n_states)`` for a time grid
        """
        if self.cache is not None:
            cached = self.cache.get_reconstructed_state(floquet_state, time)
            if cached is not None:
                return cached

        original_time = np.asarray(time)
        time = np.atleast_1d(time)
        m_omega = np.arange(-self.n_trunc, self.n_trunc + 1) * self.omega
        exponentials = np.exp(-1j * np.outer(m_omega, time))
        floquet_state = np.asarray(floquet_state)

        if floquet_state.ndim == 1:
            v_m = floquet_state.reshape(self.n_blocks, self.dimension)
            state_t = np.tensordot(exponentials.T, v_m, axes=([1], [0]))
            if state_t.shape[0] == 1:
                result = state_t[0]
            else:
                result = state_t
            if self.cache is not None:
                self.cache.store_reconstructed_state(
                    floquet_state,
                    original_time,
                    result,
                )
            return result

        if floquet_state.ndim == 2:
            n_rows, n_states = floquet_state.shape
            expected_rows = self.n_blocks * self.dimension
            if n_rows != expected_rows:
                raise ValueError(
                    "Floquet eigenvector matrix must have shape "
                    f"({expected_rows}, n_states); got {floquet_state.shape}."
                )

            v_m = floquet_state.reshape(self.n_blocks, self.dimension, n_states)
            state_t = np.tensordot(exponentials.T, v_m, axes=([1], [0]))
            if state_t.shape[0] == 1:
                result = state_t[0]
            else:
                result = state_t
            if self.cache is not None:
                self.cache.store_reconstructed_state(
                    floquet_state,
                    original_time,
                    result,
                )
            return result

        raise ValueError(
            "floquet_state must be a 1D eigenvector or a 2D eigenvector matrix"
        )
