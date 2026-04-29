"""State-selection helpers shared by Floquet calculators."""

from functools import partial
import numpy as np
from ..core import DrivenBlochHamiltonian
from ..config import FloquetParameters
from ..builders import FloquetBuilder


class FloquetStateProvider:
    """Select band-connected states from Floquet eigensystems.

    Selection can be based on overlap with a static-band state embedded in the
    central Floquet block or by nearest quasienergy to the chosen static band.
    """

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Initialize block layout information from Floquet parameters."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params

        self.dimension = driven_hamiltonian.dimension
        self.omega = driven_hamiltonian.omega

        self.n_trunc = floquet_params.n_trunc
        self.n_blocks = floquet_params.n_blocks
        self.hbar = driven_hamiltonian.units.hbar

    def diagonalize_static_hamiltonian(self, kx, ky):
        """Diagonalize ``H_static(kx, ky)``.

        Returns:
            Tuple ``(eigvals, eigvecs)`` from ``numpy.linalg.eigh``.
        """
        H_static = self.driven_hamiltonian.H_static(kx, ky)
        if not np.allclose(H_static, H_static.conj().T):
            raise ValueError("Static Hamiltonian is not Hermitian")
        static_energy, static_state = np.linalg.eigh(H_static)
        return static_energy, static_state

    def diagonalize_floquet_hamiltonian(self, kx, ky):
        """Diagonalize the truncated Floquet Hamiltonian at one momentum.

        Returns:
            Tuple ``(quasi_energy, floquet_states)`` for the extended-space
            Floquet matrix.
        """

        builder = FloquetBuilder(
            partial(self.driven_hamiltonian.Ht, kx=kx, ky=ky),
            self.omega,
            self.hbar,
            self.floquet_params,
        )
        hamiltonian = builder.compute_floquet_hamiltonian()
        quasi_energy, floquet_states = np.linalg.eigh(hamiltonian)
        return quasi_energy, floquet_states

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
        mode: str = "overlap",
    ):
        """Select an extended-space eigenstate associated with a static band.

        Args:
            kx: Momentum along x.
            ky: Momentum along y.
            band: Target band selector, `conduction`, `valence`, or an index.
            H0: Optional extended-space Hamiltonian to diagonalize and select
                from. If omitted, use the exact Floquet Hamiltonian.
            mode: ``"overlap"`` embeds the target static state in the central
                block and finds the eigenstate that maximizes overlap.
                ``"quasi_energy"`` chooses the nearest eigenvalue to the
                target static energy.

        Returns:
            Column index of the selected extended-space eigenstate, the target
            static state, and the selected eigenstate.
        """
        static_energy, static_states = self.diagonalize_static_hamiltonian(kx, ky)

        if H0 is not None:
            quasi_energy, floquet_states = np.linalg.eigh(H0)
        else:
            quasi_energy, floquet_states = self.diagonalize_floquet_hamiltonian(
                kx,
                ky,
            )

        band_index = self.resolve_band_index(static_energy, band)
        target_energy = static_energy[band_index]
        target_state = static_states[:, band_index]

        if mode == "overlap":
            reference = np.zeros(self.dimension * (self.n_blocks), dtype=complex)
            center = self.n_trunc
            reference[self.dimension * center : self.dimension * (center + 1)] = target_state

            overlaps = np.abs(floquet_states.conj().T @ reference) ** 2
            band_index = np.argmax(overlaps)
        elif mode == "quasi_energy":
            band_index = np.argmin(np.abs(quasi_energy - target_energy))
        else:
            raise ValueError("mode must be 'overlap' or 'quasi_energy'")
        return band_index, target_state, floquet_states[:, band_index]

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
        time = np.atleast_1d(time)
        m_omega = np.arange(-self.n_trunc, self.n_trunc + 1) * self.omega
        exponentials = np.exp(1j * np.outer(m_omega, time))
        floquet_state = np.asarray(floquet_state)

        if floquet_state.ndim == 1:
            v_m = floquet_state.reshape(self.n_blocks, self.dimension)
            state_t = np.tensordot(exponentials.T, v_m, axes=([1], [0]))
            if state_t.shape[0] == 1:
                return state_t[0]
            return state_t

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
                return state_t[0]
            return state_t

        raise ValueError(
            "floquet_state must be a 1D eigenvector or a 2D eigenvector matrix"
        )
