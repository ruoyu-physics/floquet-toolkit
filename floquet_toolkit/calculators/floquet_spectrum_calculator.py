"""Quasienergy and spectral-weight calculations for Floquet spectra."""

import numpy as np
from .floquet_state_provider import FloquetStateProvider
from ..config import HBAR
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian


class FloquetSpectrumCalculator:
    """Compute quasienergies, central-block weights, and static-band overlaps."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Initialize with a driven Hamiltonian and Floquet parameters.

        Args:
            driven_hamiltonian: Model providing ``Ht(t, kx, ky)`` and static
                Hamiltonian information.
            floquet_params: Floquet truncation and sampling parameters.
        """
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)
        self.Ht = driven_hamiltonian.Ht
        self.omega = driven_hamiltonian.omega
        self.period = 2.0 * np.pi / driven_hamiltonian.omega

        self.floquet_params = floquet_params
        self.n_trunc = floquet_params.n_trunc
        self.n_harmonics = floquet_params.n_harmonics
        self.n_blocks = floquet_params.n_blocks
        self.dimension = driven_hamiltonian.dimension

    def _fold_quasi_energies(self, eigvals):
        """Fold extended-space eigenvalues into the first Floquet zone."""
        zone_width = HBAR * self.omega
        return ((eigvals + 0.5 * zone_width) % zone_width) - 0.5 * zone_width

    def _sort_spectrum(self, eigvals, eigvecs):
        """Sort a spectrum by quasienergy."""
        order = np.argsort(eigvals)
        return eigvals[order], eigvecs[:, order]

    def compute_spectral_weight(self, static_band, floquet_states):
        """Compute central-block weights and static-band overlaps.

        Args:
            static_band: One normalized static-band eigenvector at the target
                momentum.
            floquet_states: Extended-space Floquet eigenvectors with states
                stored in columns.

        Returns:
            Tuple ``(central_block_weight, band_overlap)`` where both arrays
            have length ``n_states``. The first stores the total weight in the
            central Floquet block ``m = 0``. The second stores the overlap of
            the reconstructed Floquet state at ``t = 0`` with the supplied
            static-band vector.
        """
        central_block = floquet_states[
            self.n_trunc * self.dimension : (self.n_trunc + 1) * self.dimension,
            :,
        ]
        central_block_weight = np.sum(np.abs(central_block)**2, axis=0)

        reconstructed_floquet_states = self.state_provider.reconstruct_floquet_state(
            floquet_states,
            time=0.0,
        )
        band_overlap = np.abs(static_band.conj().T @ reconstructed_floquet_states)**2
        return central_block_weight, band_overlap

    def compute_quasi_energies(self, kx, ky, band="conduction", fold_to_zone: bool = False):
        """Compute quasienergies, central-block weights, and band overlaps.

        Returns:
            Tuple ``(quasi_energies, central_block_weight, band_overlap)``.
            The last two arrays have length ``n_states`` and measure,
            respectively, visibility in the central Floquet block and overlap
            with the selected static band.
        """
        quasi_energy, floquet_states = self.state_provider.diagonalize_floquet_hamiltonian(kx, ky)
        static_energy, static_states = self.state_provider.diagonalize_static_hamiltonian(kx, ky)
        if fold_to_zone:
            quasi_energy = self._fold_quasi_energies(quasi_energy)
            quasi_energy, floquet_states = self._sort_spectrum(quasi_energy, floquet_states)

        band_idx = self.state_provider.resolve_band_index(static_energy, band)
        central_block_weight, band_overlap = self.compute_spectral_weight(
            static_states[:, band_idx],
            floquet_states,
        )

        return quasi_energy, central_block_weight, band_overlap

    def compute_floquet_spectrum(
        self,
        kx_values,
        ky_values,
        band="conduction",
        fold_to_zone: bool = False,
    ):
        """Compute a quasienergy spectrum over a momentum grid.

        Args:
            kx_values: 1D array of sampled ``kx`` values.
            ky_values: 1D array of sampled ``ky`` values.
            band: Static band label or index used for the overlap projection.
            fold_to_zone: Whether to fold eigenvalues into the first Floquet
                zone ``[-hbar*omega/2, hbar*omega/2)``.

        Returns:
            Tuple ``(quasi_energies, central_block_weight, band_overlap)``.
            Each returned array has shape
            ``(len(kx_values), len(ky_values), n_states)``.
        """
        kx_values = np.asarray(kx_values, dtype=float)
        ky_values = np.asarray(ky_values, dtype=float)

        n_states = self.n_blocks * self.dimension
        quasi_energies = np.zeros((len(kx_values), len(ky_values), n_states))
        central_block_weights = np.zeros((len(kx_values), len(ky_values), n_states))
        band_overlaps = np.zeros((len(kx_values), len(ky_values), n_states))

        for i, kx in enumerate(kx_values):
            for j, ky in enumerate(ky_values):
                quasi_energy, weights, band_overlap = self.compute_quasi_energies(
                    kx,
                    ky,
                    band=band,
                    fold_to_zone=fold_to_zone,
                )
                quasi_energies[i, j] = quasi_energy
                central_block_weights[i, j] = weights
                band_overlaps[i, j] = band_overlap

        return quasi_energies, central_block_weights, band_overlaps
