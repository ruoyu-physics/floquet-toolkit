"""Perturbative approximations in extended Floquet space."""

from functools import partial
import numpy as np
from ..builders import FloquetBuilder
from .floquet_state_provider import FloquetStateProvider
from ..config import HBAR


class FloquetPerturbationCalculator:
    """Compute first-order perturbative corrections to Floquet states.

    The unperturbed problem keeps static blocks and photon shifts, while
    off-diagonal harmonic couplings are treated as the perturbation. This is a
    non-degenerate first-order approximation unless resonant subspaces are
    handled separately.
    """

    def __init__(self, driven_hamiltonian, floquet_params):
        """Initialize with the driven model and Floquet truncation settings."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)

        self.omega = driven_hamiltonian.omega
        self.n_trunc = floquet_params.n_trunc
        self.n_blocks = floquet_params.n_blocks
        self.n_harmonics = floquet_params.n_harmonics
        self.dimension = driven_hamiltonian.dimension
        

    def build_perturbation_matrix(self, kx, ky):
        """Split the extended-space Hamiltonian into ``H0`` and ``V``.

        ``H0`` contains diagonal sideband blocks ``H_0 - m*hbar*omega``.
        ``V`` contains off-diagonal harmonic couplings ``H_{m-n}``.

        Returns:
            Tuple ``(H0, V)`` of complex matrices in extended Floquet space.
        """

        builder = FloquetBuilder(
            partial(self.driven_hamiltonian.Ht, kx=kx, ky=ky),
            self.driven_hamiltonian.omega,
            self.floquet_params,
        )
        hs = builder.compute_fourier_harmonics()

        n_harmonics = self.n_harmonics
        omega = self.omega
        n_trunc = self.n_trunc
        N = hs.shape[1]

        H0 = np.zeros(((self.n_blocks) * N, (self.n_blocks) * N), dtype=complex)
        V = np.zeros(((self.n_blocks) * N, (self.n_blocks) * N), dtype=complex)
        for m in range(-n_trunc, n_trunc + 1):
            for n in range(-n_trunc, n_trunc + 1):
                idx_m = m + n_trunc
                idx_n = n + n_trunc

                row = slice(idx_m * N, (idx_m + 1) * N)
                col = slice(idx_n * N, (idx_n + 1) * N)

                harm = m - n

                if m == n:
                    H0[row, col] = hs[n_harmonics] - m * HBAR * omega * np.eye(N)
                elif -n_harmonics <= harm <= n_harmonics:
                    V[row, col] = hs[harm + n_harmonics]

        return H0, V

    def _energy_denominator(self, eig_energy, target_idx, state_idx, atol):
        """Return E_target - E_state, raising on near-degeneracy."""
        energy_diff = eig_energy[target_idx] - eig_energy[state_idx]
        if np.isclose(energy_diff, 0.0, atol=atol):
            raise ValueError(
                "Near-degenerate Floquet sideband encountered; "
                "non-degenerate perturbation theory is invalid."
            )
        return energy_diff

    def compute_perturbed_state(self, kx, ky, band="conduction", order: int = 1, atol=None):
        """Compute a non-degenerate perturbative Floquet state.

        Args:
            kx: Momentum along x.
            ky: Momentum along y.
            band: Static band used to select the central unperturbed Floquet copy.
            order: Perturbative order of the state correction. Supported values
                are ``0`` for the unperturbed state, ``1`` for first order, and
                ``2`` for second order.
            atol: Absolute energy tolerance for rejecting near-degenerate
                denominators. Defaults to ``1e-8*hbar*omega``.

        Returns:
            Tuple ``(unperturbed_state, perturbed_state)``. The returned
            perturbed state is normalized after adding the requested corrections.
        """
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")

        if atol is None:
            atol = 1e-8 * HBAR * self.omega

        H0, V = self.build_perturbation_matrix(kx, ky)

        eig_energy, eig_states = np.linalg.eigh(H0)

        target_idx, static_state, floquet_state = self.state_provider.select_floquet_state(
            kx,
            ky,
            H0=H0,
            band=band,
            mode="overlap",
        )
        unperturbed_state = floquet_state
        perturbed_state = unperturbed_state.copy()

        if order >= 1:
            first_order = np.zeros_like(unperturbed_state, dtype=complex)
            for m in range(len(eig_energy)):
                if m == target_idx:
                    continue
                denom_m = self._energy_denominator(eig_energy, target_idx, m, atol)
                V_mn = eig_states[:, m].conj().T @ V @ unperturbed_state
                first_order += (V_mn / denom_m) * eig_states[:, m]
            perturbed_state += first_order

        if order >= 2:
            second_order = np.zeros_like(unperturbed_state, dtype=complex)
            V_nn = unperturbed_state.conj().T @ V @ unperturbed_state

            for m in range(len(eig_energy)):
                if m == target_idx:
                    continue

                denom_m = self._energy_denominator(eig_energy, target_idx, m, atol)
                state_m = eig_states[:, m]
                V_mn = state_m.conj().T @ V @ unperturbed_state

                coefficient = -V_nn * V_mn / (denom_m ** 2)
                for ell in range(len(eig_energy)):
                    if ell == target_idx:
                        continue
                    denom_ell = self._energy_denominator(eig_energy, target_idx, ell, atol)
                    state_ell = eig_states[:, ell]
                    V_mell = state_m.conj().T @ V @ state_ell
                    V_elln = state_ell.conj().T @ V @ unperturbed_state
                    coefficient += V_mell * V_elln / (denom_m * denom_ell)

                second_order += coefficient * state_m

            # Standard normalization correction in intermediate normalization.
            norm_correction = 0.0
            for m in range(len(eig_energy)):
                if m == target_idx:
                    continue
                denom_m = self._energy_denominator(eig_energy, target_idx, m, atol)
                V_mn = eig_states[:, m].conj().T @ V @ unperturbed_state
                norm_correction += abs(V_mn / denom_m) ** 2
            second_order += -0.5 * norm_correction * unperturbed_state

            perturbed_state += second_order

        perturbed_state /= np.linalg.norm(perturbed_state)

        return unperturbed_state, perturbed_state
