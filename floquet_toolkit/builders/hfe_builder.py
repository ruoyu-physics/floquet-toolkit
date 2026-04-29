"""High-frequency expansion builders for periodic Hamiltonians."""

import numpy as np
from .floquet_builder import FloquetBuilder
from ..config import HBAR

class HFEBuilder:
    """Compute high-frequency effective Hamiltonians from Fourier harmonics.

    The class consumes a ``FloquetBuilder`` so it can reuse the same harmonic
    convention and truncation parameters as the exact Floquet construction.
    """

    def __init__(self, floquet_builder: FloquetBuilder):
        """Store the momentum-resolved Floquet builder used by the expansion."""
        self.floquet_builder = floquet_builder

    def _commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Return the matrix commutator [A, B] = AB - BA."""
        return A @ B - B @ A

    def compute_hfe_hamiltonian(self, order: int = 2):
        """Compute the van Vleck effective Hamiltonian through a chosen order.

        Args:
            order: Expansion order to retain. Supported values are ``0`` for
                ``H_0``, ``1`` for the leading commutator correction, and ``2``
                for the nested-commutator terms of order ``omega^{-2}``.

        Returns:
            Effective static Hamiltonian matrix with shape ``(N, N)``.
        """
        if order not in (0, 1, 2):
            raise ValueError("order must be 0, 1, or 2")
        
        n_harmonics = self.floquet_builder.n_harmonics
        omega = self.floquet_builder.omega

        hs = self.floquet_builder.compute_fourier_harmonics()
        H0 = hs[n_harmonics]  # H_0
        Heff = H0.copy()

        if order >= 1:
            for m in range(1, n_harmonics + 1):
                Hm = hs[n_harmonics + m]  # H_m
                H_minus_m = hs[n_harmonics - m]  # H_{-m}
                Heff += self._commutator(H_minus_m, Hm) / (m * HBAR * omega)

        if order >= 2:
            omega_scale_sq = (HBAR * omega) ** 2

            for m in range(1, n_harmonics + 1):
                Hm = hs[n_harmonics + m]
                H_minus_m = hs[n_harmonics - m]
                Heff += self._commutator(
                    H_minus_m,
                    self._commutator(H0, Hm),
                ) / (2 * (m ** 2) * omega_scale_sq)

            for m in range(-n_harmonics, n_harmonics + 1):
                if m == 0:
                    continue
                H_minus_m = hs[n_harmonics - m]
                for n in range(-n_harmonics, n_harmonics + 1):
                    if n == 0 or m + n == 0:
                        continue
                    intermediate = m + n
                    if not (-n_harmonics <= intermediate <= n_harmonics):
                        continue
                    H_n = hs[n_harmonics + n]
                    H_m_plus_n = hs[n_harmonics + intermediate]
                    Heff += self._commutator(H_minus_m, self._commutator(H_m_plus_n, H_n)) / (
                        3 * m * n * omega_scale_sq
                    )
        return Heff
