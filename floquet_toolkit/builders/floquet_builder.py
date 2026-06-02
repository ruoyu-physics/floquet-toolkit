"""Builders for Fourier harmonics and truncated Floquet Hamiltonians."""

import numpy as np
from ..config import FloquetParameters


class FloquetBuilder:
    """Construct Fourier harmonics and the extended-space Floquet matrix.

    The input Hamiltonian ``Ht(t)`` should already be fixed at a single
    momentum point. Fourier components use the convention
    ``H_m = (1/T) int_0^T dt exp(i m omega t) H(t)``, so the time-domain
    Hamiltonian is reconstructed as ``H(t) = sum_m H_m exp(-i m omega t)``.
    """

    def __init__(self, Ht, omega: float, hbar: float, floquet_params: FloquetParameters):
        """Initialize a builder for one momentum-resolved periodic Hamiltonian.

        Args:
            Ht: Callable ``Ht(t)`` returning an ``N x N`` Hamiltonian matrix.
            omega: Drive angular frequency in radians per second.
            hbar: Reduced Planck constant.
            floquet_params: Truncation and sampling parameters.
        """
        self.Ht = Ht
        self.omega = omega
        self.hbar = hbar
        self.period = 2.0 * np.pi / omega

        self.floquet_params = floquet_params
        self.n_trunc = floquet_params.n_trunc
        self.n_harmonics = floquet_params.n_harmonics
        self.n_time = floquet_params.n_time
        self.n_blocks = floquet_params.n_blocks

    def compute_fourier_harmonics(self):
        """Compute numerical Fourier harmonics ``H_m`` for ``m=-M..M``.

        Returns:
            Complex array with shape ``(2*n_harmonics + 1, N, N)``. The
            harmonic with integer index ``m`` is stored at
            ``hs[m + n_harmonics]``.
        """
        Ht = self.Ht
        ms = np.arange(-self.n_harmonics, self.n_harmonics + 1)
        N = Ht(0).shape[0]
        ts = np.linspace(0, self.period, self.n_time, endpoint=False)

        # Fast path: evaluate H(t) on the whole time grid in one call and
        # contract against the Fourier phase factors. This requires ``Ht`` to
        # be array-aware (returning a stacked ``(n_time, N, N)`` result); when
        # it is not, we fall back to the scalar per-time loop below.
        stacked = None
        try:
            candidate = np.asarray(Ht(ts), dtype=complex)
            if candidate.shape == (self.n_time, N, N):
                stacked = candidate
        except Exception:
            stacked = None

        if stacked is not None:
            phases = np.exp(1j * np.outer(ms, self.omega * ts))  # (2M+1, n_time)
            return np.einsum("mt,tab->mab", phases, stacked) / self.n_time

        # Generic fallback: sample H(t) at Nt scalar points in [0, T).
        hs = np.zeros((2 * self.n_harmonics + 1, N, N), dtype=complex)
        for t in ts:
            ht = Ht(t)
            for j, m in enumerate(ms):
                hs[j] += ht * np.exp(1j * m * self.omega * t)
        hs /= self.n_time  # Average over time samples
        return hs

    def compute_floquet_hamiltonian(self):
        """Build the truncated extended-space Floquet Hamiltonian.

        The block indexed by sidebands ``(m, n)`` is
        ``H_{m-n} - m*hbar*omega*I*delta_{mn}``, with sidebands truncated to
        ``m,n in [-n_trunc, n_trunc]``.

        Returns:
            Complex matrix with shape ``(n_blocks*N, n_blocks*N)``.
        """
        
        hs = self.compute_fourier_harmonics()
        n_trunc = self.n_trunc
        n_harmonics = self.n_harmonics
        N = hs.shape[1]
    
        # The Floquet Hamiltonian has blocks H_{m-n} - m ω δ_{mn}
        # We can construct it as a block matrix where each block is of size N x N
        F_blocks = np.zeros((self.n_blocks, self.n_blocks, N, N), dtype=complex)
        for m in range(-n_trunc, n_trunc + 1):
            for n in range(-n_trunc, n_trunc + 1):
                idx_m = m + n_trunc
                idx_n = n + n_trunc
    
                harm = m - n
                if -n_harmonics <= harm <= n_harmonics:
                    hs_idx = harm + n_harmonics
                    F_blocks[idx_m, idx_n] = hs[hs_idx]

                if m == n:
                    F_blocks[idx_m, idx_n] -= m * self.hbar * self.omega * np.eye(N)  # -m ω δ_{mn}

        # Reshape to (2M+1)*N x (2M+1)*N
        F_matrix = F_blocks.transpose(0, 2, 1, 3).reshape((self.n_blocks) * N, (self.n_blocks) * N)
        return F_matrix
    
