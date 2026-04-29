"""Container for driven Bloch Hamiltonians and their static averages."""

from collections.abc import Callable
import numpy as np

class DrivenBlochHamiltonian:
    """Represent a 2D driven Bloch Hamiltonian ``H(k, t)``.

    ``H_t(t, kx, ky)`` is always the full time-dependent Hamiltonian. An
    analytic static Hamiltonian ``H_static(kx, ky)`` may also be supplied; when
    it is omitted, static quantities are computed from the numerical time
    average of ``H_t`` over one drive period.
    """

    def __init__(
        self,
        H_t: Callable,
        omega: float,
        H_static: Callable | None = None,
        analytic_static_berry_curvature: Callable | None = None,
        static_average_samples: int = 128,
    ):
        """Initialize and validate the driven Hamiltonian callables.

        Args:
            H_t: Callable returning the full Hamiltonian ``H(t, kx, ky)``.
                This may include both static and oscillatory contributions.
            omega: Drive angular frequency.
            H_static: Optional callable returning the static matrix at
                ``(kx, ky)``. If omitted, it is computed by time averaging
                ``H_t`` numerically.
            analytic_static_berry_curvature: Optional callable for analytic
                static Berry curvature.
            static_average_samples: Number of uniform samples used for the
                numerical time average when ``H_static`` is omitted.
        """
    
        if static_average_samples <= 0:
            raise ValueError("static_average_samples must be a positive integer")

        self.Ht = H_t
        self.omega = omega
        self.period = 2.0 * np.pi / self.omega
        self.static_average_samples = static_average_samples
        self.analytic_static_berry_curvature = analytic_static_berry_curvature
        self._static_cache = {}

        sample_full = np.asarray(H_t(t=0, kx=0, ky=0), dtype=complex)
        if sample_full.shape[0] != sample_full.shape[1]:
            raise ValueError("Hamiltonian must be square")
        if not np.allclose(sample_full, sample_full.conj().T):
            raise ValueError("H_t must return a Hermitian matrix")

        self.dimension = sample_full.shape[0]

        # Validate H_static if provided, otherwise set up for numerical time averaging.
        if H_static is None:
            sample_static = self._compute_static_average(kx=0, ky=0)
            self.H_static = self._compute_static_average
        else:
            sample_static = np.asarray(H_static(kx=0, ky=0), dtype=complex)
            if sample_static.shape != sample_full.shape:
                raise ValueError("H_static and H_t must have the same shape")
            if not np.allclose(sample_static, sample_static.conj().T):
                raise ValueError("H_static must return a Hermitian matrix")
            self.H_static = H_static

        if not np.allclose(sample_static, sample_static.conj().T):
            raise ValueError("H_static must return a Hermitian matrix")

    def _compute_static_average(self, kx, ky):
        """Return the numerical time average of ``H_t``."""
        cache_key = (kx, ky)
        if cache_key in self._static_cache:
            return self._static_cache[cache_key]

        time = np.linspace(0.0, self.period, self.static_average_samples, endpoint=False)
        hamiltonians = np.asarray(
            [self.Ht(t, kx, ky) for t in time],
            dtype=complex,
        )
        average = np.mean(hamiltonians, axis=0)
        average = 0.5 * (average + average.conj().T)
        self._static_cache[cache_key] = average
        return average

    
