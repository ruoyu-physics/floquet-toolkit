"""Container for driven Bloch Hamiltonians and their static averages."""

from collections.abc import Callable
import numpy as np
from ..config import UnitConvention

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
        analytic_velocity_operator: Callable | None = None,
        supports_vectorized_time: bool = False,
        static_average_samples: int = 128,
        units: UnitConvention = UnitConvention.SI_UNITS(),
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
            analytic_velocity_operator: Optional callable
                ``f(time, kx, ky, axis)`` returning the velocity operator
                ``(1/hbar) dH_t/dk`` in closed form. When supplied, the
                velocity calculator uses it instead of finite differencing
                ``H_t`` — both faster and exact. ``axis`` is ``"x"`` or
                ``"y"``.
            supports_vectorized_time: Whether ``H_t`` accepts a 1D array of
                times and returns a stacked ``(n_time, dim, dim)`` result. When
                ``True`` the velocity calculator builds the finite-difference
                operator with a single vectorized call instead of a Python loop
                over time samples.
            static_average_samples: Number of uniform samples used for the
                numerical time average when ``H_static`` is omitted.
        """

        if static_average_samples <= 0:
            raise ValueError("static_average_samples must be a positive integer")

        self.Ht = H_t
        self.omega = omega
        self.units = units
        self.hbar = units.hbar
        self.period = 2.0 * np.pi / self.omega
        self.static_average_samples = static_average_samples
        self.analytic_static_berry_curvature = analytic_static_berry_curvature
        self.analytic_velocity_operator = analytic_velocity_operator
        self.supports_vectorized_time = supports_vectorized_time
        self._static_cache = {}

        # Validate with the calling convention the solvers actually use: a
        # positional time argument plus keyword-addressed momenta (the builders
        # bind ``partial(H_t, kx=..., ky=...)``). So ``def H(time, kx, ky)`` is
        # accepted, while momenta named anything but ``kx``/``ky`` fail here
        # rather than deep inside a solver.
        try:
            sample_full = np.asarray(H_t(0.0, kx=0.0, ky=0.0), dtype=complex)
        except TypeError as exc:
            raise TypeError(
                "H_t must be callable as H_t(t, kx=..., ky=...): a positional "
                "time argument plus momenta keyword-addressable as 'kx' and 'ky'."
            ) from exc
        if sample_full.shape[0] != sample_full.shape[1]:
            raise ValueError("Hamiltonian must be square")
        if not np.allclose(sample_full, sample_full.conj().T):
            raise ValueError("H_t must return a Hermitian matrix")

        self.dimension = sample_full.shape[0]

        # Validate H_static if provided, otherwise set up for numerical time averaging.
        if H_static is None:
            sample_static = self._compute_static_average(0.0, 0.0)
            self.H_static = self._compute_static_average
        else:
            # The calculators call H_static positionally: H_static(kx, ky).
            try:
                sample_static = np.asarray(H_static(0.0, 0.0), dtype=complex)
            except TypeError as exc:
                raise TypeError(
                    "H_static must be callable as H_static(kx, ky) with two "
                    "positional momentum arguments."
                ) from exc
            if sample_static.shape != sample_full.shape:
                raise ValueError("H_static and H_t must have the same shape")
            if not np.allclose(sample_static, sample_static.conj().T):
                raise ValueError("H_static must return a Hermitian matrix")
            self.H_static = H_static

        if not np.allclose(sample_static, sample_static.conj().T):
            raise ValueError("H_static must return a Hermitian matrix")

    def _compute_static_average(self, kx, ky):
        """Return the numerical time average of ``H_t``.

        Scalar momenta are cached per ``(kx, ky)``. Array momenta are broadcast
        against each other, evaluated point by point, and stacked to shape
        ``broadcast(kx, ky) + (dim, dim)`` -- so the synthesized ``H_static``
        honors the momentum-broadcast contract of the batched solver paths
        without assuming the user's ``H_t`` itself broadcasts over momentum.
        """
        kx_arr = np.asarray(kx, dtype=float)
        ky_arr = np.asarray(ky, dtype=float)
        if kx_arr.ndim > 0 or ky_arr.ndim > 0:
            kx_b, ky_b = np.broadcast_arrays(kx_arr, ky_arr)
            stacked = np.array([
                self._compute_static_average(float(one_kx), float(one_ky))
                for one_kx, one_ky in zip(kx_b.ravel(), ky_b.ravel())
            ])
            return stacked.reshape(kx_b.shape + stacked.shape[-2:])

        cache_key = (float(kx_arr), float(ky_arr))
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

    
