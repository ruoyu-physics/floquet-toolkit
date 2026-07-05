"""Integrated current observables built from local Floquet velocities."""

from __future__ import annotations

import numpy as np

from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils.kquadrature import KQuadrature
from .states import FloquetStateCache, FloquetStateTracker
from .floquet_velocity_calculator import FloquetVelocityCalculator


class FloquetCurrentCalculator:
    """Integrate local Floquet current over occupied regions in momentum space."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        cache: FloquetStateCache | None = None,
    ):
        """Initialize from one driven model and one Floquet parameter set."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.velocity_calculator = FloquetVelocityCalculator(
            driven_hamiltonian,
            floquet_params,
            cache=cache,
        )
        self.state_tracker = FloquetStateTracker(
            self.velocity_calculator.state_provider
        )
        self.period = driven_hamiltonian.period
        self.cache = cache

    @staticmethod
    def _adaptive_cache_key(kx: float, ky: float) -> tuple[float, float]:
        """Return a stable cache key for one sampled momentum."""
        return (round(float(kx), 15), round(float(ky), 15))

    @staticmethod
    def _cell_distance_bounds(
        kx_min: float,
        kx_max: float,
        ky_min: float,
        ky_max: float,
        k_center=(0.0, 0.0),
    ) -> tuple[float, float]:
        """Return the minimum and maximum distance from one cell to the disk center."""
        cx, cy = k_center
        closest_x = min(max(cx, kx_min), kx_max)
        closest_y = min(max(cy, ky_min), ky_max)
        min_distance = float(np.hypot(closest_x - cx, closest_y - cy))

        corner_distances = [
            float(np.hypot(kx_min - cx, ky_min - cy)),
            float(np.hypot(kx_min - cx, ky_max - cy)),
            float(np.hypot(kx_max - cx, ky_min - cy)),
            float(np.hypot(kx_max - cx, ky_max - cy)),
        ]
        max_distance = max(corner_distances)
        return min_distance, max_distance

    def _integrate_local_current_adaptive_cartesian(
        self,
        current_fn,
        k_radius: float,
        k_center=(0.0, 0.0),
        n_k_points: int = 21,
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate one local current using adaptive midpoint refinement."""
        time = self.floquet_params.time_grid(self.period)
        if n_k_points < 2:
            raise ValueError("adaptive_cartesian requires n_k_points >= 2.")

        if adaptive_tol is None:
            adaptive_tol = 1.0e-3
        if adaptive_max_depth is None:
            adaptive_max_depth = 6

        cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        zero_trace = np.zeros(time.size, dtype=float)
        base_cell_width = 2.0 * k_radius / (n_k_points - 1)

        def current_at_point(kx: float, ky: float) -> tuple[np.ndarray, np.ndarray]:
            key = self._adaptive_cache_key(kx, ky)
            cached = cache.get(key)
            if cached is not None:
                return cached
            jx_t, jy_t = current_fn(time, kx, ky)
            value = (np.asarray(jx_t, dtype=float), np.asarray(jy_t, dtype=float))
            cache[key] = value
            return value

        def integrate_cell(
            kx_min: float,
            kx_max: float,
            ky_min: float,
            ky_max: float,
            depth: int,
        ) -> tuple[np.ndarray, np.ndarray]:
            min_distance, max_distance = self._cell_distance_bounds(
                kx_min,
                kx_max,
                ky_min,
                ky_max,
                k_center=k_center,
            )
            if min_distance >= k_radius:
                return zero_trace.copy(), zero_trace.copy()

            width_x = kx_max - kx_min
            width_y = ky_max - ky_min
            area = width_x * width_y
            center_kx = 0.5 * (kx_min + kx_max)
            center_ky = 0.5 * (ky_min + ky_max)
            coarse_jx_t, coarse_jy_t = current_at_point(center_kx, center_ky)
            coarse_estimate_jx = coarse_jx_t * area
            coarse_estimate_jy = coarse_jy_t * area

            if depth >= adaptive_max_depth:
                return coarse_estimate_jx, coarse_estimate_jy

            half_x = 0.5 * width_x
            half_y = 0.5 * width_y
            quarter_area = 0.25 * area
            child_centers = [
                (kx_min + 0.5 * half_x, ky_min + 0.5 * half_y),
                (kx_min + 0.5 * half_x, ky_min + 1.5 * half_y),
                (kx_min + 1.5 * half_x, ky_min + 0.5 * half_y),
                (kx_min + 1.5 * half_x, ky_min + 1.5 * half_y),
            ]
            fine_estimate_jx = np.zeros(time.size, dtype=float)
            fine_estimate_jy = np.zeros(time.size, dtype=float)
            for child_kx, child_ky in child_centers:
                child_jx_t, child_jy_t = current_at_point(child_kx, child_ky)
                fine_estimate_jx += child_jx_t * quarter_area
                fine_estimate_jy += child_jy_t * quarter_area

            error_trace = np.hypot(
                fine_estimate_jx - coarse_estimate_jx,
                fine_estimate_jy - coarse_estimate_jy,
            )
            cell_error = float(np.max(error_trace))
            cell_is_fully_inside = max_distance <= k_radius
            cell_is_small = max(width_x, width_y) <= 0.5 * base_cell_width
            if cell_is_fully_inside and cell_error <= adaptive_tol:
                return fine_estimate_jx, fine_estimate_jy
            if cell_is_small and cell_error <= adaptive_tol:
                return fine_estimate_jx, fine_estimate_jy

            kx_mid = 0.5 * (kx_min + kx_max)
            ky_mid = 0.5 * (ky_min + ky_max)
            quadrants = [
                (kx_min, kx_mid, ky_min, ky_mid),
                (kx_min, kx_mid, ky_mid, ky_max),
                (kx_mid, kx_max, ky_min, ky_mid),
                (kx_mid, kx_max, ky_mid, ky_max),
            ]
            refined_jx = np.zeros(time.size, dtype=float)
            refined_jy = np.zeros(time.size, dtype=float)
            for quadrant in quadrants:
                child_jx, child_jy = integrate_cell(*quadrant, depth + 1)
                refined_jx += child_jx
                refined_jy += child_jy
            return refined_jx, refined_jy

        integrated_jx, integrated_jy = integrate_cell(
            k_center[0] - k_radius,
            k_center[0] + k_radius,
            k_center[1] - k_radius,
            k_center[1] + k_radius,
            0,
        )
        return time, integrated_jx, integrated_jy

    def _current_velocity_map(
        self,
        time,
        quadrature: KQuadrature,
        kind: str,
        band,
        include_charge: bool,
        band_selection_mode: str,
        state_selection_algorithm: str,
    ):
        """Build the ``(jx_map, jy_map)`` local-current map for ``kind``.

        The map is evaluated on the quadrature's sample points and shares its
        grid shape, so it can be reduced directly with ``quadrature.integrate``.
        """
        kx_grid = quadrature.kx_grid
        ky_grid = quadrature.ky_grid
        if kind == "floquet":
            if state_selection_algorithm == "tracked":
                tracked = self.state_tracker.track_floquet_states_on_grid(
                    kx_grid,
                    ky_grid,
                    band=band,
                    init_mode=band_selection_mode,
                )
                return (
                    self.velocity_calculator.compute_floquet_velocity_map_from_states(
                        time,
                        kx_grid,
                        ky_grid,
                        tracked["floquet_states"],
                        include_charge=include_charge,
                    )
                )
            if state_selection_algorithm == "pointwise":
                return self.velocity_calculator.compute_floquet_velocity_map(
                    time,
                    kx_grid,
                    ky_grid,
                    band=band,
                    include_charge=include_charge,
                    band_selection_mode=band_selection_mode,
                )
            raise ValueError(
                "state_selection_algorithm must be either 'tracked' or 'pointwise'."
            )
        if kind == "adiabatic":
            return self.velocity_calculator.compute_adiabatic_velocity_map(
                time,
                kx_grid,
                ky_grid,
                band=band,
                include_charge=include_charge,
            )
        raise ValueError("kind must be 'floquet' or 'adiabatic'.")

    def integrate_current(
        self,
        quadrature: KQuadrature,
        kind: str = "floquet",
        band="conduction",
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
        state_selection_algorithm: str = "pointwise",
    ):
        """Integrate a local current over an explicit k-space quadrature.

        This is the core integration entry point. Supply any
        :class:`~floquet_toolkit.utils.kquadrature.KQuadrature` -- a built-in
        :meth:`KQuadrature.polar`/:meth:`KQuadrature.cartesian`, or a custom one
        for a non-disk region or model-specific sampling -- and select which
        current ``kind`` to integrate. The ``*_on_fermi_disk`` methods are thin
        wrappers that build a disk quadrature and call this.

        Args:
            quadrature: Sample points and local integration measure (weights).
            kind: Which local current to integrate. ``"floquet"`` uses the exact
                Floquet velocity; ``"adiabatic"`` uses the instantaneous-band
                velocity.
            band: Target band label or integer index.
            include_charge: Multiply the velocity by the charge to return a
                charge current.
            band_selection_mode: State-selection rule (``"floquet"`` only).
            state_selection_algorithm: ``"tracked"`` (grid-wide branch tracking)
                or ``"pointwise"`` (independent per-momentum selection).
                ``"floquet"`` only; ignored for ``"adiabatic"``.

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)`` on the default Floquet
            time grid.
        """
        time = self.floquet_params.time_grid(self.period)
        jx_map, jy_map = self._current_velocity_map(
            time,
            quadrature,
            kind,
            band,
            include_charge,
            band_selection_mode,
            state_selection_algorithm,
        )
        return time, quadrature.integrate(jx_map), quadrature.integrate(jy_map)

    def integrate_adaptive_current(
        self,
        k_radius: float,
        kind: str = "floquet",
        k_center=(0.0, 0.0),
        n_k_points: int = 21,
        band="conduction",
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
        state_selection_algorithm: str = "pointwise",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate a local current with adaptive Cartesian refinement.

        This is the separate, data-dependent integration engine: the sample
        points are chosen by recursive midpoint refinement of the integrand, so
        the region is specified directly by ``k_radius``/``k_center`` rather than
        by a precomputed quadrature. Use :meth:`integrate_current` for the
        fixed-quadrature case.

        Args:
            k_radius: Radius of the circular integration region.
            kind: Which local current to integrate (``"floquet"`` or
                ``"adiabatic"``).
            k_center: Center of the circular region.
            n_k_points: Sets the base cell width used to stop refinement.
            band: Target band label or integer index.
            include_charge: Multiply the velocity by the charge to return a
                charge current.
            band_selection_mode: State-selection rule (``"floquet"`` only).
            state_selection_algorithm: ``"floquet"`` only; adaptive refinement
                supports ``"pointwise"`` selection (the data-dependent grid is
                incompatible with grid-wide branch tracking).
            adaptive_tol: Per-cell error tolerance for refinement.
            adaptive_max_depth: Maximum recursion depth.

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)`` on the default Floquet
            time grid.
        """
        if kind == "floquet":
            if state_selection_algorithm != "pointwise":
                raise ValueError(
                    "Adaptive integration supports "
                    "state_selection_algorithm='pointwise' only."
                )

            def current_fn(time, kx, ky):
                return self.velocity_calculator.compute_floquet_velocity(
                    time,
                    kx,
                    ky,
                    band=band,
                    band_selection_mode=band_selection_mode,
                    include_charge=include_charge,
                )

        elif kind == "adiabatic":

            def current_fn(time, kx, ky):
                return np.asarray(
                    [
                        self.velocity_calculator.compute_adiabatic_velocity(
                            t,
                            kx,
                            ky,
                            band=band,
                            include_charge=include_charge,
                        )
                        for t in time
                    ]
                ).T

        else:
            raise ValueError("kind must be 'floquet' or 'adiabatic'.")

        return self._integrate_local_current_adaptive_cartesian(
            current_fn,
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )
