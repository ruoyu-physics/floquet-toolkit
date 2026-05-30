"""Integrated current observables built from local Floquet velocities."""

from __future__ import annotations

import numpy as np

from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils.kspace import (
    create_cartesian_k_grid,
    create_circular_mask,
    create_polar_k_grid,
    integrate_cartesian_grid,
    integrate_polar_grid,
)
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
        time = np.linspace(
            0.0,
            self.period,
            self.floquet_params.n_time,
            endpoint=False,
        )
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

    def _build_integration_grid(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        grid_type: str = "polar",
    ):
        """Construct the integration grid and weights metadata."""
        if grid_type == "cartesian":
            kx_values, ky_values, kx_grid, ky_grid = create_cartesian_k_grid(
                kx_range=(k_center[0] - k_radius, k_center[0] + k_radius),
                ky_range=(k_center[1] - k_radius, k_center[1] + k_radius),
                num_kx=n_k_points,
                num_ky=n_k_points,
            )
            mask = create_circular_mask(
                kx_grid,
                ky_grid,
                k_radius=k_radius,
                k_center=k_center,
            )
            return {
                "kx_values": kx_values,
                "ky_values": ky_values,
                "kx_grid": kx_grid,
                "ky_grid": ky_grid,
                "mask": mask,
                "grid_type": grid_type,
            }

        if grid_type == "polar":
            r_values, theta_values, _, _, kx_grid, ky_grid = create_polar_k_grid(
                r_range=(0.0, k_radius),
                theta_range=(0.0, 2.0 * np.pi),
                num_r=n_k_points,
                num_theta=n_k_points,
                k_center=k_center,
            )
            return {
                "r_values": r_values,
                "theta_values": theta_values,
                "kx_grid": kx_grid,
                "ky_grid": ky_grid,
                "mask": None,
                "grid_type": grid_type,
            }

        raise ValueError(
            "grid_type must be either 'polar' or 'cartesian'."
        )

    def _integrate_local_current(
        self,
        current_fn,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        grid_type: str = "polar",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate a time-dependent local current function over a circular grid."""
        if grid_type == "adaptive_cartesian":
            return self._integrate_local_current_adaptive_cartesian(
                current_fn,
                k_radius=k_radius,
                k_center=k_center,
                n_k_points=n_k_points,
                adaptive_tol=adaptive_tol,
                adaptive_max_depth=adaptive_max_depth,
            )

        time = np.linspace(
            0.0,
            self.period,
            self.floquet_params.n_time,
            endpoint=False,
        )
        grid_data = self._build_integration_grid(
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            grid_type=grid_type,
        )
        kx_grid = grid_data["kx_grid"]
        ky_grid = grid_data["ky_grid"]

        if grid_type == "cartesian":
            kx_values = grid_data["kx_values"]
            ky_values = grid_data["ky_values"]
            mask = grid_data["mask"]
            local_jx = np.zeros((n_k_points, n_k_points, time.size), dtype=float)
            local_jy = np.zeros((n_k_points, n_k_points, time.size), dtype=float)

            for i in range(n_k_points):
                for j in range(n_k_points):
                    if not mask[i, j]:
                        continue
                    jx_t, jy_t = current_fn(time, kx_grid[i, j], ky_grid[i, j])
                    local_jx[i, j, :] = jx_t
                    local_jy[i, j, :] = jy_t

            integrated_jx = integrate_cartesian_grid(
                np.moveaxis(local_jx, -1, 0),
                kx_values,
                ky_values,
                mask=mask,
            )
            integrated_jy = integrate_cartesian_grid(
                np.moveaxis(local_jy, -1, 0),
                kx_values,
                ky_values,
                mask=mask,
            )
            return time, integrated_jx, integrated_jy

        if grid_type == "polar":
            r_values = grid_data["r_values"]
            theta_values = grid_data["theta_values"]
            local_jx = np.zeros((n_k_points, n_k_points, time.size), dtype=float)
            local_jy = np.zeros((n_k_points, n_k_points, time.size), dtype=float)

            for i in range(n_k_points):
                for j in range(n_k_points):
                    jx_t, jy_t = current_fn(time, kx_grid[i, j], ky_grid[i, j])
                    local_jx[i, j, :] = jx_t
                    local_jy[i, j, :] = jy_t

            integrated_jx = integrate_polar_grid(
                np.moveaxis(local_jx, -1, 0),
                r_values,
                theta_values,
            )
            integrated_jy = integrate_polar_grid(
                np.moveaxis(local_jy, -1, 0),
                r_values,
                theta_values,
            )
            return time, integrated_jx, integrated_jy

        raise ValueError(
            "grid_type must be one of 'polar', 'cartesian', or 'adaptive_cartesian'."
        )

    def _integrate_tracked_floquet_current(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        band="conduction",
        include_charge=False,
        band_selection_mode: str = "overlap",
        grid_type: str = "polar",
    ):
        """Integrate Floquet current using one branch tracked over the full grid."""
        time = np.linspace(
            0.0,
            self.period,
            self.floquet_params.n_time,
            endpoint=False,
        )
        grid_data = self._build_integration_grid(
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            grid_type=grid_type,
        )
        kx_grid = grid_data["kx_grid"]
        ky_grid = grid_data["ky_grid"]
        tracked = self.state_tracker.track_floquet_states_on_grid(
            kx_grid,
            ky_grid,
            band=band,
            init_mode=band_selection_mode,
        )
        tracked_states = tracked["floquet_states"]
        if self.cache is not None:
            self.cache.precompute_reconstructed_states_on_grid(
                self.velocity_calculator.state_provider,
                self.state_tracker,
                kx_grid,
                ky_grid,
                time,
                band=band,
                seed_indices=tracked["seed_indices"],
                init_mode=band_selection_mode,
            )
        local_jx = np.zeros((n_k_points, n_k_points, time.size), dtype=float)
        local_jy = np.zeros((n_k_points, n_k_points, time.size), dtype=float)

        if grid_type == "cartesian":
            mask = grid_data["mask"]
            for i in range(n_k_points):
                for j in range(n_k_points):
                    if not mask[i, j]:
                        continue
                    jx_t, jy_t = self.velocity_calculator.compute_floquet_velocity_from_state(
                        time,
                        kx_grid[i, j],
                        ky_grid[i, j],
                        tracked_states[i, j, :],
                        include_charge=include_charge,
                    )
                    local_jx[i, j, :] = jx_t
                    local_jy[i, j, :] = jy_t

            integrated_jx = integrate_cartesian_grid(
                np.moveaxis(local_jx, -1, 0),
                grid_data["kx_values"],
                grid_data["ky_values"],
                mask=mask,
            )
            integrated_jy = integrate_cartesian_grid(
                np.moveaxis(local_jy, -1, 0),
                grid_data["kx_values"],
                grid_data["ky_values"],
                mask=mask,
            )
            return time, integrated_jx, integrated_jy

        for i in range(n_k_points):
            for j in range(n_k_points):
                jx_t, jy_t = self.velocity_calculator.compute_floquet_velocity_from_state(
                    time,
                    kx_grid[i, j],
                    ky_grid[i, j],
                    tracked_states[i, j, :],
                    include_charge=include_charge,
                )
                local_jx[i, j, :] = jx_t
                local_jy[i, j, :] = jy_t

        integrated_jx = integrate_polar_grid(
            np.moveaxis(local_jx, -1, 0),
            grid_data["r_values"],
            grid_data["theta_values"],
        )
        integrated_jy = integrate_polar_grid(
            np.moveaxis(local_jy, -1, 0),
            grid_data["r_values"],
            grid_data["theta_values"],
        )
        return time, integrated_jx, integrated_jy

    def integrate_floquet_current_on_fermi_disk(
        self,
        k_radius: float,
        k_center=(0.0, 0.0),
        n_k_points: int = 21,
        band="conduction",
        include_charge=False,
        state_selection_algorithm: str = "tracked",
        band_selection_mode: str = "overlap",
        grid_type: str = "polar",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate the exact Floquet current over a circular momentum region.

        Args:
            k_radius: Radius of the circular integration region in momentum
                space.
            k_center: Center of the circular region.
            n_k_points: Number of samples per coordinate axis. For
                ``grid_type="cartesian"``, this is the number of grid
                points along each Cartesian axis before masking to the disk.
                For ``grid_type="polar"``, it is used for both the
                radial and angular sample counts. For
                ``grid_type="adaptive_cartesian"``, it sets the base cell
                width used for stopping refinement.
            band: Target band label or integer index for Floquet-state
                selection at each momentum.
            state_selection_algorithm: ``"tracked"`` uses the new grid-wide
                branch-tracking algorithm. ``"pointwise"`` uses the previous
                independent selection at each momentum point.
            band_selection_mode: State-selection rule passed to
                ``FloquetVelocityCalculator``.
            grid_type: Sampling/integration rule. ``"cartesian"`` uses
                a masked Cartesian grid over the enclosing square, while
                ``"polar"`` uses a polar grid with the appropriate Jacobian.
                ``"adaptive_cartesian"`` uses recursive midpoint refinement
                on the enclosing Cartesian square.

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)`` where the current
            components are evaluated on the default Floquet time grid.
        """
        if state_selection_algorithm == "tracked":
            if grid_type == "adaptive_cartesian":
                raise ValueError(
                    "grid_type='adaptive_cartesian' currently supports "
                    "state_selection_algorithm='pointwise' only."
                )
            return self._integrate_tracked_floquet_current(
                k_radius=k_radius,
                k_center=k_center,
                n_k_points=n_k_points,
                band=band,
                include_charge=include_charge,
                band_selection_mode=band_selection_mode,
                grid_type=grid_type,
            )

        if state_selection_algorithm != "pointwise":
            raise ValueError(
                "state_selection_algorithm must be either 'tracked' or 'pointwise'."
            )

        return self._integrate_local_current(
            lambda time, kx, ky: self.velocity_calculator.compute_floquet_velocity(
                time,
                kx,
                ky,
                band=band,
                band_selection_mode=band_selection_mode,
                include_charge=include_charge,
            ),
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            grid_type=grid_type,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )

    def integrate_adiabatic_current_on_fermi_disk(
        self,
        k_radius: float,
        k_center=(0.0, 0.0),
        n_k_points: int = 21,
        band="conduction",
        include_charge=False,
        grid_type: str = "polar",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate the adiabatic current over a circular momentum region."""
        return self._integrate_local_current(
            lambda time, kx, ky: np.asarray(
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
            ).T,
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            grid_type=grid_type,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )
