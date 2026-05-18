"""Integrated current observables built from local Floquet velocities."""

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
from .floquet_velocity_calculator import FloquetVelocityCalculator


class FloquetCurrentCalculator:
    """Integrate local Floquet current over occupied regions in momentum space."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        """Initialize from one driven model and one Floquet parameter set."""
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.velocity_calculator = FloquetVelocityCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.period = driven_hamiltonian.period

    def _build_integration_grid(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        integration_mode: str = "polar",
    ):
        """Construct the integration grid and weights metadata."""
        if integration_mode == "cartesian":
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
                "integration_mode": integration_mode,
            }

        if integration_mode == "polar":
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
                "integration_mode": integration_mode,
            }

        raise ValueError(
            "integration_mode must be either 'polar' or 'cartesian'."
        )

    def _integrate_local_current(
        self,
        current_fn,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        integration_mode: str = "polar",
    ):
        """Integrate a time-dependent local current function over a circular grid."""
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
            integration_mode=integration_mode,
        )
        kx_grid = grid_data["kx_grid"]
        ky_grid = grid_data["ky_grid"]

        if integration_mode == "cartesian":
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

        if integration_mode == "polar":
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
            "integration_mode must be either 'polar' or 'cartesian'."
        )

    def _integrate_tracked_floquet_current(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        band="conduction",
        include_charge=False,
        band_selection_mode: str = "overlap",
        integration_mode: str = "polar",
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
            integration_mode=integration_mode,
        )
        kx_grid = grid_data["kx_grid"]
        ky_grid = grid_data["ky_grid"]
        tracked = self.velocity_calculator.state_provider.track_floquet_states_on_grid(
            kx_grid,
            ky_grid,
            band=band,
            init_mode=band_selection_mode,
        )
        tracked_states = tracked["floquet_states"]
        local_jx = np.zeros((n_k_points, n_k_points, time.size), dtype=float)
        local_jy = np.zeros((n_k_points, n_k_points, time.size), dtype=float)

        if integration_mode == "cartesian":
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
        integration_mode: str = "polar",
    ):
        """Integrate the exact Floquet current over a circular momentum region.

        Args:
            k_radius: Radius of the circular integration region in momentum
                space.
            k_center: Center of the circular region.
            n_k_points: Number of samples per coordinate axis. For
                ``integration_mode="cartesian"``, this is the number of grid
                points along each Cartesian axis before masking to the disk.
                For ``integration_mode="polar"``, it is used for both the
                radial and angular sample counts.
            band: Target band label or integer index for Floquet-state
                selection at each momentum.
            state_selection_algorithm: ``"tracked"`` uses the new grid-wide
                branch-tracking algorithm. ``"pointwise"`` uses the previous
                independent selection at each momentum point.
            band_selection_mode: State-selection rule passed to
                ``FloquetVelocityCalculator``.
            integration_mode: Sampling/integration rule. ``"cartesian"`` uses
                a masked Cartesian grid over the enclosing square, while
                ``"polar"`` uses a polar grid with the appropriate Jacobian.

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)`` where the current
            components are evaluated on the default Floquet time grid.
        """
        if state_selection_algorithm == "tracked":
            return self._integrate_tracked_floquet_current(
                k_radius=k_radius,
                k_center=k_center,
                n_k_points=n_k_points,
                band=band,
                include_charge=include_charge,
                band_selection_mode=band_selection_mode,
                integration_mode=integration_mode,
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
            integration_mode=integration_mode,
        )

    def integrate_adiabatic_current_on_fermi_disk(
        self,
        k_radius: float,
        k_center=(0.0, 0.0),
        n_k_points: int = 21,
        band="conduction",
        include_charge=False,
        integration_mode: str = "polar",
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
            integration_mode=integration_mode,
        )
