import numpy as np

from ..utils.kspace import (
    create_cartesian_k_grid,
    create_circular_mask,
    create_polar_k_grid,
    integrate_cartesian_grid,
    integrate_polar_grid,
)
from .floquet_curvature_calculator import FloquetCurvatureCalculator
from .floquet_state_provider import FloquetStateProvider


class FloquetBerryPhaseCalculator:
    """Integrate curvature- and Berry-phase-related quantities over k-space."""

    def __init__(self, driven_hamiltonian, floquet_params):
        """Initialize from one driven model and one Floquet parameter set."""
        self.curvature_calculator = FloquetCurvatureCalculator(
            driven_hamiltonian,
            floquet_params,
        )
        self.driven_hamiltonian = driven_hamiltonian
        self.state_provider = FloquetStateProvider(driven_hamiltonian, floquet_params)

    def integrate_curvature_over_kgrid(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=51,
        band="conduction",
        integration_mode: str = "polar",
    ):
        """Integrate instantaneous Berry curvature over a circular zone.

        Args:
            time: Single time or time grid passed through to the Floquet-state
                reconstruction used in the local curvature calculation.
            k_radius: Radius of the circular integration region.
            k_center: Center of the circular integration region in momentum
                space.
            n_points: Number of samples per coordinate axis. For
                ``integration_mode="cartesian"``, this is the number of grid
                points along each Cartesian axis before masking to the disk.
                For ``integration_mode="polar"``, it is used for both the
                radial and angular sample counts.
            band: Target band label or integer index.
            integration_mode: Sampling/integration rule. ``"cartesian"`` uses
                a masked Cartesian grid over the enclosing square, while
                ``"polar"`` uses a polar grid with the appropriate Jacobian.
        """
        if integration_mode == "cartesian":
            kx_values, ky_values, kx_grid, ky_grid = create_cartesian_k_grid(
                kx_range=(k_center[0] - k_radius, k_center[0] + k_radius),
                ky_range=(k_center[1] - k_radius, k_center[1] + k_radius),
                num_kx=n_points,
                num_ky=n_points,
            )
            mask = create_circular_mask(
                kx_grid,
                ky_grid,
                k_radius=k_radius,
                k_center=k_center,
            )
            curvature_values = None

            for i in range(n_points):
                for j in range(n_points):
                    if not mask[i, j]:
                        continue
                    local_curvature = np.asarray(
                        self.curvature_calculator.compute_instantaneous_berry_curvature(
                            time,
                            kx_grid[i, j],
                            ky_grid[i, j],
                            band=band,
                        )
                    )
                    if curvature_values is None:
                        curvature_values = np.zeros(
                            local_curvature.shape + (n_points, n_points),
                            dtype=local_curvature.dtype,
                        )
                    curvature_values[..., i, j] = local_curvature

            if curvature_values is None:
                curvature_values = np.zeros((n_points, n_points), dtype=float)

            return integrate_cartesian_grid(
                curvature_values,
                kx_values,
                ky_values,
                mask=mask,
            )

        if integration_mode == "polar":
            r_values, theta_values, _, _, kx_grid, ky_grid = create_polar_k_grid(
                r_range=(0.0, k_radius),
                theta_range=(0.0, 2.0 * np.pi),
                num_r=n_points,
                num_theta=n_points,
                k_center=k_center,
            )
            curvature_values = None

            for i in range(n_points):
                for j in range(n_points):
                    local_curvature = np.asarray(
                        self.curvature_calculator.compute_instantaneous_berry_curvature(
                            time,
                            kx_grid[i, j],
                            ky_grid[i, j],
                            band=band,
                        )
                    )
                    if curvature_values is None:
                        curvature_values = np.zeros(
                            local_curvature.shape + (n_points, n_points),
                            dtype=local_curvature.dtype,
                        )
                    curvature_values[..., i, j] = local_curvature

            if curvature_values is None:
                curvature_values = np.zeros((n_points, n_points), dtype=float)

            return integrate_polar_grid(
                curvature_values,
                r_values,
                theta_values,
            )

        raise ValueError(
            "integration_mode must be either 'polar' or 'cartesian'."
        )

    def _compute_berry_phase_on_circle(
        self,
        bloch_state,
        k_radius,
        k_center=(0, 0),
        n_points=101,
        band="conduction",
    ):
        """Compute a Berry phase on a circular loop using link variables.

        Args:
            bloch_state: Callable returning a normalized state on the loop.
            k_center: Center of the circular loop.
            k_radius: Radius of the circular loop.
            n_points: Number of sampled points on the loop.
            band: Target band label or index for state selection.
        Returns:
            Berry phase accumulated around the loop. For time-resolved Floquet
            states this returns one phase per time sample.
        """
        # Create a clockwise circular path in k-space
        k_angles = np.linspace(0, -2 * np.pi, n_points, endpoint=False)
        kx_circle = k_center[0] + k_radius * np.cos(k_angles)
        ky_circle = k_center[1] + k_radius * np.sin(k_angles)

        curvature_calculator = self.curvature_calculator

        state_k_cache = []
        for i in range(n_points):
            kx, ky = kx_circle[i], ky_circle[i]
            state = bloch_state(kx, ky, band=band)
            state_k_cache.append(state)

        berry_phase = 1 + 0j
        for i in range(n_points):
            state1 = state_k_cache[i]
            state2 = state_k_cache[(i + 1) % n_points]
            berry_phase *= curvature_calculator._compute_link_variable(state1, state2)
        return np.angle(berry_phase)

    def compute_static_berry_phase(
        self,
        k_radius,
        k_center=(0, 0),
        n_points=101,
        band="conduction",
    ):
        """Compute the static-band Berry phase on a circular momentum-space loop."""

        def static_bloch_state(kx, ky, band="conduction"):
            """Helper to select the static Bloch state at one momentum."""

            static_energy, static_states = (
                self.state_provider.diagonalize_static_hamiltonian(kx, ky)
            )
            target_idx = self.state_provider.resolve_band_index(static_energy, band)
            return static_states[:, target_idx]

        return self._compute_berry_phase_on_circle(
            bloch_state=static_bloch_state,
            k_radius=k_radius,
            k_center=k_center,
            n_points=n_points,
            band=band,
        )

    def compute_floquet_berry_phase(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=101,
        band="conduction",
    ):
        """Compute the Floquet-mode Berry phase on a circular momentum-space loop.

        Args:
            time: Single time or time grid used to reconstruct the Floquet mode.
            k_radius: Radius of the circular loop.
            k_center: Center of the circular loop in momentum space.
            n_points: Number of sampled points on the loop.
            band: Target band label or integer index.
        """

        def floquet_bloch_state(kx, ky, band="conduction"):
            """Select and reconstruct the time-dependent Floquet state."""
            idx, _, f_state = self.state_provider.select_floquet_state(
                kx,
                ky,
                band=band,
            )
            
            return self.state_provider.reconstruct_floquet_state(f_state, time=time)

        return self._compute_berry_phase_on_circle(
            bloch_state=floquet_bloch_state,
            k_radius=k_radius,
            k_center=k_center,
            n_points=n_points,
            band=band,
        )
