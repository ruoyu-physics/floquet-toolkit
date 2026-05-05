import numpy as np

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

    def _create_k_grid_for_circular_zone(self, k_radius, k_center=(0, 0), n_points=51):
        """Create a masked Cartesian grid inside a circular region."""
        kx_values = np.linspace(
            k_center[0] - k_radius,
            k_center[0] + k_radius,
            n_points,
        )
        ky_values = np.linspace(
            k_center[1] - k_radius,
            k_center[1] + k_radius,
            n_points,
        )

        kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing="xy")
        mask = (
            (kx_grid - k_center[0]) ** 2 + (ky_grid - k_center[1]) ** 2
            <= k_radius**2
        )

        return kx_grid, ky_grid, mask

    def integrate_curvature_over_kgrid(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=51,
        band="conduction",
    ):
        """Integrate instantaneous Berry curvature over a circular zone.

        Args:
            time: Single time or time grid passed through to the Floquet-state
                reconstruction used in the local curvature calculation.
            k_radius: Radius of the circular integration region.
            k_center: Center of the circular integration region in momentum
                space.
            n_points: Number of Cartesian samples along each axis before masking.
            band: Target band label or integer index.
        """
        kx_grid, ky_grid, mask = self._create_k_grid_for_circular_zone(
            k_radius=k_radius,
            k_center=k_center,
            n_points=n_points,
        )

        dk = 2.0 * k_radius / (n_points - 1)
        total_curvature = 0.0

        for kx, ky in zip(kx_grid[mask], ky_grid[mask]):
            total_curvature += self.curvature_calculator.compute_instantaneous_berry_curvature(
                time,
                kx,
                ky,
                band=band,
            )

        return total_curvature * dk * dk

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
