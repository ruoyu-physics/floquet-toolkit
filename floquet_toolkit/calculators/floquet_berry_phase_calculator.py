import numpy as np
from functools import partial

from ..utils.kspace import (
    create_cartesian_k_grid,
    create_circular_mask,
    create_polar_k_grid,
    integrate_cartesian_grid,
    integrate_polar_grid,
)
from ..builders import FloquetBuilder, HFEBuilder
from .floquet_curvature_calculator import FloquetCurvatureCalculator
from .states import FloquetStateCache, FloquetStateProvider, FloquetStateTracker


class FloquetBerryPhaseCalculator:
    """Integrate curvature- and Berry-phase-related quantities over k-space."""

    def __init__(
        self,
        driven_hamiltonian,
        floquet_params,
        cache: FloquetStateCache | None = None,
    ):
        """Initialize from one driven model and one Floquet parameter set."""
        self.curvature_calculator = FloquetCurvatureCalculator(
            driven_hamiltonian,
            floquet_params,
            cache=cache,
        )
        self.driven_hamiltonian = driven_hamiltonian
        self.state_provider = FloquetStateProvider(
            driven_hamiltonian,
            floquet_params,
            cache=cache,
        )
        self.state_tracker = FloquetStateTracker(self.state_provider)

    def _build_shared_corner_disk_lattice(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_plaquettes=51,
    ):
        """Build a shared-corner Cartesian plaquette lattice for a circular region."""
        corner_values_x = np.linspace(
            k_center[0] - k_radius,
            k_center[0] + k_radius,
            n_plaquettes + 1,
        )
        corner_values_y = np.linspace(
            k_center[1] - k_radius,
            k_center[1] + k_radius,
            n_plaquettes + 1,
        )
        kx_corner_grid, ky_corner_grid = np.meshgrid(
            corner_values_x,
            corner_values_y,
            indexing="ij",
        )
        kx_centers = 0.5 * (corner_values_x[:-1] + corner_values_x[1:])
        ky_centers = 0.5 * (corner_values_y[:-1] + corner_values_y[1:])
        kx_center_grid, ky_center_grid = np.meshgrid(
            kx_centers,
            ky_centers,
            indexing="ij",
        )
        mask = create_circular_mask(
            kx_center_grid,
            ky_center_grid,
            k_radius=k_radius,
            k_center=k_center,
        )
        return kx_corner_grid, ky_corner_grid, mask

    def _select_shared_corner_seed_indices(
        self,
        mask,
        kx_corner_grid,
        ky_corner_grid,
        k_center=(0.0, 0.0),
    ):
        """Pick a boundary seed corner that belongs to the integrated disk."""
        mask = np.asarray(mask, dtype=bool)
        center_x, center_y = k_center
        boundary_candidates = []

        for i, j in np.argwhere(mask):
            is_boundary = (
                i == 0
                or j == 0
                or i == mask.shape[0] - 1
                or j == mask.shape[1] - 1
                or not mask[i - 1, j]
                or not mask[i + 1, j]
                or not mask[i, j - 1]
                or not mask[i, j + 1]
            )
            if not is_boundary:
                continue

            corner_indices = [
                (int(i), int(j)),
                (int(i + 1), int(j)),
                (int(i), int(j + 1)),
                (int(i + 1), int(j + 1)),
            ]
            outer_corner = max(
                corner_indices,
                key=lambda idx: float(
                    (kx_corner_grid[idx] - center_x) ** 2
                    + (ky_corner_grid[idx] - center_y) ** 2
                ),
            )
            boundary_candidates.append(
                (
                    float(
                        (kx_corner_grid[outer_corner] - center_x) ** 2
                        + (ky_corner_grid[outer_corner] - center_y) ** 2
                    ),
                    outer_corner,
                )
            )

        if not boundary_candidates:
            raise ValueError(
                "Could not find a masked boundary plaquette for shared-corner integration."
            )

        boundary_candidates.sort(reverse=True)
        return boundary_candidates[0][1]

    def _track_reconstructed_states_on_shared_corner_lattice(
        self,
        time,
        k_radius,
        k_center=(0.0, 0.0),
        n_plaquettes=51,
        band="conduction",
    ):
        """Track one Floquet branch on a shared-corner disk lattice and reconstruct it."""
        kx_corner_grid, ky_corner_grid, mask = self._build_shared_corner_disk_lattice(
            k_radius=k_radius,
            k_center=k_center,
            n_plaquettes=n_plaquettes,
        )
        seed_indices = self._select_shared_corner_seed_indices(
            mask,
            kx_corner_grid,
            ky_corner_grid,
            k_center=k_center,
        )
        tracked = self.state_tracker.track_floquet_states_on_grid(
            kx_corner_grid,
            ky_corner_grid,
            band=band,
            seed_indices=seed_indices,
        )
        tracked_states = tracked["floquet_states"]
        reconstructed_states = np.empty(
            tracked_states.shape[:2] + np.asarray(
                self.state_provider.reconstruct_floquet_state(
                    tracked_states[0, 0, :],
                    time=time,
                )
            ).shape,
            dtype=complex,
        )
        for i in range(tracked_states.shape[0]):
            for j in range(tracked_states.shape[1]):
                reconstructed_states[i, j, ...] = self.state_provider.reconstruct_floquet_state(
                    tracked_states[i, j, :],
                    time=time,
                )
        return reconstructed_states, mask

    def _build_state_grid_on_shared_corner_lattice(
        self,
        state_getter,
        k_radius,
        k_center=(0.0, 0.0),
        n_plaquettes=51,
    ):
        """Evaluate one state getter on every corner of a shared-corner disk lattice."""
        kx_corner_grid, ky_corner_grid, mask = self._build_shared_corner_disk_lattice(
            k_radius=k_radius,
            k_center=k_center,
            n_plaquettes=n_plaquettes,
        )
        sample_state = np.asarray(
            state_getter(
                float(kx_corner_grid[0, 0]),
                float(ky_corner_grid[0, 0]),
            )
        )
        state_grid = np.empty(
            kx_corner_grid.shape + sample_state.shape,
            dtype=complex,
        )
        for i in range(kx_corner_grid.shape[0]):
            for j in range(ky_corner_grid.shape[1]):
                state_grid[i, j, ...] = state_getter(
                    float(kx_corner_grid[i, j]),
                    float(ky_corner_grid[i, j]),
                )
        return state_grid, mask

    def _integrate_state_grid_on_shared_corner_lattice(
        self,
        state_grid,
        mask,
        return_wilson_surface_phase: bool = False,
    ):
        """Integrate one shared-corner state grid by local plaquette phases."""
        plaquette_product = None
        local_phase_sum = None
        for i, j in np.argwhere(mask):
            v1 = state_grid[i, j, ...]
            v2 = state_grid[i + 1, j, ...]
            v3 = state_grid[i, j + 1, ...]
            v4 = state_grid[i + 1, j + 1, ...]
            ux = self.curvature_calculator._compute_link_variable(v1, v2)
            uy_dkx = self.curvature_calculator._compute_link_variable(v2, v4)
            ux_dky = self.curvature_calculator._compute_link_variable(v3, v4)
            uy = self.curvature_calculator._compute_link_variable(v1, v3)
            plaquette_factor = ux * uy_dkx / ux_dky / uy
            if plaquette_product is None:
                plaquette_product = np.ones_like(plaquette_factor, dtype=complex)
                local_phase_sum = np.zeros_like(np.angle(plaquette_factor), dtype=float)
            plaquette_product *= plaquette_factor
            local_phase_sum += np.angle(plaquette_factor)

        if plaquette_product is None:
            return np.zeros((), dtype=float)
        if return_wilson_surface_phase:
            return np.angle(plaquette_product)
        return local_phase_sum

    def _compute_local_curvature(
        self,
        curvature_type,
        time,
        kx,
        ky,
        band="conduction",
        dk=1e5,
        order: int = 2,
    ):
        """Dispatch one local Berry-curvature evaluation by curvature source."""
        if curvature_type == "floquet":
            return self.curvature_calculator.compute_instantaneous_berry_curvature(
                time,
                kx,
                ky,
                band=band,
                dk=dk,
            )
        if curvature_type == "static":
            return self.curvature_calculator.compute_static_berry_curvature(
                kx,
                ky,
                band=band,
                dk=dk,
                method="numeric",
            )
        if curvature_type == "perturbed":
            return self.curvature_calculator.compute_perturbed_state_berry_curvature(
                time,
                kx,
                ky,
                band=band,
                dk=dk,
                order=order,
            )
        if curvature_type == "hfe":
            return self.curvature_calculator.compute_hfe_berry_curvature(
                kx,
                ky,
                band=band,
                dk=dk,
                order=order,
            )
        raise ValueError(
            "curvature_type must be one of 'floquet', 'static', 'perturbed', or 'hfe'."
        )

    def _integrate_curvature_over_shared_corner_lattice(
        self,
        time,
        k_radius,
        k_center=(0.0, 0.0),
        n_plaquettes=51,
        band="conduction",
        curvature_type: str = "floquet",
        order: int = 2,
        return_wilson_surface_phase: bool = False,
    ):
        """Integrate Berry curvature on one tracked shared-corner disk lattice."""
        if curvature_type == "floquet":
            state_grid, mask = self._track_reconstructed_states_on_shared_corner_lattice(
                time=time,
                k_radius=k_radius,
                k_center=k_center,
                n_plaquettes=n_plaquettes,
                band=band,
            )
            return self._integrate_state_grid_on_shared_corner_lattice(
                state_grid,
                mask,
                return_wilson_surface_phase=return_wilson_surface_phase,
            )

        if curvature_type == "static":
            def static_state(kx, ky):
                energy, states = self.state_provider.diagonalize_static_hamiltonian(kx, ky)
                band_index = self.state_provider.resolve_band_index(energy, band)
                return states[:, band_index]

            state_grid, mask = self._build_state_grid_on_shared_corner_lattice(
                static_state,
                k_radius=k_radius,
                k_center=k_center,
                n_plaquettes=n_plaquettes,
            )
            return self._integrate_state_grid_on_shared_corner_lattice(
                state_grid,
                mask,
                return_wilson_surface_phase=return_wilson_surface_phase,
            )

        if curvature_type == "perturbed":
            def perturbed_state(kx, ky):
                _, state = self.curvature_calculator.perturbation_calculator.compute_perturbed_state(
                    kx,
                    ky,
                    band=band,
                    order=order,
                )
                return self.state_provider.reconstruct_floquet_state(state, time=time)

            state_grid, mask = self._build_state_grid_on_shared_corner_lattice(
                perturbed_state,
                k_radius=k_radius,
                k_center=k_center,
                n_plaquettes=n_plaquettes,
            )
            return self._integrate_state_grid_on_shared_corner_lattice(
                state_grid,
                mask,
                return_wilson_surface_phase=return_wilson_surface_phase,
            )

        if curvature_type == "hfe":
            def hfe_state(kx, ky):
                floquet_builder = FloquetBuilder(
                    partial(self.curvature_calculator.Ht, kx=kx, ky=ky),
                    self.curvature_calculator.omega,
                    self.curvature_calculator.hbar,
                    self.curvature_calculator.floquet_params,
                )
                hfe_builder = HFEBuilder(floquet_builder)
                h_eff = hfe_builder.compute_hfe_hamiltonian(order=order)
                eigvals, eigvecs = np.linalg.eigh(h_eff)
                band_index = self.state_provider.resolve_band_index(eigvals, band)
                return eigvecs[:, band_index]

            state_grid, mask = self._build_state_grid_on_shared_corner_lattice(
                hfe_state,
                k_radius=k_radius,
                k_center=k_center,
                n_plaquettes=n_plaquettes,
            )
            return self._integrate_state_grid_on_shared_corner_lattice(
                state_grid,
                mask,
                return_wilson_surface_phase=return_wilson_surface_phase,
            )

        raise ValueError(
            "curvature_type must be one of 'floquet', 'static', 'perturbed', or 'hfe'."
        )

    def integrate_curvature_over_kgrid(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=51,
        band="conduction",
        grid_type: str = "cartesian",
        dk: float | None = None,
        integration_method: str = "shared_corner",
        curvature_type: str = "floquet",
        order: int = 2,
    ):
        """Integrate Berry curvature over a circular zone.

        Args:
            time: Single time or time grid used by time-dependent curvature
                sources. Ignored for static and HFE integrations.
            k_radius: Radius of the circular integration region.
            k_center: Center of the circular integration region in momentum
                space.
            n_points: Number of samples per coordinate axis. For
                ``grid_type="cartesian"``, this is the number of grid
                points along each Cartesian axis before masking to the disk.
                For ``grid_type="polar"``, it is used for both the
                radial and angular sample counts.
            band: Target band label or integer index.
            grid_type: Sampling/integration rule. ``"cartesian"`` uses
                a masked Cartesian grid over the enclosing square, while
                ``"polar"`` uses a polar grid with the appropriate Jacobian.
            dk: Plaquette width used in the local Berry-curvature evaluation.
                If omitted, use the calculator default.
            integration_method: ``"shared_corner"`` tracks one branch on a
                shared-corner Cartesian plaquette lattice and sums local
                plaquette phases on that same lattice. ``"pointwise"``
                evaluates local curvature independently at each sample point.
            curvature_type: Which curvature source to integrate:
                ``"floquet"``, ``"static"``, ``"perturbed"``, or ``"hfe"``.
            order: Expansion order for ``curvature_type="perturbed"`` or
                ``curvature_type="hfe"``.
        """
        if dk is None:
            dk = self.curvature_calculator.floquet_params.dk

        if integration_method == "shared_corner":
            if grid_type != "cartesian":
                raise ValueError(
                    "integration_method='shared_corner' currently requires "
                    "grid_type='cartesian'."
                )
            return self._integrate_curvature_over_shared_corner_lattice(
                time=time,
                k_radius=k_radius,
                k_center=k_center,
                n_plaquettes=n_points,
                band=band,
                curvature_type=curvature_type,
                order=order,
            )

        if integration_method != "pointwise":
            raise ValueError(
                "integration_method must be either 'pointwise' or 'shared_corner'."
            )

        if grid_type == "cartesian":
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
                        self._compute_local_curvature(
                            curvature_type,
                            time,
                            kx_grid[i, j],
                            ky_grid[i, j],
                            band=band,
                            dk=dk,
                            order=order,
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

        if grid_type == "polar":
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
                        self._compute_local_curvature(
                            curvature_type,
                            time,
                            kx_grid[i, j],
                            ky_grid[i, j],
                            band=band,
                            dk=dk,
                            order=order,
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
            "grid_type must be either 'polar' or 'cartesian'."
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
        # Create a counter-clockwise circular path in k-space
        k_angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
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

    def _track_floquet_states_on_circle(
        self,
        k_radius,
        k_center=(0, 0),
        n_points=101,
        band="conduction",
    ):
        """Track one Floquet branch continuously around a circular loop."""
        k_angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        kx_circle = k_center[0] + k_radius * np.cos(k_angles)
        ky_circle = k_center[1] + k_radius * np.sin(k_angles)
        tracked = self.state_tracker.track_floquet_states_on_path(
            kx_circle,
            ky_circle,
            band=band,
        )
        tracked_states = tracked["floquet_states"]
        return kx_circle, ky_circle, tracked_states

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

        _, _, tracked_states = self._track_floquet_states_on_circle(
            k_radius=k_radius,
            k_center=k_center,
            n_points=n_points,
            band=band,
        )

        state_k_cache = [
            self.state_provider.reconstruct_floquet_state(state, time=time)
            for state in tracked_states
        ]

        berry_phase = 1 + 0j
        for i in range(n_points):
            state1 = state_k_cache[i]
            state2 = state_k_cache[(i + 1) % n_points]
            berry_phase *= self.curvature_calculator._compute_link_variable(
                state1,
                state2,
            )
        return np.angle(berry_phase)
