"""Multi-point Floquet branch tracking built on top of single-point state selection."""

from collections import deque

import numpy as np

from .floquet_state_provider import FloquetStateProvider


class FloquetStateTracker:
    """Track one Floquet branch smoothly across grids and paths in k-space."""

    def __init__(self, state_provider: FloquetStateProvider):
        """Store the single-point state provider used for all local queries."""
        self.state_provider = state_provider
        self.cache = state_provider.cache
        self.dimension = state_provider.dimension
        self.n_blocks = state_provider.n_blocks

    def _select_local_candidate_index(
        self,
        candidate_states_t0,
        reference_state_t0,
        target_state,
        tie_tolerance: float = 1.0e-3,
    ):
        """Select one local Floquet-zone candidate by continuity and tie-breaks."""
        continuity_overlaps = np.abs(
            candidate_states_t0.conj().T @ reference_state_t0
        ) ** 2

        if continuity_overlaps.size == 0:
            raise ValueError(
                "No candidate Floquet states to select from; the candidate set "
                "is empty."
            )
        if continuity_overlaps.size == 1:
            return 0

        ranked = np.argsort(continuity_overlaps)[::-1]
        best_index = int(ranked[0])
        runner_up_index = int(ranked[1])
        best_overlap = continuity_overlaps[best_index]
        runner_up_overlap = continuity_overlaps[runner_up_index]

        if abs(best_overlap - runner_up_overlap) > tie_tolerance:
            return best_index

        static_overlaps = np.abs(candidate_states_t0.conj().T @ target_state) ** 2
        return int(np.argmax(static_overlaps))

    def _select_floquet_state_from_reference(
        self,
        reference_state,
        target_state,
        quasi_energy,
        floquet_states,
    ):
        """Select one Floquet-zone eigenstate using continuity from a reference."""
        reference_state_t0 = self.state_provider.reconstruct_floquet_state(
            reference_state,
            time=0.0,
        )
        candidate_indices, _, candidate_states_t0 = self.state_provider._zone_candidate_data(
            quasi_energy,
            floquet_states,
        )
        local_index = self._select_local_candidate_index(
            candidate_states_t0,
            reference_state_t0,
            target_state,
        )
        band_index = int(candidate_indices[local_index])
        selected_state = self.state_provider._align_extended_state_phase(
            reference_state_t0,
            candidate_states_t0[:, local_index],
            floquet_states[:, band_index],
        )
        return band_index, quasi_energy[band_index], selected_state

    def track_floquet_states_on_grid(
        self,
        kx_grid,
        ky_grid,
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ):
        """Track one Floquet branch consistently across a 2D momentum grid."""
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)

        if kx_grid.shape != ky_grid.shape:
            raise ValueError("kx_grid and ky_grid must have the same shape.")
        if kx_grid.ndim != 2:
            raise ValueError("kx_grid and ky_grid must be 2D arrays.")

        grid_shape = kx_grid.shape
        if seed_indices is None:
            seed_indices = (0, 0)
        seed_i, seed_j = map(int, seed_indices)
        if not (0 <= seed_i < grid_shape[0] and 0 <= seed_j < grid_shape[1]):
            raise IndexError("seed_indices are out of range for the supplied grid.")

        if self.cache is not None:
            cached = self.cache.get_tracked_grid(
                kx_grid,
                ky_grid,
                band=band,
                seed_indices=(seed_i, seed_j),
                init_mode=init_mode,
            )
            if cached is not None:
                return cached

        n_states = self.dimension * self.n_blocks
        selected_indices = np.full(grid_shape, -1, dtype=int)
        selected_quasi_energies = np.zeros(grid_shape, dtype=float)
        selected_floquet_states = np.zeros(grid_shape + (n_states,), dtype=complex)
        visited = np.zeros(grid_shape, dtype=bool)

        seed_kx = kx_grid[seed_i, seed_j]
        seed_ky = ky_grid[seed_i, seed_j]
        seed_index, _, seed_state = self.state_provider.select_floquet_state(
            seed_kx,
            seed_ky,
            band=band,
            band_selection_mode=init_mode,
        )
        seed_quasi_energies, _ = self.state_provider.diagonalize_floquet_hamiltonian(
            seed_kx,
            seed_ky,
        )

        selected_indices[seed_i, seed_j] = seed_index
        selected_quasi_energies[seed_i, seed_j] = seed_quasi_energies[seed_index]
        selected_floquet_states[seed_i, seed_j, :] = seed_state
        visited[seed_i, seed_j] = True

        queue = deque([(seed_i, seed_j)])
        neighbor_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            i, j = queue.popleft()
            reference_state = selected_floquet_states[i, j, :]

            for di, dj in neighbor_steps:
                ni, nj = i + di, j + dj
                if not (0 <= ni < grid_shape[0] and 0 <= nj < grid_shape[1]):
                    continue
                if visited[ni, nj]:
                    continue

                quasi_energy, floquet_states = (
                    self.state_provider.diagonalize_floquet_hamiltonian(
                        kx_grid[ni, nj],
                        ky_grid[ni, nj],
                    )
                )
                static_energy, static_states = (
                    self.state_provider.diagonalize_static_hamiltonian(
                        kx_grid[ni, nj],
                        ky_grid[ni, nj],
                    )
                )
                target_band_index = self.state_provider.resolve_band_index(
                    static_energy,
                    band,
                )
                target_state = static_states[:, target_band_index]
                band_index, tracked_energy, tracked_state = (
                    self._select_floquet_state_from_reference(
                        reference_state,
                        target_state,
                        quasi_energy,
                        floquet_states,
                    )
                )

                selected_indices[ni, nj] = band_index
                selected_quasi_energies[ni, nj] = tracked_energy
                selected_floquet_states[ni, nj, :] = tracked_state
                visited[ni, nj] = True
                queue.append((ni, nj))

        tracked = {
            "indices": selected_indices,
            "quasi_energies": selected_quasi_energies,
            "floquet_states": selected_floquet_states,
            "seed_indices": (seed_i, seed_j),
        }
        if self.cache is not None:
            self.cache.store_tracked_grid(
                kx_grid,
                ky_grid,
                tracked,
                band=band,
                seed_indices=(seed_i, seed_j),
                init_mode=init_mode,
            )
        return tracked

    def track_floquet_states_on_path(
        self,
        kx_path,
        ky_path,
        band="conduction",
        init_mode: str = "overlap",
    ):
        """Track one Floquet branch along an ordered path in momentum space."""
        kx_path = np.asarray(kx_path, dtype=float)
        ky_path = np.asarray(ky_path, dtype=float)
        if kx_path.shape != ky_path.shape:
            raise ValueError("kx_path and ky_path must have the same shape.")
        if kx_path.ndim != 1:
            raise ValueError("kx_path and ky_path must be 1D arrays.")

        n_points = kx_path.size
        n_states = self.dimension * self.n_blocks
        tracked_states = np.zeros((n_points, n_states), dtype=complex)
        tracked_indices = np.full(n_points, -1, dtype=int)
        tracked_quasi_energies = np.zeros(n_points, dtype=float)

        seed_index, _, seed_state = self.state_provider.select_floquet_state(
            float(kx_path[0]),
            float(ky_path[0]),
            band=band,
            band_selection_mode=init_mode,
        )
        seed_quasi_energies, _ = self.state_provider.diagonalize_floquet_hamiltonian(
            float(kx_path[0]),
            float(ky_path[0]),
        )
        tracked_states[0, :] = seed_state
        tracked_indices[0] = seed_index
        tracked_quasi_energies[0] = seed_quasi_energies[seed_index]

        for index in range(1, n_points):
            kx = float(kx_path[index])
            ky = float(ky_path[index])
            quasi_energy, floquet_states = (
                self.state_provider.diagonalize_floquet_hamiltonian(kx, ky)
            )
            static_energy, static_states = self.state_provider.diagonalize_static_hamiltonian(
                kx,
                ky,
            )
            target_index = self.state_provider.resolve_band_index(static_energy, band)
            target_state = static_states[:, target_index]
            band_index, tracked_energy, tracked_state = (
                self._select_floquet_state_from_reference(
                    tracked_states[index - 1, :],
                    target_state,
                    quasi_energy,
                    floquet_states,
                )
            )
            tracked_states[index, :] = tracked_state
            tracked_indices[index] = band_index
            tracked_quasi_energies[index] = tracked_energy

        return {
            "indices": tracked_indices,
            "quasi_energies": tracked_quasi_energies,
            "floquet_states": tracked_states,
        }
