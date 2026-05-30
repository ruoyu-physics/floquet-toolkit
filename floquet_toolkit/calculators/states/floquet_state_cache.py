"""Cache helpers for reusable Floquet state calculations on k-space grids.

This module is intentionally standalone. It does not modify any existing
calculator behavior yet; callers can opt in explicitly by constructing a cache
object and using its storage or precompute helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _copy_if_array(value):
    """Return a defensive copy for numpy arrays, otherwise return the value."""
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


@dataclass
class FloquetStateCache:
    """Store reusable Floquet state data keyed by momentum, grid, and time.

    The cache is designed for global / grid-based workflows where multiple
    observables reuse the same expensive intermediate data:
    - static eigensystems
    - Floquet eigensystems
    - selected Floquet states
    - tracked Floquet states on 2D grids
    - reconstructed time-dependent states

    Notes:
        - The cache uses rounded floating-point keys to reduce misses caused by
          tiny binary roundoff differences.
        - Stored arrays are copied on insert and retrieval to avoid accidental
          mutation across callers.
        - This class does not depend on a specific provider implementation, but
          its helper methods expect a ``FloquetStateProvider``-like object.
    """

    decimals: int = 12
    static_eigensystems: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = (
        field(default_factory=dict)
    )
    floquet_eigensystems: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = (
        field(default_factory=dict)
    )
    selected_states: dict[
        tuple[float, float, str, str],
        tuple[int, np.ndarray, np.ndarray],
    ] = field(default_factory=dict)
    tracked_grids: dict[
        tuple[tuple[float, ...], tuple[float, ...], str, Any, str],
        dict[str, Any],
    ] = field(default_factory=dict)
    reconstructed_states: dict[
        tuple[str, tuple[float, ...]],
        np.ndarray,
    ] = field(default_factory=dict)

    def _round_float(self, value: float) -> float:
        """Round a float consistently for use in cache keys."""
        return round(float(value), self.decimals)

    def _momentum_key(self, kx: float, ky: float) -> tuple[float, float]:
        """Return one normalized key for a single momentum point."""
        return (self._round_float(kx), self._round_float(ky))

    def _band_key(self, band) -> str:
        """Normalize band labels / indices for use in cache keys."""
        return str(band)

    def _seed_key(self, seed_indices) -> Any:
        """Normalize optional grid-seed indices for use in cache keys."""
        if seed_indices is None:
            return None
        return tuple(map(int, seed_indices))

    def _time_key(self, time) -> tuple[float, ...]:
        """Return one normalized key for a scalar time or 1D time grid."""
        time = np.asarray(time, dtype=float)
        if time.ndim == 0:
            return (self._round_float(time.item()),)
        return tuple(self._round_float(t) for t in time.tolist())

    def _grid_axis_key(self, values) -> tuple[float, ...]:
        """Return a normalized 1D key for one k-axis."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("Grid axes used as cache keys must be 1D arrays.")
        return tuple(self._round_float(v) for v in values.tolist())

    def _grid_key(
        self,
        kx_grid,
        ky_grid,
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ) -> tuple[tuple[float, ...], tuple[float, ...], str, Any, str]:
        """Return one normalized key for a 2D grid-tracking request."""
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        if kx_grid.shape != ky_grid.shape:
            raise ValueError("kx_grid and ky_grid must have the same shape.")
        if kx_grid.ndim != 2:
            raise ValueError("kx_grid and ky_grid must be 2D arrays.")

        return (
            self._grid_axis_key(kx_grid[:, 0]),
            self._grid_axis_key(ky_grid[0, :]),
            self._band_key(band),
            self._seed_key(seed_indices),
            init_mode,
        )

    def _state_signature(self, floquet_state) -> str:
        """Return a stable byte signature for one extended-space Floquet state."""
        floquet_state = np.asarray(floquet_state, dtype=complex)
        return floquet_state.tobytes().hex()

    def clear(self):
        """Remove every cached entry."""
        self.static_eigensystems.clear()
        self.floquet_eigensystems.clear()
        self.selected_states.clear()
        self.tracked_grids.clear()
        self.reconstructed_states.clear()

    def clear_reconstructed_states(self):
        """Remove only cached reconstructed time-dependent states."""
        self.reconstructed_states.clear()

    def get_static_eigensystem(self, kx: float, ky: float):
        """Return a cached static eigensystem, or ``None`` if absent."""
        cached = self.static_eigensystems.get(self._momentum_key(kx, ky))
        if cached is None:
            return None
        eigvals, eigvecs = cached
        return np.array(eigvals, copy=True), np.array(eigvecs, copy=True)

    def store_static_eigensystem(self, kx: float, ky: float, eigvals, eigvecs):
        """Store one static eigensystem."""
        self.static_eigensystems[self._momentum_key(kx, ky)] = (
            np.array(eigvals, copy=True),
            np.array(eigvecs, copy=True),
        )

    def get_floquet_eigensystem(self, kx: float, ky: float):
        """Return a cached Floquet eigensystem, or ``None`` if absent."""
        cached = self.floquet_eigensystems.get(self._momentum_key(kx, ky))
        if cached is None:
            return None
        quasi_energy, floquet_states = cached
        return np.array(quasi_energy, copy=True), np.array(floquet_states, copy=True)

    def store_floquet_eigensystem(
        self,
        kx: float,
        ky: float,
        quasi_energy,
        floquet_states,
    ):
        """Store one Floquet eigensystem."""
        self.floquet_eigensystems[self._momentum_key(kx, ky)] = (
            np.array(quasi_energy, copy=True),
            np.array(floquet_states, copy=True),
        )

    def get_selected_state(
        self,
        kx: float,
        ky: float,
        band="conduction",
        band_selection_mode: str = "overlap",
    ):
        """Return a cached selected Floquet state, or ``None`` if absent."""
        key = (
            *self._momentum_key(kx, ky),
            self._band_key(band),
            band_selection_mode,
        )
        cached = self.selected_states.get(key)
        if cached is None:
            return None
        state_index, target_state, floquet_state = cached
        return (
            int(state_index),
            np.array(target_state, copy=True),
            np.array(floquet_state, copy=True),
        )

    def store_selected_state(
        self,
        kx: float,
        ky: float,
        state_index: int,
        target_state,
        floquet_state,
        band="conduction",
        band_selection_mode: str = "overlap",
    ):
        """Store one selected Floquet state and its reference static state."""
        key = (
            *self._momentum_key(kx, ky),
            self._band_key(band),
            band_selection_mode,
        )
        self.selected_states[key] = (
            int(state_index),
            np.array(target_state, copy=True),
            np.array(floquet_state, copy=True),
        )

    def get_tracked_grid(
        self,
        kx_grid,
        ky_grid,
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ):
        """Return cached tracked-grid data, or ``None`` if absent."""
        key = self._grid_key(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=seed_indices,
            init_mode=init_mode,
        )
        cached = self.tracked_grids.get(key)
        if cached is None:
            return None
        return {
            name: _copy_if_array(value)
            for name, value in cached.items()
        }

    def store_tracked_grid(
        self,
        kx_grid,
        ky_grid,
        tracked_data: dict[str, Any],
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ):
        """Store tracked Floquet-state data for one 2D grid."""
        key = self._grid_key(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=seed_indices,
            init_mode=init_mode,
        )
        self.tracked_grids[key] = {
            name: _copy_if_array(value)
            for name, value in tracked_data.items()
        }

    def get_reconstructed_state(self, floquet_state, time):
        """Return a cached reconstructed time-dependent state, or ``None``."""
        key = (self._state_signature(floquet_state), self._time_key(time))
        cached = self.reconstructed_states.get(key)
        if cached is None:
            return None
        return np.array(cached, copy=True)

    def store_reconstructed_state(self, floquet_state, time, reconstructed_state):
        """Store one reconstructed time-dependent Floquet state."""
        key = (self._state_signature(floquet_state), self._time_key(time))
        self.reconstructed_states[key] = np.array(reconstructed_state, copy=True)

    def precompute_floquet_eigensystems_on_grid(self, state_provider, kx_grid, ky_grid):
        """Diagonalize and cache the Floquet Hamiltonian on every point of a grid."""
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        if kx_grid.shape != ky_grid.shape:
            raise ValueError("kx_grid and ky_grid must have the same shape.")
        if kx_grid.ndim != 2:
            raise ValueError("kx_grid and ky_grid must be 2D arrays.")

        for i in range(kx_grid.shape[0]):
            for j in range(kx_grid.shape[1]):
                kx = float(kx_grid[i, j])
                ky = float(ky_grid[i, j])
                if self.get_floquet_eigensystem(kx, ky) is not None:
                    continue
                quasi_energy, floquet_states = state_provider.diagonalize_floquet_hamiltonian(
                    kx,
                    ky,
                )
                self.store_floquet_eigensystem(
                    kx,
                    ky,
                    quasi_energy,
                    floquet_states,
                )

    def precompute_selected_states_on_grid(
        self,
        state_provider,
        kx_grid,
        ky_grid,
        band="conduction",
        band_selection_mode: str = "overlap",
    ):
        """Select and cache one Floquet state independently at every grid point."""
        kx_grid = np.asarray(kx_grid, dtype=float)
        ky_grid = np.asarray(ky_grid, dtype=float)
        if kx_grid.shape != ky_grid.shape:
            raise ValueError("kx_grid and ky_grid must have the same shape.")
        if kx_grid.ndim != 2:
            raise ValueError("kx_grid and ky_grid must be 2D arrays.")

        for i in range(kx_grid.shape[0]):
            for j in range(kx_grid.shape[1]):
                kx = float(kx_grid[i, j])
                ky = float(ky_grid[i, j])
                if (
                    self.get_selected_state(
                        kx,
                        ky,
                        band=band,
                        band_selection_mode=band_selection_mode,
                    )
                    is not None
                ):
                    continue
                selected = state_provider.select_floquet_state(
                    kx,
                    ky,
                    band=band,
                    band_selection_mode=band_selection_mode,
                )
                self.store_selected_state(
                    kx,
                    ky,
                    selected[0],
                    selected[1],
                    selected[2],
                    band=band,
                    band_selection_mode=band_selection_mode,
                )

    def precompute_tracked_grid(
        self,
        state_tracker,
        kx_grid,
        ky_grid,
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ):
        """Track and cache one Floquet branch across a 2D grid."""
        normalized_seed = seed_indices
        if normalized_seed is None:
            normalized_seed = (0, 0)
        cached = self.get_tracked_grid(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=normalized_seed,
            init_mode=init_mode,
        )
        if cached is not None:
            return cached

        tracked_data = state_tracker.track_floquet_states_on_grid(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=seed_indices,
            init_mode=init_mode,
        )
        stored_seed = tracked_data.get("seed_indices", normalized_seed)
        self.store_tracked_grid(
            kx_grid,
            ky_grid,
            tracked_data,
            band=band,
            seed_indices=stored_seed,
            init_mode=init_mode,
        )
        return self.get_tracked_grid(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=stored_seed,
            init_mode=init_mode,
        )

    def precompute_reconstructed_states_on_grid(
        self,
        state_provider,
        state_tracker,
        kx_grid,
        ky_grid,
        time,
        band="conduction",
        seed_indices=None,
        init_mode: str = "overlap",
    ) -> np.ndarray:
        """Track one branch on a grid and cache its reconstructed states."""
        tracked = self.precompute_tracked_grid(
            state_tracker,
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=seed_indices,
            init_mode=init_mode,
        )
        tracked_states = np.asarray(tracked["floquet_states"], dtype=complex)
        reconstructed_grid = np.empty(
            tracked_states.shape[:2] + state_provider.reconstruct_floquet_state(
                tracked_states[0, 0, :],
                time=time,
            ).shape,
            dtype=complex,
        )

        for i in range(tracked_states.shape[0]):
            for j in range(tracked_states.shape[1]):
                floquet_state = tracked_states[i, j, :]
                cached = self.get_reconstructed_state(floquet_state, time)
                if cached is None:
                    cached = state_provider.reconstruct_floquet_state(
                        floquet_state,
                        time=time,
                    )
                    self.store_reconstructed_state(floquet_state, time, cached)
                reconstructed_grid[i, j, ...] = cached

        return reconstructed_grid
