"""Integrated current observables built from local Floquet velocities."""

from __future__ import annotations

import numpy as np

from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils.kquadrature import KQuadrature
from .states import FloquetStateCache, FloquetStateTracker, normalize_bands
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

    def _current_velocity_maps(
        self,
        time,
        quadrature: KQuadrature,
        kind: str,
        bands,
        include_charge: bool,
        band_selection_mode: str,
        state_selection_algorithm: str,
    ):
        """Build ``{band: (jx_map, jy_map)}`` local-current maps for ``kind``.

        Each map is evaluated on the quadrature's sample points and shares its
        grid shape, so it can be reduced directly with ``quadrature.integrate``.
        The pointwise/adiabatic paths diagonalize once and select every band
        from the shared eigensystem; the tracked path follows one band per
        traversal and simply loops (returning a one-entry-per-band dict).
        """
        bands = normalize_bands(bands)
        kx_grid = quadrature.kx_grid
        ky_grid = quadrature.ky_grid
        if kind == "floquet":
            if state_selection_algorithm == "tracked":
                result = {}
                for band in bands:
                    tracked = self.state_tracker.track_floquet_states_on_grid(
                        kx_grid,
                        ky_grid,
                        band=band,
                        init_mode=band_selection_mode,
                    )
                    result[band] = (
                        self.velocity_calculator.compute_floquet_velocity_map_from_states(
                            time,
                            kx_grid,
                            ky_grid,
                            tracked["floquet_states"],
                            include_charge=include_charge,
                        )
                    )
                return result
            if state_selection_algorithm == "pointwise":
                return self.velocity_calculator.compute_floquet_velocity_map(
                    time,
                    kx_grid,
                    ky_grid,
                    bands=bands,
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
                bands=bands,
                include_charge=include_charge,
            )
        raise ValueError("kind must be 'floquet' or 'adiabatic'.")

    def integrate_current(
        self,
        quadrature: KQuadrature,
        kind: str = "floquet",
        band="conduction",
        *,
        time=None,
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
            time: Scalar time, 1D time array, or ``None`` for the default Floquet
                time grid. ``integrated_jx``/``integrated_jy`` are returned per
                time sample.
            include_charge: Multiply the velocity by the charge to return a
                charge current.
            band_selection_mode: State-selection rule (``"floquet"`` only).
            state_selection_algorithm: ``"tracked"`` (grid-wide branch tracking)
                or ``"pointwise"`` (independent per-momentum selection).
                ``"floquet"`` only; ignored for ``"adiabatic"``.

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)``; the current arrays
            share the returned time grid's shape.
        """
        if time is None:
            time = self.floquet_params.time_grid(self.period)
        time = np.atleast_1d(np.asarray(time, dtype=float))
        maps = self._current_velocity_maps(
            time,
            quadrature,
            kind,
            (band,),
            include_charge,
            band_selection_mode,
            state_selection_algorithm,
        )
        jx_map, jy_map = maps[band]
        return time, quadrature.integrate(jx_map), quadrature.integrate(jy_map)

    def integrate_occupied_current(
        self,
        quadrature: KQuadrature,
        e_fermi,
        *,
        kind: str = "floquet",
        time=None,
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
        state_selection_algorithm: str = "pointwise",
        bands=("valence", "conduction"),
    ):
        """Integrate the total occupied-state current over a k-space quadrature.

        Each band is occupied at ``k`` where its *static* energy is below the
        Fermi level, ``E_band(k) < e_fermi``. This is the general rule for both
        signs of ``e_fermi``: for ``e_fermi > 0`` the valence band is filled
        everywhere and the conduction band only inside its Fermi pockets; for
        ``e_fermi < 0`` the conduction band is empty and the valence band is
        filled only outside its hole pockets. No band is assumed full.

        Occupation is masked *before* the expensive Floquet work: static band
        energies are computed at every sample point (a cheap batched
        diagonalization), fully empty bands are dropped, and the Floquet
        diagonalization / velocity evaluation runs only over the union of cells
        that are occupied in at least one band. Within that union each band's
        contribution is kept only where that band is occupied, then summed and
        integrated. Because both bands come from the one union diagonalization,
        the single-diagonalization sharing is preserved.

        Args:
            quadrature: Sample points and local integration measure (weights).
            e_fermi: Fermi level (in the model's energy units); may be negative.
            kind: ``"floquet"`` or ``"adiabatic"`` (see :meth:`integrate_current`).
            time: Scalar time, 1D time array, or ``None`` for the default Floquet
                time grid. ``integrated_jx``/``integrated_jy`` are returned per
                time sample.
            include_charge: Multiply the velocity by the charge to return a
                charge current.
            band_selection_mode: State-selection rule (``"floquet"`` only).
            state_selection_algorithm: ``"tracked"`` or ``"pointwise"``
                (``"floquet"`` only; ignored for ``"adiabatic"``).
            bands: Band selectors considered for occupation (default valence and
                conduction).

        Returns:
            Tuple ``(time, integrated_jx, integrated_jy)``; the current arrays
            share the returned time grid's shape.
        """
        if time is None:
            time = self.floquet_params.time_grid(self.period)
        time = np.atleast_1d(np.asarray(time, dtype=float))
        bands = normalize_bands(bands)

        kx_flat = np.asarray(quadrature.kx_grid, dtype=float).ravel()
        ky_flat = np.asarray(quadrature.ky_grid, dtype=float).ravel()
        weights_flat = np.asarray(quadrature.weights, dtype=float).ravel()

        provider = self.velocity_calculator.state_provider
        static_energies, _ = provider.diagonalize_static_hamiltonian_batched(
            kx_flat, ky_flat
        )
        reference_energy = static_energies[0]
        band_masks = {
            band: static_energies[:, provider.resolve_band_index(reference_energy, band)]
            < e_fermi
            for band in bands
        }
        occupied_bands = [band for band in bands if band_masks[band].any()]

        zero = np.zeros(time.size, dtype=float)
        if not occupied_bands:
            return time, zero.copy(), zero.copy()

        union = np.zeros(kx_flat.size, dtype=bool)
        for band in occupied_bands:
            union |= band_masks[band]

        sub_quadrature = KQuadrature(
            kx_grid=np.ascontiguousarray(kx_flat[union]),
            ky_grid=np.ascontiguousarray(ky_flat[union]),
            weights=np.ascontiguousarray(weights_flat[union]),
        )
        maps = self._current_velocity_maps(
            time,
            sub_quadrature,
            kind,
            occupied_bands,
            include_charge,
            band_selection_mode,
            state_selection_algorithm,
        )
        sum_x = np.zeros((int(union.sum()), time.size), dtype=float)
        sum_y = np.zeros_like(sum_x)
        for band in occupied_bands:
            occ = band_masks[band][union][:, None]
            vx, vy = maps[band]
            sum_x += np.where(occ, vx, 0.0)
            sum_y += np.where(occ, vy, 0.0)
        return time, sub_quadrature.integrate(sum_x), sub_quadrature.integrate(sum_y)
