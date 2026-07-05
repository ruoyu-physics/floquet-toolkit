"""Tuning presets for the adaptive graphene Brillouin-zone grid.

Configuration policy layered on top of the mesh algorithm in
:mod:`.graphene_bz_grid`: these helpers only compute energy-level placements and
keyword bundles -- they touch none of the grid internals. Expand the returned
dictionaries directly into
:func:`~floquet_toolkit.builtin_models.graphene.graphene_bz_grid.build_graphene_bz_grid`
or
:func:`~floquet_toolkit.builtin_models.graphene.graphene_bz_grid.graphene_bz_quadrature`.
"""

from __future__ import annotations

import numpy as np


def local_dirac_polar_cut_grid(
    r_cut: float,
    k_gap: float,
    adaptive_spacing: float,
    *,
    gap_region_factor: float = 1.5,
    gap_points: int = 8,
    outer_spacing_factor: float = 1.0,
    min_angular_points: int = 24,
    max_angular_points: int | None = None,
) -> dict:
    """Return a two-zone local polar grid for a Dirac-point cutout.

    The returned points are cell centers in local ``q = k - K`` coordinates.
    The inner disk resolves the gap momentum scale; the outer annulus keeps a
    spacing comparable to the surrounding adaptive BZ grid.

    Args:
        r_cut: Radius where the local polar patch hands back to the adaptive
            BZ grid.
        k_gap: Momentum width associated with the Floquet gap,
            ``M / (hbar v_F)``. This must use the same momentum units as
            ``r_cut``.
        adaptive_spacing: Target spacing of the surrounding adaptive grid.
        gap_region_factor: Inner fine region radius is
            ``gap_region_factor * k_gap``, clipped to ``r_cut``.
        gap_points: Number of radial spacings per ``k_gap`` in the inner
            region.
        outer_spacing_factor: Multiplier on ``adaptive_spacing`` for the
            outer annulus. Values below one make the polar annulus denser.
        min_angular_points: Lower bound for angular resolution.
        max_angular_points: Optional upper bound for angular resolution.

    Returns:
        Dictionary with ``qx``, ``qy``, ``radii``, ``theta`` and diagnostics.
    """
    r_cut = float(r_cut)
    k_gap = float(k_gap)
    adaptive_spacing = float(adaptive_spacing)
    if r_cut <= 0.0:
        raise ValueError("r_cut must be positive.")
    if k_gap <= 0.0:
        raise ValueError("k_gap must be positive.")
    if adaptive_spacing <= 0.0:
        raise ValueError("adaptive_spacing must be positive.")
    if gap_region_factor <= 0.0:
        raise ValueError("gap_region_factor must be positive.")
    if gap_points < 1:
        raise ValueError("gap_points must be at least 1.")
    if outer_spacing_factor <= 0.0:
        raise ValueError("outer_spacing_factor must be positive.")
    if min_angular_points < 3:
        raise ValueError("min_angular_points must be at least 3.")

    inner_edge = min(r_cut, gap_region_factor * k_gap)
    fine_spacing = k_gap / gap_points
    outer_spacing = outer_spacing_factor * adaptive_spacing

    n_inner = max(1, int(np.ceil(inner_edge / fine_spacing)))
    inner_edges = np.linspace(0.0, inner_edge, n_inner + 1)
    inner_radii = 0.5 * (inner_edges[:-1] + inner_edges[1:])

    if inner_edge < r_cut:
        n_outer = max(1, int(np.ceil((r_cut - inner_edge) / outer_spacing)))
        outer_edges = np.linspace(inner_edge, r_cut, n_outer + 1)
        outer_radii = 0.5 * (outer_edges[:-1] + outer_edges[1:])
    else:
        n_outer = 0
        outer_radii = np.empty(0)

    radii = np.concatenate([inner_radii, outer_radii])
    n_inner_theta = int(np.ceil(2.0 * np.pi * inner_edge / fine_spacing))
    n_outer_theta = int(np.ceil(2.0 * np.pi * r_cut / outer_spacing))
    n_angular = max(
        min_angular_points,
        n_inner_theta,
        n_outer_theta,
    )
    if max_angular_points is not None:
        n_angular = min(n_angular, int(max_angular_points))
    theta = np.linspace(0.0, 2.0 * np.pi, n_angular, endpoint=False)

    rr, tt = np.meshgrid(radii, theta, indexing="ij")
    return {
        "qx": (rr * np.cos(tt)).ravel(),
        "qy": (rr * np.sin(tt)).ravel(),
        "radii": radii,
        "theta": theta,
        "r_cut": r_cut,
        "k_gap": k_gap,
        "inner_edge": inner_edge,
        "fine_spacing": fine_spacing,
        "outer_spacing": outer_spacing,
        "n_inner_radial": n_inner,
        "n_outer_radial": n_outer,
        "n_angular": n_angular,
    }


def dirac_refined_pocket_levels(
    hopping: float,
    *,
    dense_fraction: float = 0.03,
    n_inner: int = 50,
    n_outer: int = 20,
    inner_power: float = 2.0,
) -> np.ndarray:
    """Return pocket energy levels concentrated near the Dirac points.

    Integer ``pocket_levels`` values in :func:`build_graphene_bz_grid` space
    rings uniformly from ``E=0`` to the van Hove seam ``E=t``. That is often too
    coarse near ``K/K'`` when the physics of interest is meV-scale compared with
    an eV-scale hopping. This helper returns explicit interior levels:

    - ``n_inner`` levels from ``0`` to ``dense_fraction * hopping``, compressed
      toward zero by ``inner_power``.
    - ``n_outer`` logarithmically spaced levels from that dense cutoff toward
      the seam, leaving the seam itself to be added by the grid builder.

    Args:
        hopping: Graphene nearest-neighbor hopping ``t`` in the desired energy
            unit. Returned levels use the same unit.
        dense_fraction: Upper energy of the Dirac-refined region as a fraction
            of ``hopping``.
        n_inner: Number of low-energy levels concentrated near ``E=0``.
        n_outer: Number of coarse levels between the dense region and ``E=t``.
        inner_power: Power-law exponent for the low-energy levels. Larger than
            one packs more levels closer to the Dirac point.

    Returns:
        Sorted unique interior pocket levels in ``(0, hopping)``.
    """
    hopping = float(hopping)
    if hopping <= 0.0:
        raise ValueError("hopping must be positive.")
    if not (0.0 < dense_fraction < 1.0):
        raise ValueError("dense_fraction must lie between 0 and 1.")
    if n_inner < 1:
        raise ValueError("n_inner must be at least 1.")
    if n_outer < 0:
        raise ValueError("n_outer must be non-negative.")
    if inner_power <= 0.0:
        raise ValueError("inner_power must be positive.")

    dense_edge = dense_fraction * hopping
    inner = dense_edge * np.linspace(0.0, 1.0, n_inner + 1)[1:] ** inner_power
    if n_outer:
        outer = np.geomspace(dense_edge, hopping, n_outer + 2)[1:-1]
        levels = np.concatenate([inner, outer])
    else:
        levels = inner
    levels = np.unique(levels[(levels > 0.0) & (levels < hopping)])
    return levels


def dirac_refined_grid_kwargs(
    hopping: float,
    *,
    spoke_mode: str = "per-patch-uniform",
    n_K: int = 96,
    n_gamma: int = 24,
    n_per_segment: int = 8,
    alpha: float = 0.5,
    gamma_levels=5,
    dense_fraction: float = 0.01,
    n_inner: int = 24,
    n_outer: int = 8,
    inner_power: float = 2.0,
) -> dict:
    """Return a graphene BZ grid preset refined near the Dirac points.

    The returned dictionary can be expanded directly into
    :func:`build_graphene_bz_grid` or :func:`graphene_bz_quadrature`.
    ``n_K`` / ``n_gamma`` control the angular density for
    ``spoke_mode='per-patch-uniform'``; ``n_per_segment`` / ``alpha`` are kept in
    the same preset for ``spoke_mode='seam-blended'``.
    """
    return {
        "spoke_mode": spoke_mode,
        "n_K": n_K,
        "n_gamma": n_gamma,
        "n_per_segment": n_per_segment,
        "alpha": alpha,
        "pocket_levels": dirac_refined_pocket_levels(
            hopping,
            dense_fraction=dense_fraction,
            n_inner=n_inner,
            n_outer=n_outer,
            inner_power=inner_power,
        ),
        "gamma_levels": gamma_levels,
    }
