"""Utilities for sampling and integrating observables over k-space grids."""

from __future__ import annotations

import numpy as np


def create_cartesian_k_grid(
    kx_range: tuple[float, float],
    ky_range: tuple[float, float],
    num_kx: int,
    num_ky: int,
    indexing: str = "ij",
):
    """Create a regular Cartesian k-space grid.

    Args:
        kx_range: Inclusive ``(min_kx, max_kx)`` sampling range.
        ky_range: Inclusive ``(min_ky, max_ky)`` sampling range.
        num_kx: Number of samples along the ``kx`` axis.
        num_ky: Number of samples along the ``ky`` axis.
        indexing: Passed through to ``numpy.meshgrid``. The default ``"ij"``
            makes the grid axes follow ``(kx, ky)`` ordering.

    Returns:
        Tuple ``(kx_values, ky_values, kx_grid, ky_grid)``.
    """
    if num_kx < 1 or num_ky < 1:
        raise ValueError("num_kx and num_ky must both be at least 1.")

    kx_values = np.linspace(kx_range[0], kx_range[1], num_kx)
    ky_values = np.linspace(ky_range[0], ky_range[1], num_ky)
    kx_grid, ky_grid = np.meshgrid(kx_values, ky_values, indexing=indexing)
    return kx_values, ky_values, kx_grid, ky_grid


def fermi_momentum(dirac_params) -> float:
    """Return the Dirac-model Fermi momentum from one parameter bundle."""
    if dirac_params.e_fermi**2 < dirac_params.mass**2:
        raise ValueError("Fermi energy must satisfy |e_fermi| >= |mass| to define a real Fermi momentum.")
    return np.sqrt(dirac_params.e_fermi**2 - dirac_params.mass**2) / (
        dirac_params.units.hbar * dirac_params.vf
    )


def create_polar_k_grid(
    r_range: tuple[float, float],
    theta_range: tuple[float, float],
    num_r: int,
    num_theta: int,
    k_center: tuple[float, float] = (0.0, 0.0),
    endpoint_theta: bool = False,
):
    """Create a regular polar k-space grid and its Cartesian embedding.

    Args:
        r_range: Inclusive ``(r_min, r_max)`` radial sampling range.
        theta_range: Angular sampling range in radians.
        num_r: Number of radial samples.
        num_theta: Number of angular samples.
        k_center: Cartesian center used to shift the grid.
        endpoint_theta: Whether to include the end of ``theta_range``.

    Returns:
        Tuple ``(r_values, theta_values, r_grid, theta_grid, kx_grid, ky_grid)``.
    """
    if num_r < 1 or num_theta < 1:
        raise ValueError("num_r and num_theta must both be at least 1.")
    if r_range[0] < 0.0:
        raise ValueError("Radial coordinates must be non-negative.")

    r_values = np.linspace(r_range[0], r_range[1], num_r)
    theta_values = np.linspace(
        theta_range[0],
        theta_range[1],
        num_theta,
        endpoint=endpoint_theta,
    )
    r_grid, theta_grid = np.meshgrid(r_values, theta_values, indexing="ij")
    kx_grid = k_center[0] + r_grid * np.cos(theta_grid)
    ky_grid = k_center[1] + r_grid * np.sin(theta_grid)
    return r_values, theta_values, r_grid, theta_grid, kx_grid, ky_grid


def create_circular_mask(
    kx_grid,
    ky_grid,
    k_radius: float,
    k_center: tuple[float, float] = (0.0, 0.0),
    include_boundary: bool = True,
):
    """Create a Boolean mask for points inside a circular k-space region."""
    radius_sq = (kx_grid - k_center[0]) ** 2 + (ky_grid - k_center[1]) ** 2
    if include_boundary:
        return radius_sq <= k_radius**2
    return radius_sq < k_radius**2


def integrate_cartesian_grid(
    values,
    kx_values,
    ky_values,
    mask=None,
):
    """Integrate sampled values on a regular Cartesian grid.

    The integration is a rectangular-rule sum on a uniform grid. ``values`` may
    contain leading dimensions, with the final two axes corresponding to
    ``(kx, ky)``.
    """
    values = np.asarray(values)
    kx_values = np.asarray(kx_values, dtype=float)
    ky_values = np.asarray(ky_values, dtype=float)

    if values.shape[-2:] != (kx_values.size, ky_values.size):
        raise ValueError(
            "The final two axes of values must match (len(kx_values), len(ky_values))."
        )

    dkx = kx_values[1] - kx_values[0] if kx_values.size > 1 else 0.0
    dky = ky_values[1] - ky_values[0] if ky_values.size > 1 else 0.0
    area_element = dkx * dky

    if mask is None:
        masked_values = values
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (kx_values.size, ky_values.size):
            raise ValueError("mask must have shape (len(kx_values), len(ky_values)).")
        masked_values = np.where(mask, values, 0.0)

    return np.sum(masked_values, axis=(-2, -1)) * area_element


def integrate_polar_grid(
    values,
    r_values,
    theta_values,
):
    """Integrate sampled values on a regular polar grid.

    The angular integral uses a uniform periodic rectangular rule, while the
    radial integral uses trapezoidal endpoint weights. This avoids the
    systematic outer-boundary overcounting that occurs if endpoint-including
    radial samples are summed with a full ``dr`` weight.

    ``values`` may contain leading dimensions, with the final two axes
    corresponding to ``(r, theta)``.
    """
    values = np.asarray(values)
    r_values = np.asarray(r_values, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float)

    if values.shape[-2:] != (r_values.size, theta_values.size):
        raise ValueError(
            "The final two axes of values must match (len(r_values), len(theta_values))."
        )

    dtheta = theta_values[1] - theta_values[0] if theta_values.size > 1 else 0.0
    angular_integral = np.sum(values, axis=-1) * dtheta

    if r_values.size == 1:
        return np.squeeze(angular_integral * r_values[0], axis=-1) * 0.0

    radial_integrand = angular_integral * r_values
    if hasattr(np, "trapezoid"):
        return np.trapezoid(radial_integrand, x=r_values, axis=-1)
    return np.trapz(radial_integrand, x=r_values, axis=-1)
