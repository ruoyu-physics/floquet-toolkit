"""Utilities for sampling and integrating observables over k-space grids.

The :class:`KQuadrature` class is the unified integration primitive: it bundles
a k-grid (the sample points) with the local integration measure (per-point
weights, Jacobian included), so a disk integral of any per-k quantity reduces to
a single weighted sum ``sum(weights * value)``. Different sampling/quadrature
rules (Cartesian, polar Riemann, polar trapezoidal) differ only in how the
weights are built, not in the reduction. The standalone ``create_*``/
``integrate_*`` helpers are retained for callers that build grids and reduce
maps explicitly (e.g. the Berry-phase calculator)."""

from __future__ import annotations

from dataclasses import dataclass

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
    mode: str = "riemann",
):
    """Integrate sampled values on a regular polar grid.

    The angular integral always uses a uniform periodic rectangular rule
    (``theta`` is sampled over a full period with ``endpoint=False``). The
    radial rule is chosen by ``mode``:

    - ``"riemann"`` (default): assign every sample the polar area element
      ``r * dr * dtheta`` and sum, i.e. ``sum(value * area)``. This is a plain
      rectangular Riemann sum in both coordinates. Because the radial samples
      include both endpoints, this slightly overcounts the outer boundary
      (a constant integrand integrates to ``pi*R**2 * n_r/(n_r-1)`` rather than
      ``pi*R**2``); the bias vanishes as ``n_r`` grows.
    - ``"trapezoidal"``: integrate the radius with trapezoidal endpoint weights
      after folding in the ``r`` Jacobian. This down-weights the ``r = 0`` and
      ``r = R`` endpoints and is exact for radially linear integrands (e.g. a
      constant integrand reproduces the disk area exactly).

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

    if mode == "riemann":
        dr = r_values[1] - r_values[0] if r_values.size > 1 else 0.0
        # Per-sample polar area element r * dr * dtheta, broadcast over theta.
        area_element = (r_values * dr * dtheta)[:, None]
        return np.sum(values * area_element, axis=(-2, -1))

    if mode == "trapezoidal":
        angular_integral = np.sum(values, axis=-1) * dtheta
        if r_values.size == 1:
            return np.squeeze(angular_integral * r_values[0], axis=-1) * 0.0
        radial_integrand = angular_integral * r_values
        if hasattr(np, "trapezoid"):
            return np.trapezoid(radial_integrand, x=r_values, axis=-1)
        return np.trapz(radial_integrand, x=r_values, axis=-1)

    raise ValueError("mode must be 'riemann' or 'trapezoidal'.")


def _radial_quadrature_weights(r_values, mode: str) -> np.ndarray:
    """Return per-sample radial quadrature weights for ``mode``.

    The returned weights ``w`` satisfy ``integral_r f(r) dr ~= sum(w * f(r))``:

    - ``"riemann"``: a uniform ``dr`` at every sample (rectangular rule).
    - ``"trapezoidal"``: composite trapezoid weights (half-width at the two
      endpoints), valid for non-uniform radial spacing as well.
    """
    r_values = np.asarray(r_values, dtype=float)
    n = r_values.size
    if mode == "riemann":
        dr = r_values[1] - r_values[0] if n > 1 else 0.0
        return np.full(n, dr, dtype=float)
    if mode == "trapezoidal":
        if n == 1:
            return np.zeros(1, dtype=float)
        spacings = np.diff(r_values)
        weights = np.empty(n, dtype=float)
        weights[0] = 0.5 * spacings[0]
        weights[-1] = 0.5 * spacings[-1]
        weights[1:-1] = 0.5 * (spacings[:-1] + spacings[1:])
        return weights
    raise ValueError("mode must be 'riemann' or 'trapezoidal'.")


@dataclass(frozen=True)
class KQuadrature:
    """A k-space grid bundled with its local integration measure.

    The three arrays share the same grid shape ``G``:

    - ``kx_grid``/``ky_grid``: the Cartesian sample points (consumed directly by
      the velocity/curvature map builders).
    - ``weights``: the local integration measure at each point, i.e. the area
      element with the coordinate Jacobian and any disk mask already folded in.

    Integrating a per-k quantity is then the single weighted sum
    :meth:`integrate`, independent of how the grid was generated. Construct one
    with :meth:`cartesian` or :meth:`polar`.
    """

    kx_grid: np.ndarray
    ky_grid: np.ndarray
    weights: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shared grid shape ``G``."""
        return self.kx_grid.shape

    def integrate(self, value_map):
        """Reduce a per-k value map to its disk integral via ``sum(weights * value)``.

        Args:
            value_map: Array of shape ``G + trailing`` giving the local quantity
                at every grid point (``trailing`` may be empty or e.g. a time
                axis). The grid axes must match :attr:`shape`.

        Returns:
            Array of shape ``trailing`` (a scalar when ``value_map`` has shape
            ``G``), contracting the grid axes against the weights.
        """
        value_map = np.asarray(value_map)
        grid_ndim = self.weights.ndim
        if value_map.shape[:grid_ndim] != self.weights.shape:
            raise ValueError(
                "value_map leading axes must match the quadrature grid shape "
                f"{self.weights.shape}; got {value_map.shape}."
            )
        axes = list(range(grid_ndim))
        return np.tensordot(self.weights, value_map, axes=(axes, axes))

    @classmethod
    def cartesian(
        cls,
        k_radius: float,
        k_center: tuple[float, float] = (0.0, 0.0),
        n_k_points: int = 21,
    ) -> "KQuadrature":
        """Build a masked Cartesian quadrature over the disk's enclosing square.

        Samples a uniform ``n_k_points x n_k_points`` grid on the enclosing
        square and assigns each in-disk point the constant area element
        ``dkx * dky`` (points outside the disk get weight ``0``).
        """
        kx_values, ky_values, kx_grid, ky_grid = create_cartesian_k_grid(
            kx_range=(k_center[0] - k_radius, k_center[0] + k_radius),
            ky_range=(k_center[1] - k_radius, k_center[1] + k_radius),
            num_kx=n_k_points,
            num_ky=n_k_points,
        )
        mask = create_circular_mask(
            kx_grid, ky_grid, k_radius=k_radius, k_center=k_center
        )
        dkx = kx_values[1] - kx_values[0] if kx_values.size > 1 else 0.0
        dky = ky_values[1] - ky_values[0] if ky_values.size > 1 else 0.0
        weights = mask.astype(float) * (dkx * dky)
        return cls(kx_grid=kx_grid, ky_grid=ky_grid, weights=weights)

    @classmethod
    def polar(
        cls,
        k_radius: float,
        k_center: tuple[float, float] = (0.0, 0.0),
        n_k_points: int = 21,
        mode: str = "riemann",
    ) -> "KQuadrature":
        """Build a polar quadrature over the disk.

        Samples a uniform polar grid (``n_k_points`` radial and angular points)
        and folds the polar Jacobian ``r`` and the radial quadrature rule into
        the weights: ``w[i, j] = radial_weight[i] * r[i] * dtheta``. ``mode``
        selects the radial rule (``"riemann"`` or ``"trapezoidal"``); see
        :func:`_radial_quadrature_weights`.
        """
        r_values, theta_values, _, _, kx_grid, ky_grid = create_polar_k_grid(
            r_range=(0.0, k_radius),
            theta_range=(0.0, 2.0 * np.pi),
            num_r=n_k_points,
            num_theta=n_k_points,
            k_center=k_center,
        )
        dtheta = theta_values[1] - theta_values[0] if theta_values.size > 1 else 0.0
        radial_weight = _radial_quadrature_weights(r_values, mode)
        radial_measure = (radial_weight * r_values * dtheta)[:, None]
        weights = np.broadcast_to(radial_measure, kx_grid.shape).copy()
        return cls(kx_grid=kx_grid, ky_grid=ky_grid, weights=weights)
