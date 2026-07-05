"""Graphene Brillouin-zone geometry in the toolkit's physical frame.

Single source of truth for the reciprocal-lattice basis, the Dirac points, the
parallelogram zone outline, and the conduction-band dispersion implied by
:class:`~floquet_toolkit.builtin_models.graphene.graphene.GrapheneModel`'s
neighbor convention (``delta_1 = a (0, 1)``, ``delta_2 = a (sqrt3/2, -1/2)``,
``delta_3 = a (-sqrt3/2, -1/2)``). Previously these formulas were duplicated
across several analysis scripts; they now live next to the model they describe.

All functions take the carbon-carbon bond length ``a`` (``lattice_spacing``)
and return momenta in ``1 / length`` units, so they compose directly with model
evaluations and with the adaptive grid of :mod:`.graphene_bz_grid` (whose
``pole == 0`` / ``pole == 1`` pockets surround :func:`dirac_points`' ``K`` /
``K'``).
"""

from __future__ import annotations

import numpy as np


def reciprocal_vectors(lattice_spacing: float = 1.0):
    """Return graphene's reciprocal-lattice vectors ``(b1, b2)``.

    The Bravais primitive vectors connecting same-sublattice sites are
    ``a1 = delta_1 - delta_2 = a (-sqrt3/2, 3/2)`` and
    ``a2 = delta_1 - delta_3 = a (sqrt3/2, 3/2)``; the dual basis satisfies
    ``a_i . b_j = 2 pi delta_ij``.

    Args:
        lattice_spacing: Bond length ``a`` (``1.0`` for dimensionless units).

    Returns:
        Tuple ``(b1, b2)`` of length-2 arrays in ``1 / length`` units.
    """
    a = float(lattice_spacing)
    direct = a * np.array([[-np.sqrt(3.0) / 2.0, 1.5],
                           [np.sqrt(3.0) / 2.0, 1.5]])
    reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
    return reciprocal[0], reciprocal[1]


def dirac_points(lattice_spacing: float = 1.0):
    """Return the two inequivalent Dirac points ``(K, K')``.

    ``K = (b1 + 2 b2) / 3`` and ``K' = (2 b1 + b2) / 3`` — the zeros of the
    structure factor, expressed inside the parallelogram zone spanned by
    ``b1, b2`` with one corner at Gamma. These are the pocket centers labeled
    ``pole == 0`` (``K``, at positive ``kx``) and ``pole == 1`` (``K'``) by
    :func:`~floquet_toolkit.builtin_models.graphene.graphene_bz_grid.build_graphene_bz_grid`.
    """
    b1, b2 = reciprocal_vectors(lattice_spacing)
    return (b1 + 2.0 * b2) / 3.0, (2.0 * b1 + b2) / 3.0


def bz_corners(lattice_spacing: float = 1.0, center: bool = False):
    """Return the closed outline of the parallelogram Brillouin zone.

    The primitive cell spanned by ``b1, b2``. With ``center=False`` (default)
    one corner sits at Gamma — the zone tiled by the adaptive graphene grid;
    with ``center=True`` the cell is centered on Gamma (corners at fractional
    coordinates ``(+/-1/2, +/-1/2)``). The first corner is repeated at the end
    so the result can be plotted directly.

    Returns:
        Array of shape ``(5, 2)`` of Cartesian corner coordinates.
    """
    b1, b2 = reciprocal_vectors(lattice_spacing)
    if center:
        fracs = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    else:
        fracs = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    corners = fracs[:, [0]] * b1 + fracs[:, [1]] * b2
    return np.vstack([corners, corners[0]])


def conduction_band(kx, ky, lattice_spacing: float = 1.0, hopping: float = 1.0):
    """Return the nearest-neighbor conduction band ``E(k) = t |gamma(k)|``.

    Same dispersion as ``hopping * |GrapheneModel.structure_factor|`` in the
    physical codebase frame. Broadcasts over array-valued momenta.

    Args:
        kx, ky: Momenta in ``1 / length`` units (scalars or arrays).
        lattice_spacing: Bond length ``a``.
        hopping: Hopping energy ``t``; sets the energy unit of the result.
    """
    a = float(lattice_spacing)
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    g = (3.0 + 2.0 * np.cos(np.sqrt(3.0) * a * kx)
         + 4.0 * np.cos(1.5 * a * ky) * np.cos(0.5 * np.sqrt(3.0) * a * kx))
    return hopping * np.sqrt(np.clip(g, 0.0, None))
