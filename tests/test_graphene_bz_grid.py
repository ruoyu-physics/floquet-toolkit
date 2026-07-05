"""Regression tests for the adaptive graphene Brillouin-zone grid.

Geometric/consistency checks (no Floquet physics): the cells tile the zone to
machine precision, the reported BZ area matches the reciprocal lattice and the
BZ polygon, the Dirac points sit at the band nodes, and the build is
deterministic.
"""

import numpy as np
import pytest

from floquet_toolkit.builtin_models.graphene import (
    GrapheneParameters,
    build_graphene_bz_grid,
    bz_corners,
    conduction_band,
    dirac_points,
    reciprocal_vectors,
)
from floquet_toolkit.utils.geometry import polygon_area_centroid

GRID_KW = dict(n_K=24, n_gamma=24, pocket_levels=3, gamma_levels=3)

# Golden reference for GRID_KW (default SI GrapheneParameters). Robust aggregates
# only -- exact cell count plus quadrature integrals / energy order-statistics,
# which are insensitive to cell ordering and last-ULP noise but move if the
# tiling, level placement, or tracing changes. To regenerate after an intended
# change: rebuild with GRID_KW and reprint these values.
GOLDEN = dict(
    n_cells=288,
    bz_area=7.53583119455671e20,
    int_energy=589.5300791605695,          # sum(areas * E_centroid)
    int_smooth=5.418870275885343e19,        # sum(areas * cos(a*kx) exp(-(a*ky)^2))
    e_min=7.875023788089802e-20,
    e_max=1.3744606256447276e-18,
    e_mean=5.013386377433138e-19,
)


def test_cells_tile_the_zone_to_machine_precision():
    grid = build_graphene_bz_grid(GrapheneParameters(), **GRID_KW)
    assert abs(grid["meta"]["area_error"]) < 1.0e-12
    assert grid["areas"].sum() == pytest.approx(grid["meta"]["bz_area"], rel=1e-12)
    assert np.all(grid["areas"] > 0.0)


def test_bz_area_matches_reciprocal_and_polygon():
    a = GrapheneParameters().lattice_spacing
    b1, b2 = reciprocal_vectors(a)
    grid = build_graphene_bz_grid(GrapheneParameters(), **GRID_KW)
    # reciprocal-vector cross product
    assert grid["meta"]["bz_area"] == pytest.approx(abs(np.cross(b1, b2)), rel=1e-12)
    # shoelace area of the BZ hexagon (bz_corners closes the ring; drop the repeat)
    corners = bz_corners(a)
    polygon_area, _ = polygon_area_centroid(corners[:-1])
    assert polygon_area == pytest.approx(grid["meta"]["bz_area"], rel=1e-9)


def test_dirac_points_are_band_nodes():
    a = GrapheneParameters().lattice_spacing
    K, K_prime = dirac_points(a)
    # The conduction band vanishes at the Dirac points (gapless nodes).
    assert conduction_band(K[0], K[1], a) == pytest.approx(0.0, abs=1e-6)
    assert conduction_band(K_prime[0], K_prime[1], a) == pytest.approx(0.0, abs=1e-6)


def test_dirac_points_follow_reciprocal_formula():
    a = GrapheneParameters().lattice_spacing
    b1, b2 = reciprocal_vectors(a)
    K, K_prime = dirac_points(a)
    assert K == pytest.approx((b1 + 2.0 * b2) / 3.0)
    assert K_prime == pytest.approx((2.0 * b1 + b2) / 3.0)


def test_grid_build_is_deterministic():
    grid1 = build_graphene_bz_grid(GrapheneParameters(), **GRID_KW)
    grid2 = build_graphene_bz_grid(GrapheneParameters(), **GRID_KW)
    assert grid1["meta"]["n_cells"] == grid2["meta"]["n_cells"]
    assert np.array_equal(grid1["k_points"], grid2["k_points"])
    assert np.array_equal(grid1["areas"], grid2["areas"])


def test_lattice_spacing_scales_area_as_inverse_square():
    # BZ area is in 1/length^2, so halving a quadruples the zone area.
    base = build_graphene_bz_grid(GrapheneParameters(lattice_spacing=1.0), **GRID_KW)
    half = build_graphene_bz_grid(GrapheneParameters(lattice_spacing=0.5), **GRID_KW)
    assert half["meta"]["bz_area"] == pytest.approx(4.0 * base["meta"]["bz_area"], rel=1e-9)


def test_grid_matches_golden_reference():
    # Drift detector: catches silent changes to the tiling that the property
    # tests above would miss. n_cells is exact; float aggregates use rtol=1e-6
    # (loose enough for cross-environment ULP noise, tight enough that a real
    # algorithm change -- which moves things by percent -- fails loudly).
    a = GrapheneParameters().lattice_spacing
    grid = build_graphene_bz_grid(GrapheneParameters(), **GRID_KW)
    areas = grid["areas"]
    energy = grid["E_centroid"]
    kpoints = grid["k_points"]
    smooth = np.cos(a * kpoints[:, 0]) * np.exp(-((a * kpoints[:, 1]) ** 2))

    assert grid["meta"]["n_cells"] == GOLDEN["n_cells"]
    assert areas.sum() == pytest.approx(GOLDEN["bz_area"], rel=1e-6)
    assert float((areas * energy).sum()) == pytest.approx(GOLDEN["int_energy"], rel=1e-6)
    assert float((areas * smooth).sum()) == pytest.approx(GOLDEN["int_smooth"], rel=1e-6)
    assert float(energy.min()) == pytest.approx(GOLDEN["e_min"], rel=1e-6)
    assert float(energy.max()) == pytest.approx(GOLDEN["e_max"], rel=1e-6)
    assert float(energy.mean()) == pytest.approx(GOLDEN["e_mean"], rel=1e-6)
