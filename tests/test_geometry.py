"""Regression tests for the shoelace/polygon helpers in ``utils.geometry``.

Pure, deterministic geometry -- no model or Floquet machinery. Guards the
shoelace area/centroid rewrite (exact polygon integrals, centering for numerical
stability, degeneracy handling) and the pure-NumPy point-in-polygon test.
"""

import numpy as np
import pytest

from floquet_toolkit.utils.geometry import (
    points_in_polygon,
    polygon_area_centroid,
    polygon_signed_area,
    signed_loop_area,
)

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def test_unit_square_area_and_centroid():
    area, centroid = polygon_area_centroid(UNIT_SQUARE)
    assert area == pytest.approx(1.0, abs=1e-14)
    assert centroid == pytest.approx([0.5, 0.5], abs=1e-14)


def test_right_triangle_area_and_centroid():
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    area, centroid = polygon_area_centroid(triangle)
    assert area == pytest.approx(3.0, abs=1e-14)          # 1/2 * 2 * 3
    assert centroid == pytest.approx([2.0 / 3.0, 1.0], abs=1e-12)  # vertex mean


def test_regular_hexagon_area_and_centroid():
    radius = 2.5
    angles = np.arange(6) * (np.pi / 3.0)
    hexagon = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    area, centroid = polygon_area_centroid(hexagon)
    assert area == pytest.approx(1.5 * np.sqrt(3.0) * radius**2, rel=1e-12)
    assert centroid == pytest.approx([0.0, 0.0], abs=1e-12)


def test_signed_area_flips_sign_under_orientation_reversal():
    ccw = polygon_signed_area(UNIT_SQUARE)
    cw = polygon_signed_area(UNIT_SQUARE[::-1])
    assert ccw == pytest.approx(1.0)
    assert cw == pytest.approx(-1.0)


def test_area_is_translation_invariant():
    offset = np.array([1.0e6, -3.0e6])
    area0, centroid0 = polygon_area_centroid(UNIT_SQUARE)
    area1, centroid1 = polygon_area_centroid(UNIT_SQUARE + offset)
    assert area1 == pytest.approx(area0, rel=1e-12)
    assert (centroid1 - offset) == pytest.approx(centroid0, abs=1e-6)


def test_area_is_invariant_under_cyclic_vertex_rotation():
    rolled = np.roll(UNIT_SQUARE, 2, axis=0)
    area0, centroid0 = polygon_area_centroid(UNIT_SQUARE)
    area1, centroid1 = polygon_area_centroid(rolled)
    assert area1 == pytest.approx(area0)
    assert centroid1 == pytest.approx(centroid0)


def test_collinear_vertices_are_degenerate():
    line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    area, centroid = polygon_area_centroid(line)
    assert area == 0.0
    assert centroid == pytest.approx(line.mean(axis=0))


def test_coincident_vertices_are_degenerate():
    point = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    area, centroid = polygon_area_centroid(point)
    assert area == 0.0
    assert centroid == pytest.approx([5.0, 5.0])


def test_signed_area_magnitude_matches_unsigned_area():
    signed = polygon_signed_area(UNIT_SQUARE)
    unsigned, _ = polygon_area_centroid(UNIT_SQUARE)
    assert abs(signed) == pytest.approx(unsigned)


def test_points_in_polygon_square():
    points = np.array(
        [[0.5, 0.5], [0.1, 0.9], [2.0, 2.0], [-1.0, 0.5], [0.5, -0.5]]
    )
    mask = points_in_polygon(UNIT_SQUARE, points)
    assert list(mask) == [True, True, False, False, False]


def test_points_in_polygon_nonconvex_L_shape():
    # L-shaped (nonconvex) polygon, traversed CCW.
    poly = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [1.0, 1.0], [1.0, 4.0], [0.0, 4.0]]
    )
    points = np.array([[0.5, 0.5], [3.5, 0.5], [2.0, 2.0], [0.5, 3.5]])
    # inside lower arm, inside lower arm, in the notch (outside), inside left arm
    mask = points_in_polygon(poly, points)
    assert list(mask) == [True, True, False, True]


def test_signed_loop_area_of_circle_converges_to_pi_r_squared():
    radius = 1.3
    exact = np.pi * radius**2

    def loop_area(n_points):
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        return signed_loop_area(radius * np.cos(theta), radius * np.sin(theta))

    error_coarse = abs(loop_area(50) - exact)
    error_fine = abs(loop_area(400) - exact)
    assert error_fine < error_coarse
    assert error_fine < 1.0e-3
