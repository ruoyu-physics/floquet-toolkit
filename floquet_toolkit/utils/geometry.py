"""Small geometric helper functions used across the Floquet toolkit."""

from __future__ import annotations

import numpy as np


def _edge_cross_terms(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    """Per-edge shoelace cross terms ``x_i y_{i+1} - x_{i+1} y_i`` (wrapping)."""
    return x_values * np.roll(y_values, -1) - np.roll(x_values, -1) * y_values


def signed_loop_area(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return the signed area enclosed by one closed 2D trajectory."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    return 0.5 * float(np.sum(_edge_cross_terms(x_values, y_values)))


def polygon_signed_area(vertices) -> float:
    """Signed area of a simple polygon via the shoelace (triangle) formula.

    ``A = 1/2 sum_i (x_i y_{i+1} - x_{i+1} y_i)`` with the index wrapping
    ``i+1 -> (i+1) mod n`` (the last vertex connects back to the first).
    Positive for counterclockwise vertex order, negative for clockwise.

    Args:
        vertices: ``(N, 2)`` array-like of boundary vertices in traversal order
            (open ring; the closing edge is implicit).
    """
    v = np.asarray(vertices, dtype=float)
    return signed_loop_area(v[:, 0], v[:, 1])


def polygon_area_centroid(vertices) -> tuple[float, np.ndarray]:
    """Unsigned area and centroid of a simple polygon (shoelace formulas).

    Exact for any simple (non-self-intersecting) polygon -- convex or not --
    given its boundary vertices in traversal order. The centroid sums use the
    *signed* area so the result is independent of orientation.

    The vertices are translated to their local mean before the shoelace sums are
    evaluated. This is algebraically equivalent, but avoids catastrophic
    cancellation for tiny polygons whose absolute coordinates are large. A
    polygon whose area is negligible relative to its own extent (a sliver
    collapsed onto a line, or all vertices coincident) is treated as degenerate:
    it reports ``area = 0.0`` and the vertex mean as centroid. Without this
    guard the centroid division by the near-cancelled signed area would produce
    arbitrarily large, meaningless coordinates.

    Args:
        vertices: ``(N, 2)`` array-like of boundary vertices in traversal order
            (open ring; the closing edge is implicit).

    Returns:
        Tuple ``(area, centroid)`` with ``area >= 0`` and ``centroid`` a
        length-2 array.
    """
    v = np.asarray(vertices, dtype=float)
    origin = v.mean(axis=0)
    centered = v - origin
    x = centered[:, 0]
    y = centered[:, 1]
    cross = _edge_cross_terms(x, y)
    signed = 0.5 * float(np.sum(cross))
    # Degeneracy scale: the squared bounding-box extent of the polygon itself.
    extent_sq = max(float(np.ptp(x)), float(np.ptp(y))) ** 2
    if abs(signed) <= 1.0e-12 * extent_sq or extent_sq == 0.0:
        return 0.0, v.mean(axis=0)
    centroid_x = float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * signed)
    centroid_y = float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * signed)
    return abs(signed), origin + np.array([centroid_x, centroid_y])


def points_in_polygon(vertices, points) -> np.ndarray:
    """Vectorized even-odd (crossing-number) point-in-polygon test.

    Pure-NumPy replacement for ``matplotlib.path.Path.contains_points``.

    Args:
        vertices: ``(V, 2)`` open polygon ring (implicitly closed).
        points: ``(P, 2)`` query points.

    Returns:
        Boolean ``(P,)`` mask of points strictly inside the polygon.
    """
    vertices = np.asarray(vertices, dtype=float)
    points = np.asarray(points, dtype=float)
    px = points[:, 0]
    py = points[:, 1]
    xs = vertices[:, 0]
    ys = vertices[:, 1]
    xs2 = np.roll(xs, -1)
    ys2 = np.roll(ys, -1)
    inside = np.zeros(px.shape[0], dtype=bool)
    with np.errstate(divide='ignore', invalid='ignore'):
        for x1, y1, x2, y2 in zip(xs, ys, xs2, ys2):
            straddles = (y1 > py) != (y2 > py)
            if not straddles.any():
                continue
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            inside ^= straddles & (px < x_cross)
    return inside
