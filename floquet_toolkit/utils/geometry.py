"""Small geometric helper functions used across the Floquet toolkit."""

from __future__ import annotations

import numpy as np


def signed_loop_area(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return the signed area enclosed by one closed 2D trajectory."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    return 0.5 * float(
        np.sum(x_values * np.roll(y_values, -1) - y_values * np.roll(x_values, -1))
    )
