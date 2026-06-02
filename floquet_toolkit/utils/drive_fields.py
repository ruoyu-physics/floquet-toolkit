"""Helpers for drive-field components used by built-in models and scripts."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import DriveParameters


@lru_cache(maxsize=None)
def _normalized_polarization_axis_cached(key) -> tuple[float, float]:
    """Return the normalized drive-frame x axis for a hashable axis ``key``.

    Memoized on a hashable key (a string or a coordinate tuple) so the
    normalization runs once per distinct polarization axis. Because the
    polarization axis is fixed for the lifetime of a drive, this collapses the
    hundreds of thousands of identical calls made during a current calculation
    into a single computation. Returns a plain tuple so the cached value stays
    immutable (NumPy arrays would be mutable and unhashable).
    """
    if isinstance(key, str):
        if key == "x":
            axis = np.array([1.0, 0.0], dtype=float)
        elif key == "y":
            axis = np.array([0.0, 1.0], dtype=float)
        else:
            raise ValueError("polarization_axis must be 'x', 'y', or a 2-vector")
    else:
        axis = np.asarray(key, dtype=float)
        if axis.shape != (2,):
            raise ValueError("polarization_axis must be 'x', 'y', or a length-2 vector")

    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("polarization_axis must be nonzero")
    normalized = axis / norm
    return float(normalized[0]), float(normalized[1])


def _normalized_polarization_axis(polarization_axis):
    """Return a unit vector specifying the drive-frame x axis in the lab frame.

    Thin wrapper that converts the (possibly unhashable, e.g. ndarray) input
    into a hashable key, then delegates to the memoized implementation.
    """
    if isinstance(polarization_axis, str):
        key = polarization_axis
    else:
        axis = np.asarray(polarization_axis, dtype=float)
        if axis.shape != (2,):
            raise ValueError("polarization_axis must be 'x', 'y', or a length-2 vector")
        key = (float(axis[0]), float(axis[1]))
    return np.array(_normalized_polarization_axis_cached(key), dtype=float)


def _rotate_drive_frame_components(ax_drive, ay_drive, polarization_axis):
    """Rotate drive-frame components into lab-frame x/y components."""
    axis_x = _normalized_polarization_axis(polarization_axis)
    axis_y = np.array([-axis_x[1], axis_x[0]], dtype=float)
    ax = axis_x[0] * ax_drive + axis_y[0] * ay_drive
    ay = axis_x[1] * ax_drive + axis_y[1] * ay_drive
    return ax, ay


def vector_potential_components(
    time,
    omega: float,
    AL: float,
    AR: float,
    polarization_axis=(1.0, 0.0),
):
    """Return the lab-frame vector-potential components ``(Ax, Ay)``."""
    drive_x = (AL + AR) / np.sqrt(2.0) * np.cos(omega * time)
    drive_y = (AL - AR) / np.sqrt(2.0) * np.sin(omega * time)
    return _rotate_drive_frame_components(drive_x, drive_y, polarization_axis)


def electric_field_components(
    time,
    omega: float,
    AL: float,
    AR: float,
    polarization_axis=(1.0, 0.0),
):
    """Return the lab-frame electric-field components ``(Ex, Ey) = -dA/dt``."""
    drive_x = (AL + AR) / np.sqrt(2.0) * omega * np.sin(omega * time)
    drive_y = -(AL - AR) / np.sqrt(2.0) * omega * np.cos(omega * time)
    return _rotate_drive_frame_components(drive_x, drive_y, polarization_axis)


def build_circular_drive(
    amplitude: float,
    *,
    units,
    omega: float,
    handedness: str = "right",
    polarization_axis=(1.0, 0.0),
) -> DriveParameters:
    """Return circular-drive parameters from one amplitude and handedness."""
    if handedness == "left":
        al = float(amplitude)
        ar = 0.0
    elif handedness == "right":
        al = 0.0
        ar = float(amplitude)
    else:
        raise ValueError("handedness must be 'left' or 'right'.")

    return DriveParameters(
        units=units,
        omega=omega,
        AL=al,
        AR=ar,
        polarization_axis=polarization_axis,
    )
