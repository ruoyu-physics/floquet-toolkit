"""Helpers for drive-field components used by built-in models and scripts."""

from __future__ import annotations

import numpy as np


def _normalized_polarization_axis(polarization_axis):
    """Return a unit vector specifying the drive-frame x axis in the lab frame."""
    if isinstance(polarization_axis, str):
        if polarization_axis == "x":
            axis = np.array([1.0, 0.0], dtype=float)
        elif polarization_axis == "y":
            axis = np.array([0.0, 1.0], dtype=float)
        else:
            raise ValueError("polarization_axis must be 'x', 'y', or a 2-vector")
    else:
        axis = np.asarray(polarization_axis, dtype=float)
        if axis.shape != (2,):
            raise ValueError("polarization_axis must be 'x', 'y', or a length-2 vector")

    norm = np.linalg.norm(axis)
    if norm == 0.0:
        raise ValueError("polarization_axis must be nonzero")
    return axis / norm


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
