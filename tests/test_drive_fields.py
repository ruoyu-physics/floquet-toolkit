"""Regression tests for the circular-drive field helpers in ``utils.drive_fields``.

Convention/algebra checks only (no Floquet physics): the electric field is
``-dA/dt``, circular drives have constant magnitude and zero cycle average, and
the ``left``/``right`` handedness maps to the expected ``(AL, AR)`` and the
expected rotation sense (``A_y`` flips, ``A_x`` is unchanged).
"""

import numpy as np
import pytest

from floquet_toolkit.config import UnitConvention
from floquet_toolkit.utils.drive_fields import (
    build_circular_drive,
    electric_field_components,
    vector_potential_components,
)

OMEGA = 3.0  # dimensionless; the identities hold for any frequency


def test_electric_field_is_minus_time_derivative_of_vector_potential():
    t, h = 0.37, 1.0e-6
    for al, ar in [(1.0, 0.0), (0.0, 1.0), (0.7, 0.3)]:
        ex, ey = electric_field_components(t, OMEGA, al, ar)
        ax_p, ay_p = vector_potential_components(t + h, OMEGA, al, ar)
        ax_m, ay_m = vector_potential_components(t - h, OMEGA, al, ar)
        assert ex == pytest.approx(-(ax_p - ax_m) / (2.0 * h), abs=1e-7)
        assert ey == pytest.approx(-(ay_p - ay_m) / (2.0 * h), abs=1e-7)


def test_circular_drive_has_constant_magnitude():
    t = np.linspace(0.0, 2.0 * np.pi / OMEGA, 200, endpoint=False)
    for al, ar in [(1.0, 0.0), (0.0, 1.0)]:  # pure left / pure right = circular
        ax, ay = vector_potential_components(t, OMEGA, al, ar)
        magnitude = np.hypot(ax, ay)
        assert np.allclose(magnitude, magnitude[0], rtol=1e-12)


def test_cycle_average_of_field_and_potential_vanishes():
    t = np.linspace(0.0, 2.0 * np.pi / OMEGA, 1000, endpoint=False)
    ax, ay = vector_potential_components(t, OMEGA, 0.7, 0.3)
    ex, ey = electric_field_components(t, OMEGA, 0.7, 0.3)
    for component in (ax, ay, ex, ey):
        assert np.mean(component) == pytest.approx(0.0, abs=1e-12)


def test_handedness_flips_Ay_and_keeps_Ax():
    t = np.linspace(0.0, 2.0 * np.pi / OMEGA, 50, endpoint=False)
    ax_left, ay_left = vector_potential_components(t, OMEGA, 1.0, 0.0)   # left = AL
    ax_right, ay_right = vector_potential_components(t, OMEGA, 0.0, 1.0)  # right = AR
    assert np.allclose(ax_left, ax_right)
    assert np.allclose(ay_left, -ay_right)


def test_left_and_right_rotate_oppositely():
    # z-component of A x dA/dt: positive for CCW (left), negative for CW (right).
    t = 0.3
    for al, ar, expected_sign in [(1.0, 0.0, +1.0), (0.0, 1.0, -1.0)]:
        ax, ay = vector_potential_components(t, OMEGA, al, ar)
        # dA/dt = -E
        ex, ey = electric_field_components(t, OMEGA, al, ar)
        cross_z = ax * (-ey) - ay * (-ex)
        assert np.sign(cross_z) == expected_sign


def test_build_circular_drive_maps_handedness():
    units = UnitConvention.SI_UNITS()
    left = build_circular_drive(3.0e-9, units=units, omega=1.0e13, handedness="left")
    right = build_circular_drive(3.0e-9, units=units, omega=1.0e13, handedness="right")
    assert (left.AL, left.AR) == (3.0e-9, 0.0)
    assert (right.AL, right.AR) == (0.0, 3.0e-9)


def test_build_circular_drive_rejects_bad_handedness():
    units = UnitConvention.SI_UNITS()
    with pytest.raises(ValueError):
        build_circular_drive(3.0e-9, units=units, omega=1.0e13, handedness="sideways")
