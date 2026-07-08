"""Regression tests for the batched adiabatic-velocity map.

Numerical-consistency checks only (no physics assumptions): computing the
velocity for a whole k-grid at once must equal computing it one k at a time, and
the charge flag must scale the result by exactly ``e_charge``. This guards the
batched band-state selection (``eigvecs[..., band_index]``) that a prior bug got
wrong by picking a row instead of a column for multi-k grids.
"""

import numpy as np
import pytest

from floquet_toolkit import FloquetLocalManager
from floquet_toolkit.builtin_models.graphene import (
    GrapheneParameters,
    driven_graphene_model,
)
from floquet_toolkit.config import DriveParameters, FloquetParameters

DRIVE = DriveParameters(AL=3.0e-9, AR=3.0e-9)
FLOQUET = FloquetParameters(n_trunc=5, n_harmonics=2, n_time=41)
# Generic k-points, away from the Dirac nodes (~1.5e10) and Gamma, so the two
# bands are gapped and their energy order is unambiguous.
KX = np.array([1.0e9, 3.0e9, -2.0e9, 4.0e9])
KY = np.array([2.0e9, -1.0e9, 3.0e9, 1.0e9])


def build_calculator():
    model = driven_graphene_model(GrapheneParameters(), DRIVE)
    return FloquetLocalManager(model, FLOQUET).floquet_velocity_calculator


@pytest.mark.parametrize("band", ["valence", "conduction"])
def test_batched_map_matches_per_point(band):
    calc = build_calculator()
    time = np.array([0.3 * DRIVE.period])
    vx_all, vy_all = calc.compute_adiabatic_velocity_map(time, KX, KY, bands=(band,))[band]
    for i in range(KX.size):
        vx_i, vy_i = calc.compute_adiabatic_velocity_map(
            time, KX[i:i + 1], KY[i:i + 1], bands=(band,)
        )[band]
        assert vx_all[i] == pytest.approx(vx_i[0])
        assert vy_all[i] == pytest.approx(vy_i[0])


@pytest.mark.parametrize("band", ["valence", "conduction"])
def test_batched_map_matches_scalar_analytic(band):
    # Batched finite-difference map vs the scalar analytic-operator routine:
    # same band-resolved state, so they agree to finite-difference accuracy.
    calc = build_calculator()
    t = 0.3 * DRIVE.period
    vx_map, vy_map = calc.compute_adiabatic_velocity_map(
        np.array([t]), KX, KY, bands=(band,)
    )[band]
    for i in range(KX.size):
        vx_s, vy_s = calc.compute_adiabatic_velocity(
            t, float(KX[i]), float(KY[i]), band=band
        )
        assert vx_map[i, 0] == pytest.approx(vx_s, rel=1e-3, abs=1.0)
        assert vy_map[i, 0] == pytest.approx(vy_s, rel=1e-3, abs=1.0)


def test_include_charge_scales_by_e_charge():
    calc = build_calculator()
    time = np.array([0.3 * DRIVE.period])
    vx, vy = calc.compute_adiabatic_velocity_map(
        time, KX, KY, bands=("conduction",), include_charge=False
    )["conduction"]
    jx, jy = calc.compute_adiabatic_velocity_map(
        time, KX, KY, bands=("conduction",), include_charge=True
    )["conduction"]
    assert np.allclose(jx, calc.e_charge * vx)
    assert np.allclose(jy, calc.e_charge * vy)


def test_multi_time_grid_matches_single_time_slices():
    # Batching over time must also match slice-by-slice evaluation.
    calc = build_calculator()
    times = np.array([0.1, 0.3, 0.55]) * DRIVE.period
    vx_all, vy_all = calc.compute_adiabatic_velocity_map(
        times, KX, KY, bands=("conduction",)
    )["conduction"]
    for j, t in enumerate(times):
        vx_j, vy_j = calc.compute_adiabatic_velocity_map(
            np.array([t]), KX, KY, bands=("conduction",)
        )["conduction"]
        assert vx_all[:, j] == pytest.approx(vx_j[:, 0])
        assert vy_all[:, j] == pytest.approx(vy_j[:, 0])


@pytest.mark.parametrize(
    "method_name",
    ["compute_floquet_velocity_map", "compute_adiabatic_velocity_map"],
)
def test_both_bands_in_one_call_match_single_band_calls(method_name):
    # The core dedup guarantee: requesting both bands from one diagonalization
    # must give the same maps as requesting each band on its own.
    calc = build_calculator()
    method = getattr(calc, method_name)
    time = np.array([0.3 * DRIVE.period])

    both = method(time, KX, KY, bands=("valence", "conduction"))
    assert set(both) == {"valence", "conduction"}
    for band in ("valence", "conduction"):
        vx_one, vy_one = method(time, KX, KY, bands=(band,))[band]
        vx_both, vy_both = both[band]
        assert vx_both == pytest.approx(vx_one)
        assert vy_both == pytest.approx(vy_one)


def test_map_returns_only_requested_bands():
    # The dict keys are a truthful record of what was computed.
    calc = build_calculator()
    time = np.array([0.3 * DRIVE.period])
    result = calc.compute_floquet_velocity_map(time, KX, KY, bands=("conduction",))
    assert set(result) == {"conduction"}
    with pytest.raises(KeyError):
        result["valence"]
