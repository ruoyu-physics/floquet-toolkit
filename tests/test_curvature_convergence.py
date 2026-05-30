import numpy as np
import pytest

from floquet_toolkit import FloquetLocalManager, FloquetTransportManager
from floquet_toolkit.builtin_models import DiracParameters, driven_dirac_model
from floquet_toolkit.config import DriveParameters, FloquetParameters, MEV_TO_J

DIRAC_PARAMS = DiracParameters(mass=-40.0 * MEV_TO_J)
DRIVE_PARAMS = DriveParameters(AL=3.0e-9, AR=3.0e-9)
FLOQUET_PARAMS = FloquetParameters(n_trunc=5, n_harmonics=2, n_time=61)
DK_VALUES = [4.0e5, 2.0e5, 1.0e5, 5.0e4]


def build_manager() -> FloquetLocalManager:
    model = driven_dirac_model(DIRAC_PARAMS, DRIVE_PARAMS)
    return FloquetLocalManager(model, FLOQUET_PARAMS)


def build_transport_manager() -> FloquetTransportManager:
    model = driven_dirac_model(DIRAC_PARAMS, DRIVE_PARAMS)
    return FloquetTransportManager(model, FLOQUET_PARAMS)


def test_instantaneous_berry_curvature_converges_with_dk():
    manager = build_manager()
    time_grid = np.linspace(0.0, DRIVE_PARAMS.period, FLOQUET_PARAMS.n_time, endpoint=False)

    curvatures = [
        manager.compute_instantaneous_berry_curvature(
            time=time_grid,
            kx=0.0,
            ky=0.0,
            band="conduction",
            dk=dk,
        )
        for dk in DK_VALUES
    ]

    reference = curvatures[-1]
    errors = [float(np.max(np.abs(curvature - reference))) for curvature in curvatures[:-1]]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 1.0e-22


def test_instantaneous_berry_curvature_uses_state_tracking(monkeypatch):
    manager = build_manager()
    calculator = manager.floquet_curvature_calculator
    original_track = calculator.state_tracker.track_floquet_states_on_grid
    calls = []

    def wrapped_track(kx_grid, ky_grid, band="conduction", seed_indices=None, init_mode="overlap"):
        calls.append(
            {
                "kx_grid": np.array(kx_grid, copy=True),
                "ky_grid": np.array(ky_grid, copy=True),
                "band": band,
                "seed_indices": seed_indices,
                "init_mode": init_mode,
            }
        )
        return original_track(
            kx_grid,
            ky_grid,
            band=band,
            seed_indices=seed_indices,
            init_mode=init_mode,
        )

    monkeypatch.setattr(
        calculator.state_tracker,
        "track_floquet_states_on_grid",
        wrapped_track,
    )

    curvature = manager.compute_instantaneous_berry_curvature(
        time=np.linspace(0.0, DRIVE_PARAMS.period, 11, endpoint=False),
        kx=0.0,
        ky=0.0,
        band="conduction",
        dk=1.0e5,
    )

    assert curvature.shape == (11,)
    assert len(calls) == 1
    assert calls[0]["band"] == "conduction"
    assert calls[0]["seed_indices"] is None
    assert calls[0]["init_mode"] == "overlap"
    assert calls[0]["kx_grid"].shape == (2, 2)
    assert calls[0]["ky_grid"].shape == (2, 2)
    assert np.allclose(
        calls[0]["kx_grid"],
        np.array([[-5.0e4, -5.0e4], [5.0e4, 5.0e4]]),
    )
    assert np.allclose(
        calls[0]["ky_grid"],
        np.array([[-5.0e4, 5.0e4], [-5.0e4, 5.0e4]]),
    )


def test_numeric_static_berry_curvature_converges_with_dk():
    manager = build_manager()
    analytic_value = manager.compute_static_berry_curvature(
        kx=0.0,
        ky=0.0,
        band="conduction",
        method="analytic",
    )

    numeric_values = [
        manager.compute_static_berry_curvature(
            kx=0.0,
            ky=0.0,
            band="conduction",
            dk=dk,
            method="numeric",
        )
        for dk in DK_VALUES
    ]
    errors = [abs(value - analytic_value) for value in numeric_values]

    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[3] < errors[2]
    assert errors[-1] < 1.0e-21


def test_shared_corner_curvature_integration_matches_same_lattice_wilson_surface():
    manager = build_transport_manager()
    calculator = manager.berry_phase_calculator
    time_grid = np.linspace(0.0, DRIVE_PARAMS.period, 11, endpoint=False)

    integrated_phase = manager.integrate_berry_curvature_on_grid(
        time=time_grid,
        k_radius=5.0e7,
        n_points=21,
        band="conduction",
        grid_type="cartesian",
        integration_method="shared_corner",
    )
    wilson_surface_phase = calculator._integrate_curvature_over_shared_corner_lattice(
        time=time_grid,
        k_radius=5.0e7,
        n_plaquettes=21,
        band="conduction",
        return_wilson_surface_phase=True,
    )

    wrapped_difference = np.angle(
        np.exp(1j * (integrated_phase - wilson_surface_phase))
    )
    assert np.max(np.abs(wrapped_difference)) < 1.0e-12


def test_shared_corner_requires_cartesian_mode():
    manager = build_transport_manager()
    time_grid = np.linspace(0.0, DRIVE_PARAMS.period, 5, endpoint=False)

    with pytest.raises(ValueError, match="requires.*cartesian"):
        manager.integrate_berry_curvature_on_grid(
            time=time_grid,
            k_radius=5.0e7,
            n_points=11,
            band="conduction",
            grid_type="polar",
            integration_method="shared_corner",
        )


@pytest.mark.parametrize(
    ("curvature_type", "time_grid", "order"),
    [
        ("static", None, 2),
        ("perturbed", np.linspace(0.0, DRIVE_PARAMS.period, 7, endpoint=False), 2),
        ("hfe", None, 2),
    ],
)
def test_shared_corner_supports_all_curvature_types(curvature_type, time_grid, order):
    manager = build_transport_manager()
    result = manager.integrate_berry_curvature_on_grid(
        time=time_grid,
        k_radius=5.0e7,
        n_points=11,
        band="conduction",
        grid_type="cartesian",
        integration_method="shared_corner",
        curvature_type=curvature_type,
        order=order,
    )

    assert np.all(np.isfinite(result))


def test_pointwise_perturbed_curvature_integration_honors_requested_order(monkeypatch):
    manager = build_transport_manager()
    calculator = manager.berry_phase_calculator.curvature_calculator
    original_compute = calculator.perturbation_calculator.compute_perturbed_state
    orders = []

    def wrapped_compute(kx, ky, band="conduction", order=1, atol=None):
        orders.append(order)
        return original_compute(kx, ky, band=band, order=order, atol=atol)

    monkeypatch.setattr(
        calculator.perturbation_calculator,
        "compute_perturbed_state",
        wrapped_compute,
    )

    result = manager.integrate_berry_curvature_on_grid(
        time=np.linspace(0.0, DRIVE_PARAMS.period, 7, endpoint=False),
        k_radius=5.0e7,
        n_points=5,
        band="conduction",
        grid_type="cartesian",
        integration_method="pointwise",
        curvature_type="perturbed",
        order=1,
    )

    assert np.all(np.isfinite(result))
    assert orders
    assert set(orders) == {1}
