import numpy as np
import pytest

from floquet_toolkit import (
    DiracModel,
    DiracParameters,
    FloquetLocalManager,
    FloquetTransportManager,
    GrapheneModel,
    GrapheneParameters,
    RotatingFrameDiracModel,
)
from floquet_toolkit.calculators import FloquetCurrentCalculator, FloquetStateCache
from floquet_toolkit.config import DriveParameters, FloquetParameters
from floquet_toolkit.utils.kquadrature import (
    KQuadrature,
    create_polar_k_grid,
    integrate_polar_grid,
)


DIRAC_PARAMS = DiracParameters()
GRAPHENE_PARAMS = GrapheneParameters()
DRIVE_PARAMS = DriveParameters(AL=3.0e-9, AR=0.0)
FLOQUET_PARAMS = FloquetParameters(n_trunc=3, n_harmonics=1, n_time=21)


def test_builtin_model_classes_convert_to_driven_hamiltonians():
    dirac = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    graphene = GrapheneModel(GRAPHENE_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    rotating = RotatingFrameDiracModel(
        DIRAC_PARAMS,
        DRIVE_PARAMS,
    ).to_driven_hamiltonian()

    for model in (dirac, graphene, rotating):
        sample = model.Ht(0.0, 0.0, 0.0)
        assert sample.shape == (2, 2)
        assert np.allclose(sample, sample.conj().T)
        assert model.units == DRIVE_PARAMS.units


def test_floquet_manager_diagonalize_floquet_hamiltonian_returns_square_eigensystem():
    model = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)

    quasi_energy, floquet_states = manager.diagonalize_floquet_hamiltonian(0.0, 0.0)

    expected_dimension = model.dimension * FLOQUET_PARAMS.n_blocks
    assert quasi_energy.shape == (expected_dimension,)
    assert floquet_states.shape == (expected_dimension, expected_dimension)
    assert np.allclose(
        floquet_states.conj().T @ floquet_states,
        np.eye(expected_dimension),
        atol=1e-12,
    )


def test_transport_manager_cache_preserves_floquet_current_values():
    model = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    # Caching is opt-in (off by default); enable it explicitly to test that it
    # preserves the current values produced by the uncached calculator.
    transport_manager = FloquetTransportManager(model, FLOQUET_PARAMS, use_cache=True)
    uncached_calculator = FloquetCurrentCalculator(model, FLOQUET_PARAMS)

    assert isinstance(transport_manager.cache, FloquetStateCache)

    quadrature = KQuadrature.cartesian(1.0e6, n_k_points=5)
    cached_time, cached_jx, cached_jy = transport_manager.integrate_current(
        quadrature,
        kind="floquet",
        include_charge=True,
        state_selection_algorithm="tracked",
        band_selection_mode="overlap",
    )
    uncached_time, uncached_jx, uncached_jy = uncached_calculator.integrate_current(
        quadrature,
        kind="floquet",
        include_charge=True,
        state_selection_algorithm="tracked",
        band_selection_mode="overlap",
    )

    assert np.allclose(cached_time, uncached_time)
    assert np.allclose(cached_jx, uncached_jx)
    assert np.allclose(cached_jy, uncached_jy)


def test_local_manager_rejects_removed_quasi_energy_selection_mode():
    model = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)

    with pytest.raises(ValueError, match="band_selection_mode must be 'overlap'"):
        manager.select_floquet_state(
            0.0,
            0.0,
            band="conduction",
            band_selection_mode="quasi_energy",
        )


def test_integrate_polar_grid_matches_disk_area_for_constant_integrand():
    radius = 2.5
    exact_area = np.pi * radius**2

    for n_points in (11, 21, 51):
        r_values, theta_values, r_grid, _, _, _ = create_polar_k_grid(
            r_range=(0.0, radius),
            theta_range=(0.0, 2.0 * np.pi),
            num_r=n_points,
            num_theta=n_points,
        )
        numerical_area = integrate_polar_grid(
            np.ones_like(r_grid),
            r_values,
            theta_values,
            mode="trapezoidal",
        )
        assert np.isclose(numerical_area, exact_area, rtol=1.0e-12, atol=1.0e-12)


def test_integrate_polar_grid_riemann_matches_closed_form_and_converges():
    radius = 2.5
    exact_area = np.pi * radius**2

    previous_error = None
    for n_points in (11, 21, 51):
        r_values, theta_values, r_grid, _, _, _ = create_polar_k_grid(
            r_range=(0.0, radius),
            theta_range=(0.0, 2.0 * np.pi),
            num_r=n_points,
            num_theta=n_points,
        )
        numerical_area = integrate_polar_grid(
            np.ones_like(r_grid),
            r_values,
            theta_values,
        )  # default mode is "riemann"
        # Closed form of the sum(value * area) rule for a constant integrand:
        # the included radial endpoints overcount the boundary by n/(n-1).
        expected = exact_area * n_points / (n_points - 1)
        assert np.isclose(numerical_area, expected, rtol=1.0e-12, atol=1.0e-12)

        # The boundary bias shrinks monotonically as the grid refines.
        error = abs(numerical_area - exact_area)
        if previous_error is not None:
            assert error < previous_error
        previous_error = error


def test_integrate_polar_grid_riemann_is_default():
    radius = 1.0
    r_values, theta_values, r_grid, _, _, _ = create_polar_k_grid(
        r_range=(0.0, radius),
        theta_range=(0.0, 2.0 * np.pi),
        num_r=17,
        num_theta=17,
    )
    integrand = 1.0 + r_grid**2
    default_result = integrate_polar_grid(integrand, r_values, theta_values)
    riemann_result = integrate_polar_grid(
        integrand, r_values, theta_values, mode="riemann"
    )
    assert np.isclose(default_result, riemann_result, rtol=1.0e-12, atol=1.0e-12)
