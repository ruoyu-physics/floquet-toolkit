import numpy as np

from floquet_toolkit import FloquetManager
from floquet_toolkit.builtin_models import DiracParameters, driven_dirac_model
from floquet_toolkit.config import DriveParameters, FloquetParameters, MEV_TO_J

DIRAC_PARAMS = DiracParameters(mass=-40.0 * MEV_TO_J)
DRIVE_PARAMS = DriveParameters(AL=3.0e-9, AR=3.0e-9)
FLOQUET_PARAMS = FloquetParameters(n_trunc=5, n_harmonics=2, n_time=61)
DK_VALUES = [4.0e5, 2.0e5, 1.0e5, 5.0e4]


def build_manager() -> FloquetManager:
    model = driven_dirac_model(DIRAC_PARAMS, DRIVE_PARAMS)
    return FloquetManager(model, FLOQUET_PARAMS)


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
