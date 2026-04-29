"""Example scan of Dirac-model Berry curvature versus circular-drive amplitude."""

import matplotlib.pyplot as plt
import numpy as np

from floquet_toolkit import FloquetManager
from floquet_toolkit.builtin_models import driven_dirac_model
from floquet_toolkit.config import DriveParameters, FloquetParameters, HBAR, MEV_TO_J, PhysicsParameters


DEFAULT_PHYSICS_PARAMS = PhysicsParameters(
    vf=1.0e6,
    mass=-40.0 * MEV_TO_J,
)
DEFAULT_FLOQUET_PARAMS = FloquetParameters(
    n_harmonics=3,
    n_trunc=15,
    n_time=31,
)


def compute_time_averaged_curvature_vs_drive_amplitude(
    amplitudes: np.ndarray,
    dk: float = 1.0e4,
):
    """Return static, Floquet, and perturbed curvature averages at k=(0,0)."""
    static_curvature_results = []
    floquet_curvature_results = []
    perturbed_curvature_results = []

    for amplitude in amplitudes:
        drive_params = DriveParameters(
            omega=17.0 * MEV_TO_J / HBAR,
            AL=0.0,
            AR=float(amplitude),
        )
        model = driven_dirac_model(DEFAULT_PHYSICS_PARAMS, drive_params)
        manager = FloquetManager(model, DEFAULT_FLOQUET_PARAMS)
        time_grid = np.linspace(0.0, drive_params.period, DEFAULT_FLOQUET_PARAMS.n_time, endpoint=False)

        floquet_curvature = manager.compute_instantaneous_berry_curvature(
            time=time_grid,
            kx=0.0,
            ky=0.0,
            band="conduction",
            dk=dk,
        )
        perturbed_curvature = manager.compute_perturbed_state_berry_curvature(
            time=time_grid,
            kx=0.0,
            ky=0.0,
            band="conduction",
            dk=dk,
        )
        static_curvature = manager.compute_static_berry_curvature(
            kx=0.0,
            ky=0.0,
            band="conduction",
            dk=dk,
        )

        floquet_curvature_results.append(float(np.mean(floquet_curvature)))
        perturbed_curvature_results.append(float(np.mean(perturbed_curvature)))
        static_curvature_results.append(float(static_curvature))

    return (
        np.asarray(static_curvature_results),
        np.asarray(floquet_curvature_results),
        np.asarray(perturbed_curvature_results),
    )


def plot_time_averaged_curvature_vs_drive_amplitude(
    amplitudes: np.ndarray | None = None,
    dk: float = 1.0e4,
):
    """Plot time-averaged Berry curvature versus circular-drive amplitude."""
    if amplitudes is None:
        amplitudes = np.linspace(0.03e-9, 3.0e-9, 21)

    static_curvature, floquet_curvature, perturbed_curvature = (
        compute_time_averaged_curvature_vs_drive_amplitude(amplitudes, dk=dk)
    )

    plt.figure(figsize=(6, 4.5))
    plt.plot(amplitudes, static_curvature, "--", label="Static Berry Curvature at k=(0,0)")
    plt.plot(amplitudes, floquet_curvature, ".-", label="Time-averaged Floquet Berry Curvature")
    plt.plot(amplitudes, perturbed_curvature, label="Time-averaged Perturbed-State Berry Curvature")
    plt.xlabel("Drive amplitude A (m)")
    plt.ylabel("Berry curvature")
    plt.title("Dirac-model curvature at k=(0,0) vs circular-drive amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_time_averaged_curvature_vs_drive_amplitude()
