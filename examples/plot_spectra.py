"""Example spectrum plots for the built-in driven Dirac and graphene models."""

import matplotlib.pyplot as plt
import numpy as np

from floquet_toolkit.calculators.floquet_spectrum_calculator import FloquetSpectrumCalculator
from floquet_toolkit.config import (
    DriveParameters,
    FloquetParameters,
    HBAR,
    MEV_TO_J,
    PhysicsParameters,
)
from floquet_toolkit.builtin_models import (
    GRAPHENE_BOND_LENGTH,
    driven_dirac_model,
    driven_graphene_model,
)


DEFAULT_PHYSICS_PARAMS = PhysicsParameters(
    vf=1.0e6,
    mass=10.0 * MEV_TO_J,
)
DEFAULT_DRIVE_PARAMS = DriveParameters(
    omega=50.0 * MEV_TO_J / HBAR,
    AL=3.0e-9,
    AR=0.0,
)
DEFAULT_FLOQUET_PARAMS = FloquetParameters(
    n_harmonics=3,
    n_trunc=7,
    n_time=21,
)


def plot_spectrum(
    model,
    floquet_params: FloquetParameters,
    kx_range: tuple[float, float],
    ky_range: tuple[float, float],
    num_kx: int = 151,
    num_ky: int = 25,
    title: str = "Floquet quasienergy spectrum",
):
    """Plot a Floquet quasienergy spectrum over a rectangular momentum grid."""
    spectrum_calculator = FloquetSpectrumCalculator(model, floquet_params)

    kx_values = np.linspace(*kx_range, num=num_kx)
    ky_values = np.linspace(*ky_range, num=num_ky)
    quasi_energy, spectral_weight, band_overlaps = spectrum_calculator.compute_floquet_spectrum(
        kx_values,
        ky_values,
        fold_to_zone=False,
    )
    quasi_energy_mev = quasi_energy / MEV_TO_J

    fig, ax = plt.subplots(figsize=(6, 5))
    for ky_index, _ in enumerate(ky_values):
        energy_slice = quasi_energy_mev[:, ky_index, :]
        weight_slice = spectral_weight[:, ky_index, :]
        overlap_slice = band_overlaps[:, ky_index, :]
        kx_points = np.repeat(kx_values[:, None], energy_slice.shape[1], axis=1)

        mask = weight_slice > 1.0e-4
        ax.scatter(
            kx_points[mask],
            energy_slice[mask],
            s=weight_slice[mask],
            c=overlap_slice[mask],
            cmap="coolwarm",
            alpha=1,
        )

    ax.set_xlabel("kx (1/m)")
    ax.set_ylabel("Quasienergy (meV)")
    ax.set_title(title)
    fig.colorbar(ax.collections[-1], ax=ax, label="Static-band overlap")
    fig.tight_layout()
    plt.show()


def plot_driven_dirac_spectrum():
    """Plot the driven Dirac Floquet spectrum near the Fermi circle."""
    model = driven_dirac_model(DEFAULT_PHYSICS_PARAMS, DEFAULT_DRIVE_PARAMS)
    k_radius = np.sqrt(
        DEFAULT_PHYSICS_PARAMS.e_fermi**2 - DEFAULT_PHYSICS_PARAMS.mass**2
    ) / (HBAR * DEFAULT_PHYSICS_PARAMS.vf)

    plot_spectrum(
        model,
        DEFAULT_FLOQUET_PARAMS,
        kx_range=(-k_radius, k_radius),
        ky_range=(-k_radius, k_radius),
        num_kx=151,
        num_ky=25,
        title="Driven Dirac Floquet spectrum",
    )


def plot_driven_graphene_spectrum():
    """Plot the driven graphene Floquet spectrum near one K point."""
    model = driven_graphene_model(DEFAULT_PHYSICS_PARAMS, DEFAULT_DRIVE_PARAMS)
    graphene_brillouin_zone = np.pi / GRAPHENE_BOND_LENGTH / np.sqrt(3.0)
    kx_k_point = 4.0 * np.pi / (3.0 * np.sqrt(3.0) * GRAPHENE_BOND_LENGTH)
    delta_k = 0.01 * graphene_brillouin_zone

    plot_spectrum(
        model,
        DEFAULT_FLOQUET_PARAMS,
        kx_range=(kx_k_point - delta_k, kx_k_point + delta_k),
        ky_range=(-delta_k, delta_k),
        num_kx=151,
        num_ky=25,
        title="Driven graphene Floquet spectrum",
    )


if __name__ == "__main__":
    plot_driven_dirac_spectrum()
