"""Manager for Floquet transport-style and integrated observables."""

from ..calculators import FloquetBerryPhaseCalculator
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian


class FloquetTransportManager:
    """Facade for Berry phases and integrated k-space transport observables."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
    ):
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.berry_phase_calculator = FloquetBerryPhaseCalculator(
            driven_hamiltonian,
            floquet_params,
        )

    def compute_floquet_berry_phase(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=100,
        band="conduction",
    ):
        return self.berry_phase_calculator.compute_floquet_berry_phase(
            time,
            k_radius,
            k_center,
            n_points,
            band=band,
        )

    def integrate_berry_curvature_on_grid(
        self,
        time,
        k_radius,
        k_center=(0, 0),
        n_points=51,
        band="conduction",
    ):
        return self.berry_phase_calculator.integrate_curvature_over_kgrid(
            time,
            k_radius,
            k_center,
            n_points,
            band,
        )
