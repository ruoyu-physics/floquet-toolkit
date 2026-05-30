"""Manager for Floquet transport-style and integrated observables."""

from ..calculators import (
    FloquetBerryPhaseCalculator,
    FloquetCurrentCalculator,
    FloquetStateCache,
)
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian


class FloquetTransportManager:
    """Facade for Berry phases and integrated k-space transport observables."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        cache: FloquetStateCache | None = None,
    ):
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.cache = cache if cache is not None else FloquetStateCache()
        self.berry_phase_calculator = FloquetBerryPhaseCalculator(
            driven_hamiltonian,
            floquet_params,
            cache=self.cache,
        )
        self.current_calculator = FloquetCurrentCalculator(
            driven_hamiltonian,
            floquet_params,
            cache=self.cache,
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
        grid_type: str = "cartesian",
        dk: float | None = None,
        integration_method: str = "shared_corner",
        curvature_type: str = "floquet",
        order: int = 2,
    ):
        return self.berry_phase_calculator.integrate_curvature_over_kgrid(
            time,
            k_radius,
            k_center,
            n_points,
            band,
            grid_type=grid_type,
            dk=dk,
            integration_method=integration_method,
            curvature_type=curvature_type,
            order=order,
        )

    def integrate_floquet_current_on_fermi_disk(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        band="conduction",
        include_charge=False,
        state_selection_algorithm: str = "tracked",
        band_selection_mode: str = "overlap",
        grid_type: str = "polar",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate the exact Floquet current over a circular momentum region."""
        return self.current_calculator.integrate_floquet_current_on_fermi_disk(
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            band=band,
            include_charge=include_charge,
            state_selection_algorithm=state_selection_algorithm,
            band_selection_mode=band_selection_mode,
            grid_type=grid_type,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )

    def integrate_adiabatic_current_on_fermi_disk(
        self,
        k_radius,
        k_center=(0.0, 0.0),
        n_k_points=21,
        band="conduction",
        include_charge=False,
        grid_type: str = "polar",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate the adiabatic current over a circular momentum region."""
        return self.current_calculator.integrate_adiabatic_current_on_fermi_disk(
            k_radius=k_radius,
            k_center=k_center,
            n_k_points=n_k_points,
            band=band,
            include_charge=include_charge,
            grid_type=grid_type,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )
