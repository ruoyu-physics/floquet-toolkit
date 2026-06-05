"""Manager for Floquet transport-style and integrated observables."""

from ..calculators import (
    FloquetBerryPhaseCalculator,
    FloquetCurrentCalculator,
    FloquetStateCache,
)
from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from ..utils.kquadrature import KQuadrature


class FloquetTransportManager:
    """Facade for Berry phases and integrated k-space transport observables."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        cache: FloquetStateCache | None = None,
        use_cache: bool = False,
    ):
        """Initialize the transport manager.

        Args:
            driven_hamiltonian: The driven Bloch Hamiltonian to analyze.
            floquet_params: Floquet truncation / sampling parameters.
            cache: An explicit ``FloquetStateCache`` to share across
                calculators. When provided, it is always used (and implies
                caching is on), regardless of ``use_cache``.
            use_cache: Whether to enable Floquet-state caching when no explicit
                ``cache`` is given. Defaults to ``False``: a single cold pass
                visits each momentum once, so caching only adds key-building and
                copy overhead there. Set to ``True`` for workflows that revisit
                the same momenta — multiple observables sharing a grid, repeated
                calls, or adaptive refinement — where it gives a large speedup.
        """
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        if cache is not None:
            self.cache = cache
        elif use_cache:
            self.cache = FloquetStateCache()
        else:
            self.cache = None
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

    def integrate_current(
        self,
        quadrature: KQuadrature,
        kind: str = "floquet",
        band="conduction",
        include_charge: bool = False,
        band_selection_mode: str = "overlap",
        state_selection_algorithm: str = "tracked",
    ):
        """Integrate a local current over an explicit k-space quadrature.

        Build a :class:`~floquet_toolkit.utils.kquadrature.KQuadrature`
        (``KQuadrature.polar``/``KQuadrature.cartesian``, or a custom one) and
        select which current ``kind`` (``"floquet"``/``"adiabatic"``) to
        integrate. See
        :meth:`FloquetCurrentCalculator.integrate_current` for details.
        """
        return self.current_calculator.integrate_current(
            quadrature,
            kind=kind,
            band=band,
            include_charge=include_charge,
            band_selection_mode=band_selection_mode,
            state_selection_algorithm=state_selection_algorithm,
        )

    def integrate_adaptive_current(
        self,
        k_radius,
        kind: str = "floquet",
        k_center=(0.0, 0.0),
        n_k_points=21,
        band="conduction",
        include_charge=False,
        band_selection_mode: str = "overlap",
        state_selection_algorithm: str = "pointwise",
        adaptive_tol: float | None = None,
        adaptive_max_depth: int | None = None,
    ):
        """Integrate a local current with adaptive Cartesian refinement.

        The data-dependent counterpart of :meth:`integrate_current`; see
        :meth:`FloquetCurrentCalculator.integrate_adaptive_current`.
        """
        return self.current_calculator.integrate_adaptive_current(
            k_radius,
            kind=kind,
            k_center=k_center,
            n_k_points=n_k_points,
            band=band,
            include_charge=include_charge,
            band_selection_mode=band_selection_mode,
            state_selection_algorithm=state_selection_algorithm,
            adaptive_tol=adaptive_tol,
            adaptive_max_depth=adaptive_max_depth,
        )
