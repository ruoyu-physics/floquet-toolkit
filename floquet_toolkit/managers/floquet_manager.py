"""Compatibility facade combining local and transport managers."""

from ..config import FloquetParameters
from ..core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .floquet_local_manager import FloquetLocalManager
from .floquet_transport_manager import FloquetTransportManager


class FloquetManager:
    """Compatibility wrapper exposing ``local`` and ``transport`` managers."""

    def __init__(
        self,
        driven_hamiltonian: DrivenBlochHamiltonian,
        floquet_params: FloquetParameters,
        use_cache: bool = False,
    ):
        """Initialize the combined facade.

        Args:
            driven_hamiltonian: The driven Bloch Hamiltonian to analyze.
            floquet_params: Floquet truncation / sampling parameters.
            use_cache: Forwarded to the transport manager to enable
                Floquet-state caching. Defaults to ``False``; set to ``True``
                for workflows that revisit the same momenta.
        """
        self.driven_hamiltonian = driven_hamiltonian
        self.floquet_params = floquet_params
        self.local = FloquetLocalManager(driven_hamiltonian, floquet_params)
        self.transport = FloquetTransportManager(
            driven_hamiltonian,
            floquet_params,
            use_cache=use_cache,
        )

    def __getattr__(self, name):
        """Delegate legacy attribute access to the local or transport manager."""
        if hasattr(self.local, name):
            return getattr(self.local, name)
        if hasattr(self.transport, name):
            return getattr(self.transport, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
