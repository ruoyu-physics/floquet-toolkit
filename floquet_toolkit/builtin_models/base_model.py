"""Shared helpers for built-in driven-model specifications."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import DriveParameters


def resolve_units(
    model_params,
    drive_params: DriveParameters,
):
    """Return one shared convention for the supplied model parameter sets."""
    if model_params.units != drive_params.units:
        raise ValueError(
            "model_params.units and drive_params.units must match for a built-in model"
        )
    return model_params.units


@dataclass
class BuiltinDrivenModelSpec:
    """Base class for built-in model specifications.

    Subclasses expose model-specific Hamiltonian methods and convert
    themselves into a ``DrivenBlochHamiltonian`` through
    ``to_driven_hamiltonian``.
    """

    model_params: object
    drive_params: DriveParameters

    def __post_init__(self):
        self.units = resolve_units(self.model_params, self.drive_params)
        self.hbar = self.units.hbar
        self.e_charge = self.units.e_charge
        self.omega = self.drive_params.omega
        self.AL = self.drive_params.AL
        self.AR = self.drive_params.AR
        self.polarization_axis = self.drive_params.polarization_axis
        self.period = self.drive_params.period
