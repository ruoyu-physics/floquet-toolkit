"""Built-in two-band Dirac-style driven model factories."""

from functools import partial
import numpy as np

from .core.driven_bloch_hamiltonian import DrivenBlochHamiltonian
from .config import DriveParameters, E_CHARGE, HBAR, PhysicsParameters

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_PLUS = 0.5 * (SIGMA_X + 1j * SIGMA_Y)
SIGMA_MINUS = 0.5 * (SIGMA_X - 1j * SIGMA_Y)
IDENTITY_2 = np.eye(2, dtype=complex)

GRAPHENE_BOND_LENGTH = 1.42e-10  # m


def driven_dirac_model(
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a lab-frame massive Dirac model with circular drive components.

    Args:
        physics_params: Static model parameters such as velocity and mass.
        drive_params: Drive frequency and circular-polarization amplitudes.

    Returns:
        ``DrivenBlochHamiltonian`` with an analytic static Berry curvature.
    """
    AL = drive_params.AL
    AR = drive_params.AR
    mass = physics_params.mass
    vf = physics_params.vf
    omega = drive_params.omega

    def H_static(kx: float, ky: float) -> np.ndarray:
        """Static Dirac Hamiltonian."""
        return mass * SIGMA_Z + HBAR * vf * (kx * SIGMA_X + ky * SIGMA_Y)

    def H_t(t, kx, ky, AL, AR):
        Ax = (AL + AR) / np.sqrt(2) * np.cos(omega * t)
        Ay = (AL - AR) / np.sqrt(2) * np.sin(omega * t)

        shifted_kx = kx - E_CHARGE / HBAR * Ax
        shifted_ky = ky - E_CHARGE / HBAR * Ay
        return H_static(shifted_kx, shifted_ky)

    def analytic_static_berry_curvature(kx, ky, band="conduction"):
        if band in ("conduction", 1):
            return -0.5 * mass * (HBAR * vf) ** 2 / (
                ((HBAR * vf) ** 2 * (kx**2 + ky**2) + mass**2) ** (3 / 2)
            )
        if band in ("valence", 0):
            return 0.5 * mass * (HBAR * vf) ** 2 / (
                ((HBAR * vf) ** 2 * (kx**2 + ky**2) + mass**2) ** (3 / 2)
            )
        raise ValueError("Band must be 'conduction', 'valence', 0, or 1")

    return DrivenBlochHamiltonian(
        H_t=partial(H_t, AL=AL, AR=AR),
        omega=omega,
        H_static=H_static,
        analytic_static_berry_curvature=analytic_static_berry_curvature,
    )

def driven_graphene_model(
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a nearest-neighbor graphene tight-binding model with a mass gap."""
    AL = drive_params.AL
    AR = drive_params.AR
    mass = 0
    vf = physics_params.vf
    omega = drive_params.omega

    # Infer the nearest-neighbor hopping from the Dirac velocity:
    # v_F = (3/2) * a_cc * t / hbar.
    hopping = 2.0 * HBAR * vf / (3.0 * GRAPHENE_BOND_LENGTH)

    delta_1 = GRAPHENE_BOND_LENGTH * np.array([0.0, 1.0])
    delta_2 = GRAPHENE_BOND_LENGTH * np.array([np.sqrt(3.0) / 2.0, -0.5])
    delta_3 = GRAPHENE_BOND_LENGTH * np.array([-np.sqrt(3.0) / 2.0, -0.5])
    neighbor_vectors = np.array([delta_1, delta_2, delta_3])

    def structure_factor(kx: float, ky: float) -> complex:
        k_vector = np.array([kx, ky], dtype=float)
        phases = np.exp(1j * (neighbor_vectors @ k_vector))
        return np.sum(phases)

    def H_static(kx: float, ky: float) -> np.ndarray:
        """Nearest-neighbor graphene Bloch Hamiltonian with a sublattice gap."""
        gamma_k = structure_factor(kx, ky)
        return np.array(
            [
                [mass, -hopping * gamma_k],
                [-hopping * np.conj(gamma_k), -mass],
            ],
            dtype=complex,
        )

    def H_t(t, kx, ky, AL, AR):
        Ax = (AL + AR) / np.sqrt(2.0) * np.cos(omega * t)
        Ay = (AL - AR) / np.sqrt(2.0) * np.sin(omega * t)
        shifted_kx = kx - E_CHARGE / HBAR * Ax
        shifted_ky = ky - E_CHARGE / HBAR * Ay
        return H_static(shifted_kx, shifted_ky)

    return DrivenBlochHamiltonian(
        H_t=partial(H_t, AL=AL, AR=AR),
        omega=omega,
        H_static=H_static,
    )


def rotating_frame_dirac_model(
    physics_params: PhysicsParameters,
    drive_params: DriveParameters,
):
    """Create a user-specified rotating-frame version of the Dirac model.

    The rotating-frame transformation is encoded directly in the returned
    Hamiltonian callables; downstream builders treat it like any other periodic
    Hamiltonian.
    """
    AL = drive_params.AL
    AR = drive_params.AR
    mass = physics_params.mass
    vf = physics_params.vf
    omega = drive_params.omega

    def H_static(kx: float, ky: float) -> np.ndarray:
        """Static Dirac Hamiltonian in the rotating frame."""
        return (mass - HBAR * omega / 2) * SIGMA_Z

    def H_periodic(t, kx, ky, AL, AR):
        rotation_matrix = np.array(
            [
                [np.cos(omega * t), -np.sin(omega * t)],
                [np.sin(omega * t), np.cos(omega * t)],
            ]
        )
        k_vector = np.array([kx, ky])
        Ax = (AL + AR) / np.sqrt(2) * np.cos(omega * t)
        Ay = (AL - AR) / np.sqrt(2) * np.sin(omega * t)
        A_vector = np.array([Ax, Ay])

        rotated_k = rotation_matrix @ (k_vector - E_CHARGE * A_vector / HBAR)
        return HBAR * vf * (rotated_k[0] * SIGMA_X + rotated_k[1] * SIGMA_Y)

    def H_t(t, kx, ky, AL, AR):
        return H_static(kx, ky) + H_periodic(t, kx, ky, AL, AR)

    return DrivenBlochHamiltonian(
        H_t=partial(H_t, AL=AL, AR=AR),
        omega=omega,
        H_static=H_static,
    )
