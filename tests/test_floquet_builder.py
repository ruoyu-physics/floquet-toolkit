from functools import partial

import numpy as np

from floquet_toolkit.builders import FloquetBuilder
from floquet_toolkit.builtin_models import DiracParameters, driven_dirac_model
from floquet_toolkit.calculators.floquet_state_provider import FloquetStateProvider
from floquet_toolkit.config import DriveParameters, FloquetParameters

DIRAC_PARAMS = DiracParameters()
DRIVE_PARAMS = DriveParameters(AL=3.0e-9, AR=3.0e-9)
K_POINTS = [(0.0, 0.0), (1.2e8, -0.8e8)]


def build_model():
    return driven_dirac_model(DIRAC_PARAMS, DRIVE_PARAMS)


def diagonalize_builder(kx: float, ky: float, floquet_params: FloquetParameters):
    model = build_model()
    builder = FloquetBuilder(
        partial(model.Ht, kx=kx, ky=ky),
        model.omega,
        model.hbar,
        floquet_params,
    )
    floquet_matrix = builder.compute_floquet_hamiltonian()
    quasi_energies, floquet_states = np.linalg.eigh(floquet_matrix)
    return builder, floquet_matrix, quasi_energies, floquet_states


def selected_state_and_quasienergy(kx: float, ky: float, floquet_params: FloquetParameters):
    model = build_model()
    provider = FloquetStateProvider(model, floquet_params)
    quasi_energies, _ = provider.diagonalize_floquet_hamiltonian(kx, ky)
    state_index, _, floquet_state = provider.select_floquet_state(
        kx,
        ky,
        band="conduction",
        mode="overlap",
    )
    time_grid = np.linspace(0.0, DRIVE_PARAMS.period, 11, endpoint=False)
    reconstructed = provider.reconstruct_floquet_state(floquet_state, time=time_grid)
    reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True)
    return quasi_energies[state_index], reconstructed


def min_time_overlap(reference_state: np.ndarray, trial_state: np.ndarray) -> float:
    overlaps = np.einsum("ij,ij->i", np.conj(reference_state), trial_state)
    return float(np.min(np.abs(overlaps)))


def test_matrix_size_consistency():
    floquet_params = FloquetParameters(n_trunc=4, n_harmonics=2, n_time=61)
    builder, floquet_matrix, quasi_energies, floquet_states = diagonalize_builder(
        kx=0.0,
        ky=0.0,
        floquet_params=floquet_params,
    )

    expected_dimension = 2 * floquet_params.n_blocks
    harmonics = builder.compute_fourier_harmonics()

    assert harmonics.shape == (2 * floquet_params.n_harmonics + 1, 2, 2)
    assert floquet_matrix.shape == (expected_dimension, expected_dimension)
    assert quasi_energies.shape == (expected_dimension,)
    assert floquet_states.shape == (expected_dimension, expected_dimension)
    assert np.allclose(floquet_matrix, floquet_matrix.conj().T)
    assert np.allclose(
        floquet_states.conj().T @ floquet_states,
        np.eye(expected_dimension),
        atol=1e-12,
    )


def test_selected_quasienergy_converges_with_resolution():
    configs = [
        FloquetParameters(n_trunc=3, n_harmonics=1, n_time=31),
        FloquetParameters(n_trunc=5, n_harmonics=1, n_time=61),
        FloquetParameters(n_trunc=5, n_harmonics=2, n_time=61),
        FloquetParameters(n_trunc=7, n_harmonics=2, n_time=121),
    ]

    for kx, ky in K_POINTS:
        results = [selected_state_and_quasienergy(kx, ky, params) for params in configs]
        ref_quasienergy, ref_state = results[-1]

        coarse_error = abs(results[0][0] - ref_quasienergy)
        medium_error = abs(results[1][0] - ref_quasienergy)
        refined_error = abs(results[2][0] - ref_quasienergy)

        assert medium_error < coarse_error
        assert refined_error < coarse_error
        assert refined_error <= 6.0 * medium_error

        for quasienergy, reconstructed_state in results[:-1]:
            assert min_time_overlap(ref_state, reconstructed_state) > 1.0 - 5.0e-9
