import numpy as np

from floquet_toolkit import (
    DiracModel,
    DiracParameters,
    FloquetLocalManager,
    GrapheneModel,
    GrapheneParameters,
    RotatingFrameDiracModel,
)
from floquet_toolkit.config import DriveParameters, FloquetParameters


DIRAC_PARAMS = DiracParameters()
GRAPHENE_PARAMS = GrapheneParameters()
DRIVE_PARAMS = DriveParameters(AL=3.0e-9, AR=0.0)
FLOQUET_PARAMS = FloquetParameters(n_trunc=3, n_harmonics=1, n_time=21)


def test_builtin_model_classes_convert_to_driven_hamiltonians():
    dirac = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    graphene = GrapheneModel(GRAPHENE_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    rotating = RotatingFrameDiracModel(
        DIRAC_PARAMS,
        DRIVE_PARAMS,
    ).to_driven_hamiltonian()

    for model in (dirac, graphene, rotating):
        sample = model.Ht(0.0, 0.0, 0.0)
        assert sample.shape == (2, 2)
        assert np.allclose(sample, sample.conj().T)
        assert model.units == DRIVE_PARAMS.units


def test_floquet_manager_diagonalize_floquet_hamiltonian_returns_square_eigensystem():
    model = DiracModel(DIRAC_PARAMS, DRIVE_PARAMS).to_driven_hamiltonian()
    manager = FloquetLocalManager(model, FLOQUET_PARAMS)

    quasi_energy, floquet_states = manager.diagonalize_floquet_hamiltonian(0.0, 0.0)

    expected_dimension = model.dimension * FLOQUET_PARAMS.n_blocks
    assert quasi_energy.shape == (expected_dimension,)
    assert floquet_states.shape == (expected_dimension, expected_dimension)
    assert np.allclose(
        floquet_states.conj().T @ floquet_states,
        np.eye(expected_dimension),
        atol=1e-12,
    )
