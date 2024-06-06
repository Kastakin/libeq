import pytest
from inline_snapshot import snapshot, outsource
from libeq import PotentiometryOptimizer, SolverData

from inline_snapshot import external


@pytest.fixture
def potentiometry_data():
    return SolverData.load_from_bstac("tests/data/Zn-EDTA")


def test_potentiometry(potentiometry_data):
    betas, concs, b_error, cor_matrix, cov_matrix, return_extra = (
        PotentiometryOptimizer(potentiometry_data)
    )

    assert outsource(betas.tobytes()) == snapshot(external("2959115252bf*.bin"))
    assert outsource(concs.tobytes()) == snapshot(external("5b33bcbb29f9*.bin"))
    assert outsource(b_error.tobytes()) == snapshot(external("6df3497012ff*.bin"))
    assert outsource(cor_matrix.tobytes()) == snapshot(external("496c293c8c92*.bin"))
    assert outsource(cov_matrix.tobytes()) == snapshot(external("d87945d8f70e*.bin"))
