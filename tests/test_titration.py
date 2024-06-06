import pytest
from inline_snapshot import snapshot, outsource
from libeq.data_structure import SolverData
from libeq import EqSolver

from inline_snapshot import external


@pytest.fixture
def solver_data():
    return SolverData.load_from_pyes("tests/data/cu_gly_solid.json")


def test_titration_fix(solver_data):
    solver_data.ionic_strength_dependence = False
    result, log_beta, log_ks, saturation_index, total_concentration = EqSolver(
        solver_data, mode="titration"
    )

    assert outsource(result.tobytes()) == snapshot(external("ec94ea85f0a6*.bin"))
    assert outsource(log_beta.tobytes()) == snapshot(external("d72a93ac3eac*.bin"))
    assert outsource(log_ks.tobytes()) == snapshot(external("1167b0e94b2a*.bin"))
    assert outsource(saturation_index.tobytes()) == snapshot(
        external("e23a0f3b589c*.bin")
    )
    assert outsource(total_concentration.tobytes()) == snapshot(
        external("0b166a60f644*.bin")
    )


def test_titration_variable(solver_data):
    solver_data.ionic_strength_dependence = True

    result, log_beta, log_ks, saturation_index, total_concentration = EqSolver(
        solver_data, mode="titration"
    )

    assert outsource(result.tobytes()) == snapshot(external("383cd7bd6892*.bin"))
    assert outsource(log_beta.tobytes()) == snapshot(external("0f794d0393dc*.bin"))
    assert outsource(log_ks.tobytes()) == snapshot(external("a348013df9fb*.bin"))
    assert outsource(saturation_index.tobytes()) == snapshot(
        external("8eecb627a156*.bin")
    )
    assert outsource(total_concentration.tobytes()) == snapshot(
        external("0b166a60f644*.bin")
    )
