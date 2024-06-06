import pytest
from inline_snapshot import snapshot, outsource
from libeq.data_structure import SolverData
from libeq import EqSolver

from inline_snapshot import external


@pytest.fixture
def solver_data():
    return SolverData.load_from_pyes("tests/data/cu_gly_solid.json")


def test_distribution_fix(solver_data):
    solver_data.ionic_strength_dependence = False
    result, log_beta, log_ks, saturation_index, total_concentration = EqSolver(
        solver_data, mode="distribution"
    )

    assert outsource(result.tobytes()) == snapshot(external("5c36e010b6ad*.bin"))
    assert outsource(log_beta.tobytes()) == snapshot(external("daf9c5a0eaf4*.bin"))
    assert outsource(log_ks.tobytes()) == snapshot(external("edbcdb45f49f*.bin"))
    assert outsource(saturation_index.tobytes()) == snapshot(
        external("a723ec678e7f*.bin")
    )
    assert outsource(total_concentration.tobytes()) == snapshot(
        external("4023b4451f5a*.bin")
    )


def test_distribution_variable(solver_data):
    solver_data.ionic_strength_dependence = True

    result, log_beta, log_ks, saturation_index, total_concentration = EqSolver(
        solver_data, mode="distribution"
    )

    assert outsource(result.tobytes()) == snapshot(external("e98bdbc4ab7f*.bin"))
    assert outsource(log_beta.tobytes()) == snapshot(external("6e4ffec14eb2*.bin"))
    assert outsource(log_ks.tobytes()) == snapshot(external("b4781f4b85b1*.bin"))
    assert outsource(saturation_index.tobytes()) == snapshot(
        external("68388a10667e*.bin")
    )
    assert outsource(total_concentration.tobytes()) == snapshot(
        external("bc1e07eaed78*.bin")
    )


# Run the tests
if __name__ == "__main__":
    pytest.main()
