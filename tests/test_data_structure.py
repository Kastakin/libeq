import pytest
from libeq.data_structure import SolverData



def test_load_from_txt(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("[[1,2,3],[4,5,6]]\n[[1,2,3],[4,5,6]]\n[1.1,2.2,3.3]\n2\n3\n")
    data = SolverData.load_from_txt(str(file_path))
    assert data.stoichiometry.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert data.solid_stoichiometry.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert data.beta.tolist() == [1.1, 2.2, 3.3]
    assert data.field2 == 2
    assert data.field3 == 3.0


def test_load_from_json(tmp_path):
    file_path = tmp_path / "data.json"
    file_path.write_text('{"field2": 10, "field3": 3.14}')
    data = SolverData.load_from_json(str(file_path))
    assert data.field2 == 10
    assert data.field3 == 3.14


# Run the tests
if __name__ == "__main__":
    pytest.main()
