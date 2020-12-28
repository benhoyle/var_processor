import numpy as np
from src.var_processor.surface_stage import AbstractStage, decompose, recompose


# @pytest.fixture
# def stage():
#     """Returns a stage object."""
#     return AbstractStage()

class TestSurfaceStage:
    stage = AbstractStage()

    def test_forward_pre_processing(self):
        self.stage.forward_pre_processing(None)

    def test_forward_post_processing(self):
        self.stage.forward_post_processing(None)

    def test_forward(self):
        assert False

    def test_backward_pre_processing(self):
        assert False

    def test_backward_post_processing(self):
        assert False

    def test_backward(self):
        assert False

    def test_residuals(self):
        assert False

    def test_get_surfaces(self):
        assert False

    def test_lower_input(self):
        assert False


def test_decompose():
    test_array = np.ones(shape=(2, 2), dtype=np.uint8)
    [A, H, V, D] = decompose(test_array)
    assert A[0] == 1
    assert H[0] == 0
    assert V[0] == 0
    assert D[0] == 0
    # Watch out - it won't do fractions due to int data type
    # Test horizontal
    test_array = np.array([[4, 4], [0, 0]], dtype=np.uint8)
    [A, H, V, D] = decompose(test_array)
    assert A[0] == 2
    assert H[0] == 2
    assert V[0] == 0
    assert D[0] == 0
    # Test vertical
    test_array = np.array([[4, 0], [4, 0]], dtype=np.uint8)
    [A, H, V, D] = decompose(test_array)
    assert A[0] == 2
    assert H[0] == 0
    assert V[0] == 2
    assert D[0] == 0
    # Test diagonal
    test_array = np.array([[4, 0], [0, 4]], dtype=np.uint8)
    [A, H, V, D] = decompose(test_array)
    assert A[0] == 2
    assert H[0] == 0
    assert V[0] == 0
    assert D[0] == 2


def test_recompose():
    """Test recomposing."""
    test_surfaces = [np.array([[1]]), np.array([[0]]), np.array([[0]]), np.array([[0]])]
    image = recompose(test_surfaces)
    assert np.array_equal(image, np.ones(shape=(2, 2)))
    # Watch out - it won't do fractions due to int data type
    # Test horizontal
    test_surfaces = [np.array([[2]]), np.array([[2]]), np.array([[0]]), np.array([[0]])]
    image = recompose(test_surfaces)
    assert np.array_equal(image, np.array([[4, 4], [0, 0]]))
    # Test vertical
    test_surfaces = [np.array([[2]]), np.array([[0]]), np.array([[2]]), np.array([[0]])]
    image = recompose(test_surfaces)
    assert np.array_equal(image, np.array([[4, 0], [4, 0]]))
    # Test diagonal
    test_surfaces = [np.array([[2]]), np.array([[0]]), np.array([[0]]), np.array([[2]])]
    image = recompose(test_surfaces)
    assert np.array_equal(image, np.array([[4, 0], [0, 4]]))