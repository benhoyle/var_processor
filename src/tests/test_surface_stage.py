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

