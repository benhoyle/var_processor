"""Test Stage.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.stage import Stage, pad_array


def test_pad_array():
    """Test array padding."""
    array_in = np.arange(0, 10).reshape(-1, 1)
    out = pad_array(array_in, 10)
    assert np.array_equal(array_in, out)
    out = pad_array(array_in, 8)
    assert np.array_equal(array_in[:8], out)
    assert out.shape[0] == 8
    out = pad_array(array_in, 12)
    assert np.array_equal(array_in, out[0:10])
    assert out.shape[0] == 12


def test_stage():
    """Test  stage."""
    # Initialise time stage
    stages = Stage(3, 10)
    assert len(stages.vpus) == 10
    assert not stages.causes.any()
    assert not stages.pred_inputs.any()
    assert "10" in stages.__repr__()
    # Check data in
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(stages.size, 1))
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes1, pred_inputs1 = stages.iterate(data_in, r_backwards)
    # assert stages.causes.any()
    # assert stages.pred_inputs.any()
    for _ in range(0, 1000):
        data_in = np.random.randint(2, size=(stages.size, 1))
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes2, pred_inputs2 = stages.iterate(data_in, r_backwards)
    assert not np.array_equal(causes1, causes2)
    assert not np.array_equal(pred_inputs1, pred_inputs2)
