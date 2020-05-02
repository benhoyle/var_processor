"""Test Stage.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.stage import Stage, pad_array
from src.var_processor.pb_threshold import signal_pre_processor


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
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(stages.size, 1))
        residual_in = data_in.copy()
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes1, pred_inputs1 = stages.iterate(
            data_in,
            residual_in,
            r_backwards
        )
    # assert stages.causes.any()
    # assert stages.pred_inputs.any()
    for _ in range(0, 1000):
        data_in = np.random.randint(2, size=(stages.size, 1))
        residual_in = data_in.copy()
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes2, pred_inputs2 = stages.iterate(
            data_in,
            residual_in,
            r_backwards
        )
    assert not np.array_equal(causes1, causes2)
    assert not np.array_equal(pred_inputs1, pred_inputs2)


def test_stage_function():
    """Test more advanced stage function."""
    # Testing a stage
    size = 256
    vec_len = 4
    buf_length = 100
    stage = Stage(vec_len, size//vec_len)

    # Generate fake data
    data_in = np.random.randint(255, size=(size, 1))
    mean = np.asarray([128]*size).reshape(-1, 1)

    # Generate buffers for testing
    input_buffer = np.zeros(shape=(size, buf_length))
    pred_buffer = np.zeros(shape=(size, buf_length))
    r_buffer = np.zeros(shape=(size//vec_len, buf_length))
    residual_buffer = np.zeros(shape=(size, buf_length))

    for i in range(0, buf_length):
        # Convert to ternary
        input_signal = signal_pre_processor(data_in, mean)
        # Process stage
        stage.update_cov(data_in)
        causes = stage.forward(input_signal)
        pred_inputs = stage.backward(causes)
        input_buffer[:, i] = input_signal.ravel()
        pred_buffer[:, i] = pred_inputs.ravel()
        r_buffer[:, i] = causes.ravel()
        residual_buffer[:, i] = (data_in - pred_inputs).ravel()
    # Check for all ones
    assert r_buffer.sum() < 256*buf_length
