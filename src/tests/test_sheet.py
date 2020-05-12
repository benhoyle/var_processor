"""Test Sheet.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.sheet import Sheet
from src.var_processor.pb_threshold import signal_pre_processor


def test_sheet():
    """Simple test of sheet initialisation."""
    sheet = Sheet(4, 256, 6)
    assert sheet.stack_len == 6
    assert len(sheet.stacks) == 6
    assert sheet.causes.shape == (1, 6)
    assert sheet.stack_inputs.shape == (256, 6)
    # Test no explicit stack_len
    sheet = Sheet(4, 256)
    assert sheet.stack_len == 4
    assert len(sheet.stacks) == 4
    assert sheet.causes.shape == (1, 4)
    assert sheet.stack_inputs.shape == (256, 4)


def test_sheet_iterate():
    """Test a forward pass through the stack."""
    size = 256
    sheet = Sheet(4, size)
    mean = np.asarray([127]*size).reshape(-1, 1)
    for _ in range(0, 10):
        # Convert to ternary
        # Generate fake data
        data_in = np.random.randint(254, size=(size, 1))
        input_signal = signal_pre_processor(data_in, mean)
        causes, stack_inputs = sheet.iterate(input_signal)
        assert stack_inputs.shape[1] == 4
        assert causes.max() <= 1 and causes.min() >= -1
        assert stack_inputs.max() <= 1 and stack_inputs.min() >= -1
    # Test with passed backward values
    rand_back = np.random.randint(low=-1, high=2, size=(1, 4))
    causes, stack_inputs = sheet.iterate(input_signal, rand_back)
