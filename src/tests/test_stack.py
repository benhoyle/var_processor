"""Test Stack.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.stack import Stack
from src.var_processor.pb_threshold import signal_pre_processor


def test_stack():
    """Simple test of stack initialisation."""
    stack = Stack(4, 256)
    assert stack.num_stages == 4
    assert len(stack.stages) == 4
    lengths = stack.get_lengths()
    assert lengths == ([64, 16, 4, 1], [256, 64, 16, 4])


def test_stack_forward():
    """Test a forward pass through the stack."""
    size = 256
    stack = Stack(4, size)
    mean = np.asarray([127]*size).reshape(-1, 1)
    for _ in range(0, 256):
        # Convert to ternary
        # Generate fake data
        data_in = np.random.randint(254, size=(size, 1))
        input_signal = signal_pre_processor(data_in, mean)
        cause = stack.forward(input_signal)
        assert cause.item() in [-1, 0, 1]


def test_stack_backward():
    """Test backward passes through stack."""
    size = 256
    stack = Stack(4, size)
    for _ in range(0, 256):
        # Convert to ternary
        # Generate fake data
        data_in = np.random.randint(low=-1, high=2, size=(1, 1))
        pred_inputs = stack.backward(data_in)
        assert pred_inputs.shape[0] == stack.input_len
        assert pred_inputs.max() <= 1 and pred_inputs.min() >= -1
