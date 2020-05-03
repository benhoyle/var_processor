"""Test abstract classes."""

import numpy as np
from src.var_processor.abstract_classes import (
    AbstractBase, AbstractSubUnit, AbstractSignalProcessor, TransformMixin
)


def test_abstract_base():
    """Test base class."""
    size = 4
    base = AbstractBase(size)
    assert base.vec_len == size
    base.update_cov(None)
    base.reset()
    string = base.__repr__()
    assert "AbstractBase" in string and f"{size}" in string


def test_abstract_subunit():
    """Test subunit abstract class."""
    size = 4
    subunit = AbstractSubUnit(size)
    # Test parent methods and data
    assert subunit.vec_len == size
    subunit.update_cov(None)
    subunit.reset()
    string = subunit.__repr__()
    assert "AbstractSubUnit" in string and f"{size}" in string
    ev = subunit.eigenvector
    ev = subunit.eigenvalue
    cov = subunit.covariance


def test_transform_mixin():
    """Test transform mixin."""
    transformer = TransformMixin()
    ones = np.ones(2)
    zeros = np.zeros(2)
    cause = transformer.forward(ones)
    pred_inputs = transformer.backward(cause)
    assert np.array_equal(cause, pred_inputs)
    forward_output, backward_output = transformer.iterate(ones, zeros)
    assert np.array_equal(forward_output, ones)
    assert np.array_equal(backward_output, zeros)
    forward_output, backward_output = transformer.iterate(ones)
    assert np.array_equal(forward_output, ones)
    assert np.array_equal(backward_output, ones)


def test_signal_processor():
    """Test signal processor class."""
    vec_len = 4
    input_len = 256
    processor = AbstractSignalProcessor(4, 256)
    assert processor.vec_len == vec_len
    assert processor.input_len == input_len
    processor.update_cov(None)
    processor.reset()
    string = processor.__repr__()
    assert "AbstractSignalProcessor" in string
    assert f"{vec_len}" in string
    assert f"{input_len}" in string
