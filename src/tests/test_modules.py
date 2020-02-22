"""Tests.
Run: pytest --cov=src
"""

import numpy as np
from src.var_processor.pb_threshold import get_rand_ints, pb_threshold
from src.var_processor.time_buffer import Buffer
from src.var_processor.time_series import TimeSeries


def test_get_rand_ints():
    """Test random number generator."""
    shape = (2, 2)
    bit_size = 2
    rand_ints = get_rand_ints(bit_size, shape)
    assert rand_ints.shape == shape
    assert rand_ints.max() < 2**bit_size
    shape = (4, 3)
    bit_size = 4
    rand_ints = get_rand_ints(bit_size, shape)
    assert rand_ints.shape == shape
    assert rand_ints.max() < 2**bit_size


def test_pb_threshold():
    """Test probabilistic binary thresholding."""
    shape = (2, 2)
    bit_size = 8
    rand_ints = get_rand_ints(bit_size, shape).astype(np.uint8)
    binary = pb_threshold(rand_ints)
    assert binary.max() <= 1
    assert binary.min() >= 0
    assert rand_ints.shape == binary.shape


def test_buffer():
    """Test the buffer object and methods."""
    b = Buffer(2, 2, 2)
    a = np.arange(0, 4).reshape(2, 2)
    b.add(a)
    assert np.array_equal(b.latest, a)
    b.add(a)
    assert np.array_equal(b.latest, a)
    assert b.flag is True
    b.add(a)
    assert np.array_equal(b.latest, a)
    assert b.flag is False
    average = b.average
    # Average will equal a
    assert np.array_equal(average, a)
    fb_theta = (a*2 - 1)
    b.feedback(fb_theta)
    assert np.array_equal(b.feedback_theta, fb_theta)
    output = b.output
    assert np.array_equal(output, np.clip(a - (a*2 - 1), 0, 1))


def test_time_series():
    """Test time series object."""
    ts = TimeSeries(2, 2, 2)
    a = np.ones(shape=(2, 2))
    assert len(ts.time_series) == 1
    ts.add(a)
    ts.add(a)
    ts.add(a)
    assert len(ts.time_series) == 2
    out = ts.output()
    assert out.shape[2] == 2
