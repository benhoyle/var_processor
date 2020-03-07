"""Tests.
Run: pytest --cov=src
"""

import numpy as np
from src.var_processor.pb_threshold import get_rand_ints, pb_threshold
from src.var_processor.time_buffer import Buffer
from src.var_processor.time_series import TimeSeries, process_array


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
    b.feedforward(a)
    assert np.array_equal(b.latest, a)
    b.feedforward(a)
    assert np.array_equal(b.latest, a)
    b.feedforward(a)
    assert np.array_equal(b.latest, a)
    average = b.ff_output
    # Average will equal a
    assert average.shape == a.shape
    assert average.max() <= 1
    assert average.min() >= 0
    b.feedback(a)
    fb_output = b.fb_output
    assert fb_output.shape == a.shape
    assert fb_output.max() <= 1
    assert fb_output.min() >= 0


def test_timeseries():
    """Test the time series object."""
    ts = TimeSeries(4, 3, 6, 8)
    # Check everything is initialised to 0
    assert np.array_equal(ts.ff_output, np.zeros((8, 4, 3)))
    assert np.array_equal(ts.fb_output, np.zeros((8, 4, 3)))
    assert np.array_equal(ts.latest, np.zeros((8, 4, 3)))
    a = np.random.randint(255, size=(4, 3), dtype=np.uint8)
    pb_a = pb_threshold(a)
    ts.add(pb_a)


def test_processarray():
    """Test process array function."""
    a = np.arange(0, 4, dtype=np.uint8)
    b = a.reshape(1, 1, -1)
    assert process_array(b) == 1.5
