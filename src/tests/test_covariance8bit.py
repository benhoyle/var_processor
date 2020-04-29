"""Test Covariance 8-bit version.
Run: pytest --cov=src --cov-report term-missing

This can run in parallel with test_covariance and eventually replace
that file.

"""

import numpy as np

from src.var_processor.covariance8bit import CovarianceUnit
from src.tests.test_vpu import rand_same, rand_diff, rand_opposite


def test_different_sign():
    """Test applying with different sign."""
    size = 2
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        data = rand_same(negative=True)
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(cov_unit.covariance/127, np.cov(data_buffer))
    assert np.allclose(cov_unit.covariance/127, np.cov(data_buffer), atol=0.1)


def test_non_neg():
    """Test applying with binary only input (i.e. positive)."""
    size = 2
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        data = rand_same(negative=False)
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(cov_unit.covariance/254, np.cov(data_buffer))
    assert np.allclose(cov_unit.covariance/254, np.cov(data_buffer), atol=0.1)


def test_opposite():
    """Test with opposite signs."""
    size = 3
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        data = rand_opposite(size=size, negative=True)
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(data_buffer.mean(axis=1), "\n" )
    # print(cov_unit.covariance/127, "\n\n", np.cov(data_buffer))
    assert np.allclose(
        data_buffer.mean(axis=1), np.zeros(shape=(2, 1)), atol=0.1)
    assert np.allclose(cov_unit.covariance/127, np.cov(data_buffer), atol=0.2)


def test_diff_neg():
    """Test applying with ternary input with different elements."""
    size = 2
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        data = rand_diff(negative=True)
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(data_buffer.mean(axis=1), )
    # print(cov_unit.covariance/127, np.cov(data_buffer))
    assert np.allclose(
        data_buffer.mean(axis=1), np.zeros(shape=(2, 1)), atol=0.1)
    assert np.allclose(cov_unit.covariance/127, np.cov(data_buffer), atol=0.1)


def test_random():
    """Test applying with random ternary input."""
    size = 4
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        data = np.random.randint(low=-1, high=2, size=(size, 1))
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(data_buffer.mean(axis=1), )
    # print(cov_unit.covariance/127, "\n", np.cov(data_buffer))
    assert np.allclose(
        data_buffer.mean(axis=1), np.zeros(shape=(2, 1)), atol=0.1)
    assert np.allclose(cov_unit.covariance/127, np.cov(data_buffer), atol=0.15)


def test_half_half():
    """Test applying with random ternary input."""
    size = 4
    buf_length = 1000
    data_buffer = np.zeros(shape=(size, buf_length))
    cov_unit = CovarianceUnit(size)
    for i in range(0, buf_length):
        coin_flip = np.random.randint(2)
        if coin_flip:
            # Make all entries the same
            data = rand_same(size=4, negative=True)
        else:
            # Make entries random
            data = np.random.randint(low=-1, high=2, size=(size, 1))
        cov_unit.update_cov(data)
        data_buffer[:, i] = data.ravel()
    # Check covariance estimate is within range of actual estimate
    # print(data_buffer.mean(axis=1), )
    # print(cov_unit.covariance/127, "\n", np.cov(data_buffer))
    # print(cov_unit)
    assert np.allclose(
        data_buffer.mean(axis=1), np.zeros(shape=(2, 1)), atol=0.1)
    assert np.allclose(cov_unit.covariance/127, np.cov(data_buffer), atol=0.15)
