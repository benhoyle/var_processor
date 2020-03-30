"""Test Covariance.
Run: pytest --cov=src --cov-report term-missing
"""

import numpy as np
from src.var_processor.covariance import CovarianceUnit


def test_covariance_unit():
    """Test the covariance unit."""
    # Test initialising
    cov_unit = CovarianceUnit(2)
    assert not cov_unit.x_sum.any()
    assert not cov_unit.square_sum.any()
    # Test updating with data
    ones = np.ones(shape=(2, 1))
    cov_unit.update(ones)
    assert cov_unit.count == 1
    assert np.array_equal(cov_unit.x_sum, ones)
    assert np.array_equal(cov_unit.mean, ones)
    assert not cov_unit.covariance.any()
    threes = ones*3
    cov_unit.update(threes)
    assert cov_unit.count == 2
    assert np.array_equal(cov_unit.x_sum, ones+threes)
    assert cov_unit.square_sum.any()
    assert np.array_equal(cov_unit.mean, ones*2)
    assert cov_unit.covariance.any()


def test_covariance_computation():
    """Statistical test that cov unit is determining the covariance."""
    # Generate random positive definite matrix
    cov = np.random.randn(3, 3)
    cov = np.dot(cov, cov.T)
    cov = cov / cov.max()
    # Generate desired mean
    mean = np.random.randn(3, 1)
    # Use Cholesky decomposition to get L
    L = np.linalg.cholesky(cov)
    cov_unit = CovarianceUnit(3)
    for _ in range(0, 10000):
        sample = np.dot(L, np.random.randn(3, 1)) + mean
        cov_unit.update(sample)
    # Check within 10%
    assert np.allclose(mean, cov_unit.mean, rtol=0.20)
    assert np.allclose(cov, cov_unit.covariance, rtol=0.20)
