"""Test Power Iterator.
Run: pytest --cov=src --cov-report term-missing
"""

import numpy as np
from src.var_processor.power_iterator import PowerIterator


def test_power_iterator():
    """Test power iterator."""
    # Test initialise
    power = PowerIterator(2)
    ev1 = power.ev
    assert ev1.any()
    assert not power.cov.any()
    # Check logic to avoid ev = nan
    ev1_a = power.iterate()
    assert np.array_equal(ev1, ev1_a)
    # Check update with non-zero cov
    random_cov = np.random.randint(255, size=(2, 2))
    random_cov = random_cov / random_cov.max()
    power.load_covariance(random_cov)
    assert np.array_equal(power.cov, random_cov)
    ev2 = power.iterate()
    assert not np.array_equal(ev1, ev2)
    assert np.array_equal(ev2, power.eigenvector)
    assert power.eigenvalue > 0
    # Check passing a cov
    ev3 = power.iterate(cov=random_cov)
    assert not np.array_equal(ev2, ev3)


def init_power(size):
    """Helper function."""
    # Test with length = size
    cov = np.random.randn(size, size)
    cov = np.dot(cov, cov.T)
    cov = cov / cov.max()
    # Generate test power iterator
    power = PowerIterator(size)
    power.load_covariance(cov)
    for _ in range(0, 1000):
        power.iterate()
    return power, cov


def test_power_computation():
    """Test power iterator is finding the eigenvector and value."""
    power, cov = init_power(3)
    evec = power.eigenvector
    evalue = power.eigenvalue
    # Use numpy linear algebra to determine eigenvectors and values
    w, v = np.linalg.eig(cov)
    # Check eigenvectors are close (abs removes difference in sign)
    assert np.allclose(
        abs(evec.T), abs(v[:, np.argmax(w)]), rtol=0.05, atol=0.05)
    # Check eigenvalues are close
    assert np.allclose(evalue, w[np.argmax(w)], rtol=0.05, atol=0.05)


def test_feature_scaling_2():
    """Test that the features are scaled to have max of 1."""
    # Test with length = 2
    p_2, _ = init_power(2)
    e_2 = p_2.eigenvector
    assert np.array_equal(p_2.feature, e_2*(np.sqrt(2)/2))
    assert np.max(np.abs(p_2.feature)) <= 1


def test_feature_scaling_3():
    """Test that the features are scaled to have max of 1."""
    # Test with length = 3
    p_3, _ = init_power(3)
    # Not a timing thing - added time.sleep still had error
    # Why does 2 work but not 3? Works when in different functions!
    assert np.array_equal(p_3.feature, p_3.eigenvector*(np.sqrt(3)/3))
    assert np.max(np.abs(p_3.feature)) <= 1
