"""Test Power Iterator - 8bit.
Run: pytest --cov=src --cov-report term-missing

Can replace other test if we use it in due course.
"""

import numpy as np
from src.var_processor.power_iterator8bit import (
    PowerIterator, normalise
)


def init_power(size):
    """Helper function."""
    # Test with length = size
    cov = np.random.uniform(low=-127, high=127, size=(size, size))
    cov = np.dot(cov.T, cov)//(127*np.sqrt(size))
    # Clip to avoid under/overflow
    cov = np.clip(cov, -127, 127)
    print(cov)
    random_cov = cov.astype(np.int8)
    # Generate test power iterator
    power = PowerIterator(size)
    power.load_covariance(random_cov)
    for _ in range(0, 1000):
        power.iterate()
    return power, random_cov


def test_normalise():
    """Test normalising an array using the L2 norm."""
    rand_vals = np.random.randint(low=-127, high=128, size=(4, 1))
    # Norm using function
    scaled = normalise(rand_vals)
    # Norm using linalg function
    linalg = ((rand_vals / np.linalg.norm(rand_vals))*127).astype(np.int8)
    assert np.allclose(scaled, linalg, atol=5)


def test_eigenvector():
    """Test power iterator is finding the eigenvector and value."""
    for _ in range(0, 100):
        random_size = np.random.randint(low=2, high=6)
        power, cov = init_power(random_size)
        evec = power.eigenvector
        evalue = power.eigenvalue
        # Use numpy linear algebra to determine eigenvectors and values
        w, v = np.linalg.eig(cov)
        # Check eigenvectors are close (abs removes difference in sign)
        assert np.allclose(
            abs(evec.T), abs(v[:, np.argmax(w)]*127), atol=10)
        # Check eigenvalues are close
        assert np.allclose(evalue, w[np.argmax(w)], atol=5)
