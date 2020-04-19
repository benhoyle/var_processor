"""Power Iterator."""

import numpy as np


class PowerIterator:
    """Module to determine an eigenvector using power iteration."""

    def __init__(self, length):
        """Initialise.

        Args:
            length: integer setting the 1D size of the eigenvector.
        """
        # Initialise eigenvector as random vector - NOTE 8 bit
        # THIS WILL BE RANDOM ANYWAY DUE TO INHERENT RANDOMNESS
        self.ev = np.random.randint(255, size=(length, 1))
        # Scale to have unit length (convert to integer values?)
        self.ev = self.ev / np.linalg.norm(self.ev)
        # Define placeholder for covariance matrix
        self.cov = np.zeros(shape=(length, length))
        # Define scaling factor as 1/sqrt(length)
        self.scaler = 1 / np.sqrt(length)

    def iterate(self, power=1, cov=None):
        """One pass of iteration."""
        # If a covariance is passed use to update
        if cov is not None:
            self.load_covariance(cov)
        # Check cov is not all zero - if all 0 you get nan
        if self.cov.any():
            # Times covariance by working eigenvector
            self.ev = np.matmul(np.power(self.cov, power), self.ev)
            # Scale to have unit length (convert to integer values?)
            self.ev = self.ev / np.linalg.norm(self.ev)
        return self.ev.copy()

    @property
    def eigenvector(self):
        """Return the top eigenvector."""
        return self.ev.copy()

    @property
    def eigenvalue(self):
        """Return associated eigenvalue."""
        top_1 = np.matmul(self.ev.T, self.cov)
        bottom = np.matmul(self.ev.T, self.ev)
        r = np.matmul(top_1, self.ev) / bottom
        return r

    @property
    def feature(self):
        """Return eigenvector scaled to ternary space."""
        return self.ev*self.scaler

    def load_covariance(self, cov):
        """Update the covariance matrix."""
        self.cov = cov
        # Put here an update of an existing matrix?
