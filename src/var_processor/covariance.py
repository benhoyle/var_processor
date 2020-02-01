"""Covariance Computation."""

import numpy as np


class CovarianceUnit:
    """A model to compute s scaled update online with no count."""

    def __init__(self, size):
        """Initialise.

        Args:
            size: integer setting the 1D size of an input;
        """
        self.size = size
        self.count = 0
        self.x_sum = np.zeros(shape=(size, 1))
        # Initialise to I
        self.square_sum = np.zeros(shape=(size, size))

    def update(self, x):
        """Add a data point x.

        x is a 1D numpy array of length 'size'.
        """
        self.count += 1
        self.x_sum += x
        x_dash = self.x_sum - self.count*x
        scale_factor = self.count*(self.count+1)
        self.square_sum += (scale_factor**-1)*np.dot(x_dash, x_dash.T)

    @property
    def mean(self):
        """Compute mean when requested."""
        return self.x_sum / self.count

    @property
    def covariance(self):
        """Compute covariance when requested."""
        return self.square_sum / self.count
