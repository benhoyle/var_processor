"""Variance Processing Unit."""

import numpy as np
from src.var_processor.covariance import CovarianceUnit
from src.var_processor.power_iterator import PowerIterator


def project(input_data, ev):
    """Project input using eigenvector.

    Args:
        input_data: 1D numpy array of length 'size'.
        ev: eigenvector - 1D numpy array of length 'size'.
    """
    return np.dot(ev.T, input_data)


def reconstruct(ev, r):
    """Reconstruct a version of the input using projected input.

    Args:
        ev: eigenvector - 1D numpy array of length 'size'.
        r: a scalar indicating a projection.
    """
    return r*ev


class VPU:
    """Variance processing unit."""

    def __init__(self, size):
        """Initialise.

        Args:
            size: integer setting the 1D size of an input;
        """
        self.cu = CovarianceUnit(size)
        self.pi = PowerIterator(size)
        self.size = size

    def iterate(self, input_data):
        """Iterate through one discrete timestep.

        Args:
            input_data: 1D numpy array of length self.size.

        Returns:
            residual: input minus the reconstructed input (1D array)
            r: scalar feature detection output
        """
        # Update covariance matrix
        self.update_cov(input_data)
        cov = self.cu.covariance
        # Power iterate
        self.pi.iterate(cov=cov)
        ev = self.pi.eigenvector
        # Project
        r = project(input_data, ev)
        r = self.process_r(r)
        # Reconstruct
        input_hat = reconstruct(ev, r)
        # Determine output
        residual = input_data - input_hat
        return r, residual

    def update_cov(self, input_data):
        """Update the covariance matrix.

        Use this to bed in the covariance.

        Args:
            input_data: 1D numpy array of length self.size.
        """
        self.cu.update(input_data)

    def process_r(self, r):
        """Perform post-processing on scalar r.

        Args:
            r: scalar.
        """
        return r

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size)


class VPUNonLin(VPU):
    """VPU with non-linear clamping."""

    def process_r(self, r):
        """Perform post-processing on scalar r.

        Args:
            r: scalar.
        """
        return np.where(r > np.random.rand(), 1, 0)
