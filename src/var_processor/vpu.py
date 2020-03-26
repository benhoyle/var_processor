"""Variance Processing Unit."""

import numpy as np
from src.var_processor.covariance import CovarianceUnit
from src.var_processor.power_iterator import PowerIterator
from src.var_processor.pb_threshold import non_linearity


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
        residual = self.process_residual(residual)
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

    def process_residual(self, residual):
        """Perform post-processing on residual array.

        Args:
            residual: numpy array.
        """
        return residual

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
        return non_linearity(r)


class BufferVPU(VPU):
    """VPU with time buffering."""

    def __init__(self, size, time_len):
        """Initialise object.

        Arg:
            size - size of input data as 1D array.
            time_line - number of time samples to buffer.
        """
        self.time_len = time_len
        # Add buffer for input
        self.buffer = np.zeros(shape=(size, time_len))
        # Set up VPU with input as flattened buffer
        super(BufferVPU, self).__init__(size*time_len)

    def iterate(self, input_data):
        """Same interfaces as parent.

        input_data is of length size.
        """
        # Add input to buffer
        self.buffer = np.roll(self.buffer, -1, axis=1)
        # Add frame to end of buffer
        self.buffer[..., -1] = input_data.flatten()
        # Flatten buffer and provide as input to parent method
        return super(BufferVPU, self).iterate(self.buffer.reshape(-1, 1))

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size, self.time_len)
