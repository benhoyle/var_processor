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


def combine(r_forward, r_backward):
    """Combine two estimates of r.

    We might later want to weight the estimates.

    Args:
        r_forward: scalar forward estimate of r.
        r_backward: feedback estimate of r.

    Returns:
        r_combined: a single estimate from the two.

    """
    # Start with simple average
    r_combined = (r_forward + r_backward) / 2
    return r_combined


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

    def iterate(self, input_data, r_backward=None):
        """Iterate through one discrete timestep.

        Args:
            input_data: 1D numpy array of length self.size.
            r_backward: scalar value indicating a prediction of r.

        Returns:
            input_hat: 1D array containing the predicted input
            r: scalar feature detection output

        """
        # Update covariance matrix
        self.update_cov(input_data)
        cov = self.cu.covariance
        # Power iterate
        self.pi.iterate(cov=cov)
        ev = self.pi.eigenvector
        # Project
        r_forward = project(input_data, ev)
        # Add for backward compatibility
        if not r_backward:
            r_backward = r_forward
        # Combine forward & back estimates
        r_combined = combine(r_forward, r_backward)
        # Reconstruct
        input_hat = reconstruct(ev, r_combined)
        return r_combined, input_hat

    def update_cov(self, input_data):
        """Update the covariance matrix.

        Use this to bed in the covariance.

        Args:
            input_data: 1D numpy array of length self.size.
        """
        self.cu.update(input_data)

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size)


class BufferVPU(VPU):
    """VPU with time buffering."""

    def __init__(self, size, time_len=1):
        """Initialise object.

        We might want to later have different time_lens for forward
        and backward buffering.

        Arg:
            size - size of input data as 1D array.
            time_line - number of time samples to buffer.
        """
        self.time_len = time_len
        # Add buffer for input
        self.forward_buffer = np.zeros(shape=(size, time_len))
        self.backward_buffer = np.zeros(shape=(1, time_len))
        # Set up VPU with input as flattened buffer
        super(BufferVPU, self).__init__(size*time_len)

    def iterate(self, input_data, r_backward=0):
        """Iterate - same interfaces as parent.

        input_data is of length size.
        """
        # Add input to buffer
        self.forward_buffer = np.roll(
            self.forward_buffer, -1, axis=1
        )
        # Add frame to end of buffer
        self.forward_buffer[..., -1] = input_data.flatten()
        # Add r_backward to rear buffer
        self.backward_buffer = np.roll(
            self.backward_buffer, -1, axis=1
        )
        # Add new r_backward to back buffer
        self.backward_buffer[..., -1] = r_backward
        # Compute average r backward from buffer
        average_r_back = np.mean(self.backward_buffer, axis=1)
        # Flatten buffer and provide as input to parent method
        r, input_hat = super(BufferVPU, self).iterate(
            self.forward_buffer.reshape(-1, 1),
            average_r_back
        )
        # "Unbuffer" input_hat by averaging in time
        unbuffered = np.mean(
            input_hat.reshape(-1, self.time_len),
            axis=1
        )
        return r, unbuffered

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size, self.time_len)


class VPUNonLin(BufferVPU):
    """VPU with non-linearity on outputs."""

    def iterate(self, input_data, r_backward=None):
        """Iterate - same interfaces as parent.

        input_data is of length size.
        """
        r, input_hat = super(VPUNonLin, self).iterate(
            input_data,
            r_backward
        )
        # Apply non-linearity to output r
        r_non_lin = non_linearity(r)
        # Apply non-lienarity to output input prediction
        input_hat_non_lin = non_linearity(input_hat)
        return r_non_lin, input_hat_non_lin
