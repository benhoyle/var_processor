"""Variance Processing Unit."""

import numpy as np
from src.var_processor.covariance import CovarianceUnit
from src.var_processor.power_iterator import PowerIterator
from src.var_processor.pb_threshold import non_linearity


def project(vec_1, vec_2):
    """Project input using eigenvector.

    Args:
        vec1: 1D numpy array.
        vec2: 1D numpy array.
    """
    return np.dot(vec_1, vec_2)


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

    def iterate(self, input_data, r_backward):
        """Iterate through one discrete timestep.

        Args:
            input_data: 1D numpy array of length self.size.
            r_backward: scalar value indicating a prediction of r.

        Returns:
            pred_inputs: 1D array containing the predicted input
            r: scalar feature detection output

        """
        r_forward = self.forward(input_data)
        pred_inputs = self.backward(r_backward)
        return r_forward, pred_inputs

    def forward(self, input_data):
        """Forward pass to generate cause - r.

        Args:
            input_data: 1D numpy array of length self.size.
            This is the residual data rather than the original data.
        Returns:
            r_forward: scalar feature detection output

        """
        cov = self.cu.covariance
        # Power iterate - we could pass in here the input_data
        self.pi.iterate(cov=cov)
        # Project
        r_forward = project(self.pi.eigenvector.T, input_data)
        return r_forward

    def backward(self, r_backward):
        """Backward pass to generate predicted inputs.

        The predicted inputs are the original not residual inputs.

        Args:
            r_backward: scalar cause feedback.
        Returns:
            pred_inputs: numpy array of predicted inputs of size - size.

        """
        # Use item to convert r to scalar
        pred_inputs = project(r_backward.item(), self.pi.eigenvector)
        return pred_inputs

    def update_cov(self, input_data):
        """Update the covariance matrix.

        Use this to bed in the covariance.

        Args:
            input_data: 1D numpy array of length self.size.
            This is the original rather than residual data.
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
        self.cov_buffer = np.zeros(shape=(size, time_len))
        self.forward_buffer = np.zeros(shape=(size, time_len))
        self.backward_buffer = np.zeros(shape=(1, time_len))
        # Set up VPU with input as flattened buffer
        super(BufferVPU, self).__init__(size*time_len)

    def update_cov(self, input_data):
        """Update the covariance matrix."""
        # Add input to buffer
        self.cov_buffer = np.roll(
            self.cov_buffer, -1, axis=1
        )
        # Add frame to end of buffer
        self.cov_buffer[..., -1] = input_data.flatten()
        flat_time_data = self.cov_buffer.reshape(-1, 1)
        super(BufferVPU, self).update_cov(flat_time_data)

    def forward(self, input_data):
        """Forward pass - same interface as parent."""
        # Add input to buffer
        self.forward_buffer = np.roll(
            self.forward_buffer, -1, axis=1
        )
        # Add frame to end of buffer
        self.forward_buffer[..., -1] = input_data.flatten()
        # Run original forward method
        return super(BufferVPU, self).forward(
            self.forward_buffer.reshape(-1, 1)
        )

    def backward(self, r_backward):
        """Backward pass - same interface as parent."""
        # Add r_backward to rear buffer
        self.backward_buffer = np.roll(
            self.backward_buffer, -1, axis=1
        )
        # Add new r_backward to back buffer
        self.backward_buffer[..., -1] = r_backward
        # Take r_backward as the average of the stored values
        average_r_back = np.mean(self.backward_buffer, axis=1)
        return super(BufferVPU, self).backward(
            average_r_back
        )

    def iterate(self, input_data, r_backward):
        """Iterate - same interfaces as parent.

        input_data is of length size.
        """
        r_forward = self.forward(input_data)
        pred_inputs = self.backward(r_backward)
        # "Unbuffer" input_hat by averaging in time
        # This isn't right and skews our function
        unbuffered = np.mean(
            pred_inputs.reshape(-1, self.time_len),
            axis=1
        )
        return r_forward, unbuffered.reshape(-1, 1)

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size, self.time_len)


class VPUNonLin(VPU):
    """VPU with non-linearity on outputs."""

    def iterate(self, input_data, r_backward):
        """Iterate - same interfaces as parent.

        input_data is of length size.
        """
        r, pred_inputs = super(VPUNonLin, self).iterate(
            input_data,
            r_backward
        )
        # Apply non-linearity to output r
        r_non_lin = non_linearity(r)
        # Apply non-lienarity to output input prediction
        pred_inputs_non_lin = non_linearity(pred_inputs)
        return r_non_lin, pred_inputs_non_lin
