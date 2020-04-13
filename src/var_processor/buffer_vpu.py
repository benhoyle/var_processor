"""Time buffered VPU."""

import numpy as np
from src.var_processor.vpu import VPU


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

    def forward(self, forward_data):
        """Forward pass - same interface as parent."""
        # Add input to buffer
        self.forward_buffer = np.roll(
            self.forward_buffer, -1, axis=1
        )
        # Add frame to end of buffer
        self.forward_buffer[..., -1] = forward_data.flatten()
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

    def iterate(self, forward_data, r_backward):
        """Iterate - same interfaces as parent.

        forward_data is of length size.
        """
        r_forward = self.forward(forward_data)
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
