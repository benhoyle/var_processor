"""Variance Processing Unit."""

import random
import numpy as np
from src.var_processor.covariance import CovarianceUnit
from src.var_processor.power_iterator import PowerIterator


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

    def forward_processing(self, forward_data):
        """Process data to apply to forward input data."""
        return forward_data

    def pred_input_processing(self, pred_inputs):
        """Process data to apply to output predicted inputs."""
        return pred_inputs

    def r_forward_processing(self, r_forward):
        """Process data to apply to output r_forward value."""
        return r_forward

    def r_backward_processing(self, r_backward):
        """Process data to apply to output r_forward value."""
        return r_backward

    def iterate(self, forward_data, r_backward):
        """Iterate through one discrete timestep.

        Args:
            forward_data: data for feedforward transformation
                1D numpy array of length self.size.
            r_backward: scalar value indicating a prediction of r.

        Returns:
            pred_inputs: 1D array containing the predicted input
            r: scalar feature detection output

        """
        r_forward = self.forward(forward_data)
        pred_inputs = self.backward(r_backward)
        return r_forward, pred_inputs

    def forward(self, forward_data):
        """Forward pass to generate cause - r.

        Args:
            forward_data: 1D numpy array of length self.size.
            This is the residual data rather than the original data.
        Returns:
            r_forward: scalar feature detection output

        """
        # Perform optional pre-processing
        processed_data = self.forward_processing(forward_data)
        cov = self.cu.covariance
        # Power iterate - THIS COULD GO IN COV_UPDATE?
        self.pi.iterate(cov=cov)
        # Project
        r_forward = project(self.pi.eigenvector.T, processed_data)
        # Perform optional post-processing
        processed_output = self.r_forward_processing(r_forward)
        return processed_output

    def backward(self, r_backward):
        """Backward pass to generate predicted inputs.

        The predicted inputs are the original not residual inputs.

        Args:
            r_backward: scalar cause feedback.
        Returns:
            pred_inputs: numpy array of predicted inputs of size - size.

        """
        # Perform optional pre-processing
        processed_r_back = self.r_backward_processing(r_backward)
        # Use item to convert r to scalar
        pred_inputs = project(processed_r_back.item(), self.pi.eigenvector)
        # Perform optional post-processing
        processed_output = self.pred_input_processing(pred_inputs)
        return processed_output

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


class VPUBinary(VPU):
    """VPU with unbiasing and non-linearity on outputs."""

    def __init__(self, size):
        """Initialise."""
        super(VPUBinary, self).__init__(size)
        self.r_sum = 0
        self.r_count = 0

    @property
    def r_mean(self):
        """Get mean r."""
        return self.r_sum / self.r_count

    def forward_processing(self, forward_data):
        """Process data to apply to forward input data."""
        return forward_data - self.cu.mean

    def pred_input_processing(self, pred_inputs):
        """Process data to apply to output predicted inputs."""
        # Add bias
        processed_output = pred_inputs + self.cu.mean
        # Convert to binary
        rand_vals = np.random.uniform(size=processed_output.shape)
        binary_values = np.where(processed_output > rand_vals, 1, 0)
        return binary_values.astype(np.uint8)

    def r_forward_processing(self, r_forward):
        """Process data to apply to output r_forward value."""
        self.r_sum += r_forward
        self.r_count += 1
        # Add bias
        r_f_out = r_forward + self.r_mean
        # Convert to binary
        binary_value = r_f_out > random.random()
        return binary_value

    def r_backward_processing(self, r_backward):
        """Process data to apply to output r_forward value."""
        # Remove bias
        r_b_out = r_backward - self.r_mean
        return r_b_out
