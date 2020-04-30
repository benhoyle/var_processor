"""Variance Processing Unit."""

import numpy as np
from src.var_processor.covariance8bit import CovarianceUnit
from src.var_processor.power_iterator8bit import PowerIterator
from src.var_processor.pb_threshold import ternary_pbt


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

    def update_cov(self, input_data, power_iterate=False):
        """Update the covariance matrix.

        Use this to bed in the covariance.

        Args:
            input_data: 1D numpy array of length self.size.
            This is the original rather than residual data.
        """
        self.cu.update_cov(input_data)
        if power_iterate:
            cov = self.cu.covariance
            # Power iterate
            self.pi.load_covariance(cov)
            self.pi.iterate()

    def reset(self):
        """Reset and clear."""
        self.__init__(self.size)
