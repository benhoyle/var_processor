"""Variance Processing Unit."""

import numpy as np
from src.var_processor.abstract_classes import AbstractSubUnit, TransformMixin
from src.var_processor.covariance import CovarianceUnit
from src.var_processor.power_iterator import PowerIterator
from src.var_processor.pb_threshold import ternary_pbt


def project(vec_1, vec_2):
    """Project input using eigenvector.

    Args:
        vec1: 1D numpy array.
        vec2: 1D numpy array.
    """
    return np.dot(vec_1, vec_2)


class VPU(AbstractSubUnit, TransformMixin):
    """Variance processing unit."""

    def __init__(self, vec_len):
        """Initialise.

        Args:
            vec_len: integer setting the 1D size of an input;
        """
        super(VPU, self).__init__(vec_len)
        self.cu = CovarianceUnit(vec_len)
        self.pi = PowerIterator(vec_len)

    def backward_pre_processing(self, backward_data):
        """Convert r to integer."""
        # Convert numpy array to integer
        if type(backward_data) is np.ndarray:
            backward_data = backward_data.item()
        else:
            backward_data = int(backward_data)
        return backward_data

    def forward(self, forward_data):
        """Forward pass to generate cause - r.

        Args:
            forward_data: 1D numpy array of length self.size.
            This is the residual data rather than the original data.
        Returns:
            r_forward: scalar feature detection output

        """
        # Perform optional pre-processing
        processed_data = self.forward_pre_processing(forward_data)
        # Project
        r_forward = project(self.eigenvector.T, processed_data)
        # Perform optional post-processing
        processed_output = self.forward_post_processing(r_forward)
        return processed_output

    def backward(self, backward_data):
        """Backward pass to generate predicted inputs.

        The predicted inputs are the original not residual inputs.

        Args:
            backward_data: scalar cause or r feedback.
        Returns:
            pred_inputs: numpy array of predicted inputs of size - size.

        """
        # Perform optional pre-processing
        processed_r_back = self.backward_pre_processing(backward_data)
        # Use item to convert r to scalar
        pred_inputs = project(processed_r_back, self.eigenvector)
        # Perform optional post-processing
        processed_output = self.backward_post_processing(pred_inputs)
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
            cov = self.covariance
            # Power iterate
            self.pi.load_covariance(cov)
            self.pi.iterate()

    def reset(self):
        """Reset and clear."""
        self.__init__(self.vec_len)

    @property
    def eigenvector(self):
        """Return eigenvector."""
        return self.pi.eigenvector

    @property
    def eigenvalue(self):
        """Return eigenvalue."""
        return self.pi.eigenvalue

    @property
    def covariance(self):
        """Return covariance."""
        return self.cu.covariance


class BinaryVPU(VPU):
    """Let's update our functions modularly."""

    def forward_pre_processing(self, forward_data):
        """Process data to apply to forward input data."""
        if forward_data.any():
            forward_data = forward_data*127//np.linalg.norm(forward_data)
        return forward_data

    def forward_post_processing(self, forward_output):
        """Scale r to -127 to 127 and PBT."""
        # Threshold r
        forward_output = forward_output//127
        pbt_output = ternary_pbt(forward_output, 127)
        return pbt_output

    def backward_post_processing(self, backward_output):
        """Apply PBT to get outputs in range -1, 0, 1."""
        # Get non-zero eigenvector values
        non_zeros = np.nonzero(backward_output.ravel())[0].shape[0]
        if non_zeros > 0:
            # Scale by sqrt of number of non-zeros
            backward_output = backward_output*np.sqrt(non_zeros)
        binary_values = ternary_pbt(backward_output, 127)
        return binary_values
