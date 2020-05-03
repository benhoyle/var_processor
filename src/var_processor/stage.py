"""Stage - stateless non-time stage."""

import numpy as np
from src.var_processor.vpu import BinaryVPU
from src.var_processor.abstract_classes import (
    AbstractSignalProcessor, TransformMixin
)


def pad_array(array_in, size):
    """Format array_in to make equal to size."""
    # Create blank array to hold / pad data
    input_array = np.zeros(shape=(size, 1))
    # Check data is of right size
    if array_in.shape[0] > size:
        # Crop input
        input_array = array_in[:size]
    elif array_in.shape[0] < size:
        input_array[:array_in.shape[0]] = array_in
    else:
        input_array = array_in
    return input_array


class Stage(AbstractSignalProcessor, TransformMixin):
    """Object to represent a stage of processing."""

    def __init__(self, vec_len, input_len):
        """Initialise stage.

        Arg:
            vec_len - length of each 1D vector processed by the VPUs.
            input_len - length of input to stage.
        """
        super(Stage, self).__init__(vec_len, input_len)
        self.vpu_len = self.input_len//self.vec_len
        self.vpus = [BinaryVPU(vec_len) for _ in range(0, self.vpu_len)]
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(self.vpu_len, 1), dtype=np.int8)
        # Create a blank array for the predicted inputs
        self.pred_inputs = np.zeros(shape=(self.input_len, 1), dtype=np.int8)
        # Helper data to keep indices
        self.ranges = [
            range(i*vec_len, (i+1)*vec_len)
            for i in range(0, self.vpu_len)
        ]

    def forward(self, forward_data):
        """Forward pass through the stage (excludes cov update).

        Args:
            forward_data - 1D numpy array of length size.
        Returns:
            r - 1D numpy array of causes.

        """
        for i, vpu in enumerate(self.vpus):
            forward_segment = forward_data[self.ranges[i]]
            self.causes[i] = vpu.forward(forward_segment)
        return self.get_causes()

    def backward(self, backward_data):
        """Backward pass through the stage.

        Args:
            backward_data - 1D numpy array of causes of stage_len.
        Returns:
            pred_inputs - 1D numpy array of length size of predicted inputs.

        """
        for i, vpu in enumerate(self.vpus):
            feedback_segment = backward_data[i]
            self.pred_inputs[self.ranges[i]] = vpu.backward(feedback_segment)
        return self.get_pred_inputs()

    def update_cov(self, input_data, power_iterate=True):
        """Update covariance data.

        Args:
            input_data: 1D numpy array of length size.
        """
        for i, vpu in enumerate(self.vpus):
            input_segment = input_data[self.ranges[i]]
            vpu.update_cov(input_segment, power_iterate=power_iterate)

    def get_causes(self):
        """Return output of VPUs as array."""
        return self.causes.copy()

    def get_pred_inputs(self):
        """Return predicted inputs as array."""
        return self.pred_inputs.copy()

    def get_eigenvectors(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [vpu.eigenvector for vpu in self.vpus]
        return evs

    def get_eigenvalues(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [vpu.eigenvalue for vpu in self.vpus]
        return evs

    def get_covariances(self):
        """Return covariance matrices."""
        covs = [vpu.covariance for vpu in self.vpus]
        return covs
