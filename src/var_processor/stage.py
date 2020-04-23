"""Stage - stateless non-time stage."""

import numpy as np
from src.var_processor.vpu import VPUBinaryZM


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


class Stage:
    """Object to represent a stage of processing."""

    def __init__(self, vec_len, stage_len):
        """Initialise stage.

        Arg:
            vec_len - length of each 1D vector processed by the VPUs.
            stage_len - integer indicating number of VPUs.
        """
        self.vec_len = vec_len
        self.stage_len = stage_len
        self.size = self.vec_len*self.stage_len
        self.vpus = [VPUBinaryZM(vec_len) for _ in range(0, stage_len)]
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(stage_len, 1))
        # Create a blank array for the predicted inputs
        self.pred_inputs = np.zeros(shape=(self.size, 1))
        # Helper data to keep indices
        self.ranges = [
            range(i*vec_len, (i+1)*vec_len)
            for i in range(0, stage_len)
        ]

    def forward(self, forward_data):
        """Forward pass through the stage (excludes cov update).

        Args:
            input_signal - 1D numpy array of length size.
        Returns:
            r - 1D numpy array of causes.

        """
        for i, vpu in enumerate(self.vpus):
            forward_segment = forward_data[self.ranges[i]]
            self.causes[i] = vpu.forward(forward_segment)
        return self.get_causes()

    def backward(self, r_backward):
        """Backward pass through the stage.

        Args:
            r_backward - 1D numpy array of causes of stage_len.
        Returns:
            pred_inputs - 1D numpy array of length size of predicted inputs.

        """
        for i, vpu in enumerate(self.vpus):
            feedback_segment = r_backward[i]
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

    def iterate(self, stage_in, residual_in, stage_feedback):
        """Pass data to the stage for processing.

        Arg:
            stage_in - 1D numpy array with data to process.
            residual_in - 1D numpy array with mix of predicted / original.
            stage_feedback - 1D numpy array with feedback data.

        Returns:
            r_out - 1D numpy array of causes.
            pred_input - 1D numpy array with predicted input.

        """
        self.update_cov(stage_in)
        causes = self.forward(residual_in)
        pred_inputs = self.backward(stage_feedback)
        return causes, pred_inputs

    def get_causes(self):
        """Return output of VPUs as array."""
        return self.causes.copy()

    def get_pred_inputs(self):
        """Return predicted inputs as array."""
        return self.pred_inputs.copy()

    def __repr__(self):
        """Print layer information."""
        string = f"There are {self.stage_len} units \n"
        string += f"with dimensionality {self.vec_len}x1"
        return string
