"""Stage - stateless non-time stage."""

import numpy as np
from src.var_processor.vpu import VPUBinary

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
        self.vpus = [VPUBinary(vec_len) for _ in range(0, stage_len)]
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(stage_len, 1))
        # Create a blank array for the predicted inputs
        self.pred_inputs = np.zeros(shape=(self.size, 1))

    def iterate(self, stage_in, stage_feedback):
        """Pass data to the stage for processing.

        Arg:
            stage_in - 1D numpy array with data to process.
            stage_feedback - 1D numpy array with feedback data.

        Returns:
            r_out - 1D numpy array of causes.
            pred_input - 1D numpy array with predicted input.

        """
        input_array = pad_array(stage_in, self.size)
        # Iterate through VPUs, passing data in
        for i, vpu in enumerate(self.vpus):
            start = i*self.vec_len
            end = (i+1)*self.vec_len
            input_segment = input_array[start:end]
            feedback_segment = stage_feedback[i]
            vpu.update_cov(input_segment)
            cause, pred_input = vpu.iterate(
                input_segment,
                feedback_segment
            )
            self.causes[i] = cause
            self.pred_inputs[start:end] = pred_input
        return self.get_causes(), self.get_pred_inputs()

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
