"""Time Stage - a group of VPUs that represent data for a given time."""

import numpy as np
from src.var_processor.vpu import VPUNonLin


class TimeStage:
    """Object to represent a time stage of processing."""

    def __init__(self, vec_len, stage_len):
        """Initialise stage.

        Arg:
            vec_len - length of each 1D vector processed by the VPUs.
            stage_len - integer indicating number of VPUs.
        """
        self.vec_len = vec_len
        self.stage_len = stage_len
        self.size = self.vec_len*self.stage_len
        self.vpus = [VPUNonLin(vec_len) for _ in range(0, stage_len)]
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(stage_len, 1))
        # Create a blank array for the predicted inputs
        self.pred_inputs = np.zeros(shape=(self.size, 1))

    def iterate(self, stage_in, stage_feedback=None):
        """Pass data to the stage for processing.

        Arg:
            stage_in - 1D numpy array with data to process.
            stage_feedback - 1D numpy array with feedback data.

            If stage_feedback is None there is no feedback, e.g.
            for a last stage in a series.

        Returns:
            r_out - 1D numpy array of causes.
            pred_input - 1D numpy array with predicted input.

        """
        # Create blank array to hold / pad data
        input_array = np.zeros(shape=(self.size, 1))
        # Check data is of right size
        if stage_in.shape[0] > self.size:
            # Crop input
            input_array = stage_in[:self.size]
        elif stage_in.shape[0] < self.size:
            input_array[:stage_in.shape[0]] = stage_in
        else:
            input_array = stage_in
        # Iterate through VPUs, passing data in
        for i, vpu in enumerate(self.vpus):
            start = i*self.vec_len
            end = (i+1)*self.vec_len
            input_segment = input_array[start:end]
            if stage_feedback is not None:
                feedback_segment = stage_feedback[i]
            else:
                feedback_segment = None
            cause, pred_input = vpu.iterate(
                input_segment,
                feedback_segment
            )
            self.causes[i] = cause
            self.pred_inputs[start:end] = pred_input
        return self.causes, self.pred_inputs

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
