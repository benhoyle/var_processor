"""Time Stage - a group of VPUs that represent data for a given time."""

import numpy as np
from src.var_processor.vpu import VPU


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
        self.vpus = [VPU(vec_len) for _ in range(0, stage_len)]
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(stage_len, 1))
        # Create a blank array for the residuals
        self.residuals = np.zeros(shape=(self.size, 1))

    def forward(self, stage_in):
        """Pass data to the stage for processing.

        Arg:
            stage_in - 1D numpy array with data to process.
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
            cause, residual = vpu.iterate(input_array[start:end])
            self.causes[i] = cause
            self.residuals[start:end] = residual

    def get_causes(self):
        """Return output of VPUs as array."""
        return self.causes.copy()

    def get_residuals(self):
        """Return residual output as array."""
        return self.residuals.copy()

    def __repr__(self):
        """Print layer information."""
        string = f"There are {self.stage_len} units \n"
        string += f"with dimensionality {self.vec_len}x1"
        return string
