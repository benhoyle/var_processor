"""8Bit Covariance Unit v2."""

import numpy as np
from src.var_processor.pb_threshold import ternary_pbt
from src.var_processor.abstract_classes import AbstractBase


class CovarianceUnit(AbstractBase):
    """Variation where the mean is assumed to be 0."""

    def __init__(self, vec_len, stages=8):
        """Initialise.

        Args:
            vec_len: integer setting the 1D size of an input.
            stages: integer setting the number of stages.
        """
        super(CovarianceUnit, self).__init__(vec_len)
        # Set max value for signed int
        self.max_value = 127
        self.stages = stages
        # Initialise Square Sums
        self.square_sum = np.zeros(
            shape=(vec_len, vec_len, self.stages), dtype=np.int8
        )
        # Initialise Store for last full values
        self.complete = np.zeros(
            shape=(vec_len, vec_len, self.stages), dtype=np.int8
        )
        # Define counter for each stage
        self.stage_counter = np.zeros(self.stages, dtype=np.uint8)
        # Define index for current cov
        self.cov_index = 0

    def update_cov(self, input_data):
        """Add a data array to the covariance data.

        This will involve a recursive check.

        Args:
            input_data is a 1D numpy array of length 'size'.
        """
        assert max(np.abs(input_data)) == 0 or 1
        # Cast data_array to 8 bit - also check binary here?
        input_data = input_data.astype(np.int8)
        # Increment current stage counter
        self.stage_counter[0] += 1
        # Add square of input array
        self.square_sum[:, :, 0] += np.dot(input_data, input_data.T)
        self.__recursive_update(0)

    def __recursive_update(self, i):
        """Update with recursive method.

        Args:
            i - stage to update - integer.
        """
        # Check i is within range
        if i > (self.stages - 1):
            return
        if i < self.stages:
            # If i is within range check counter
            if self.stage_counter[i] >= self.max_value:
                # Add to completed estimate
                self.complete[:, :, i] = self.square_sum[:, :, i]
                # Reset the previous counter and stage
                self.stage_counter[i] = 0
                self.square_sum[:, :, i] = 0
                # Set cov index as highest available
                if self.cov_index < i:
                    self.cov_index = i
                # If higher stages PBT and add to higher stages
                if i < (self.stages-1):
                    # Apply ternary PBT to square sum
                    thresholded = ternary_pbt(
                        self.complete[:, :, i], self.max_value
                    )
                    # Add to next square sum
                    self.square_sum[:, :, i+1] += thresholded
                    # Increment next stage counter
                    self.stage_counter[i+1] += 1
                    self.__recursive_update(i+1)

    @property
    def covariance(self):
        """Compute covariance when requested."""
        # This may need to change to return changing values
        # Return highest non_zero self.complete[:, :, i]
        return self.complete[:, :, self.cov_index]

    def __repr__(self):
        """Return string representation of covariance unit state."""
        string = super(CovarianceUnit, self).__repr__()
        string += (
            f"""There are {self.stages} stages to process.\n"""
            f"""Data is assumed to """
            f"""have a maximum absolute value of {self.max_value}.\n"""
            f"""-------\nCounter: {self.stage_counter}\n"""
            f"""Running sum of squares:\n"""
        )
        for i in np.nonzero(self.stage_counter)[0]:
            string += f"""{self.square_sum[:, :, i]}\n"""
        string += """Complete covariance estimates:\n"""
        complete_range = max(np.nonzero(self.stage_counter)[0])
        for i in range(0, complete_range):
            string += f"""{self.complete[:, :, i]}\n"""
        string += (
            f"""\n---------\nCurrent covariance estimate """
            f"""(index: {self.cov_index}):\n{self.covariance}\n"""
        )
        return string
