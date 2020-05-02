"""8Bit Covariance Unit v2."""

import numpy as np
from src.var_processor.pb_threshold import ternary_pbt


class CovarianceUnit:
    """Variation where the mean is assumed to be 0."""

    def __init__(self, size, stages=8):
        """Initialise.

        Args:
            size: integer setting the 1D size of an input.
            stages: integer setting the number of stages.
        """
        self.size = size
        # Set max value for signed int
        self.max_value = 127
        self.stages = stages
        # Initialise Square Sums
        self.square_sum = np.zeros(
            shape=(size, size, self.stages), dtype=np.int8
        )
        # Initialise Store for last full values
        self.complete = np.zeros(
            shape=(size, size, self.stages), dtype=np.int8
        )
        # Define counter for each stage
        self.stage_counter = np.zeros(self.stages, dtype=np.uint8)
        # Define index for current cov
        self.cov_index = 0

    def update_cov(self, data_array):
        """Add a data array to the covariance data.

        This will involve a recursive check.

        Args:
            data_array is a 1D numpy array of length 'size'.
        """
        assert max(np.abs(data_array)) == 0 or 1
        # Cast data_array to 8 bit - also check binary here?
        data_array = data_array.astype(np.int8)
        # Increment current stage counter
        self.stage_counter[0] += 1
        # Add square of input array
        self.square_sum[:, :, 0] += np.dot(data_array, data_array.T)
        self.recursive_update(0)

    def recursive_update(self, i):
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
                    self.recursive_update(i+1)

    @property
    def covariance(self):
        """Compute covariance when requested."""
        # This may need to change to return changing values
        # Return highest non_zero self.complete[:, :, i]
        return self.complete[:, :, self.cov_index]

    def __repr__(self):
        """String representation of covariance unit state."""
        string = (
            f"""There are {self.stages} stages to process """
            f"""1D arrays of length {self.size}.\nData is assumed to """
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
