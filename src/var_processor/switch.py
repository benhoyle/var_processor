"""Switch - Thalamus Model for Setup."""

import numpy as np


def norm(residual):
    """Compute a norm of a residual array.

    Arg:
        residual - numpy array.
    """
    absolute = np.absolute(residual)
    scaled_total = absolute.sum() / absolute.shape[0]
    return scaled_total


class Switch:
    """Object to manage stacks.

    Models some thalamus function.
    """

    def __init__(self, sensor, stacks):
        """Initialise.

        Args:
            sesnor - a Sensor object to provide input data.
            stacks - a list of Stack objects.
        """
        self.sensor = sensor
        self.stacks = stacks
        # Initialise list for residuals
        self.residuals = [None for s in stacks]
        # Initialise norm list
        self.norms = [None for s in stacks]

    def iterate(self):
        """Perform an iteration.

        Returns:
            residuals - list of residuals.
            norms - list of norms for each residual.

        """
        input_data = self.sensor.get_frame()
        for i, stack in enumerate(self.stacks):
            # Iterate stack
            _, pred_out = stack.iterate(input_data, None)
            # Compute residual
            self.residuals[i] = input_data - pred_out
            # Compute norm
            self.norms[i] = norm(self.residuals[i])
            # Set input data for next stack as residual
            input_data = self.residuals[i]
        return self.residuals, self.norms
