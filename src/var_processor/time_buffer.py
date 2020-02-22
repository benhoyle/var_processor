"""Time Buffer."""

import numpy as np


class Buffer:
    """Object for a time buffer."""

    def __init__(self, rows, cols, length):
        """Initialise object.

        Assumes 8-bit values (for now).
        """
        # Set up an array to store a rolling window of inputs
        self.array = np.zeros(shape=(rows, cols, length), dtype=np.uint8)
        # Initialise feedback mean
        self.feedback_theta = np.zeros(shape=(rows, cols), dtype=np.uint8)
        # Initialise variable that provides a rolling count
        self.rolling_count = 0
        # Initialise flag to indicate when a full buffer cycle is reached
        self.flag = False
        self.length = length

    def add(self, frame):
        """Add a frame to the buffer."""
        # Roll array
        self.array = np.roll(self.array, -1, axis=2)
        # Add frame to end of buffer
        self.array[..., -1] = frame
        self.rolling_count += 1
        if self.rolling_count == self.length:
            # Toggle flag & reset count
            self.flag = True
            self.rolling_count = 0
        else:
            self.flag = False
        return None

    @property
    def average(self):
        """Average frames in the buffer."""
        return np.mean(self.array, axis=2)

    @property
    def latest(self):
        """Return latest entry in buffer."""
        return self.array[..., -1]

    def feedback(self, fb_theta):
        """Update the feedback theta or mean."""
        # Check shape
        if fb_theta.shape == self.feedback_theta.shape:
            self.feedback_theta = fb_theta
        return None

    @property
    def output(self):
        """Output the adjusted theta."""
        # Calculate difference
        adjusted = (self.average - self.feedback_theta).astype(np.int8)
        # Limit to positive values in fixed range
        clipped = np.clip(adjusted, 0, np.max(adjusted)).astype(np.uint8)
        return clipped
