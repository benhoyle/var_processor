"""Time Buffer."""

import numpy as np
from src.var_processor.pb_threshold import pb_threshold


def check_size(self, frame):
    """Check size of input matches array shape."""
    return bool(frame.shape == self.forward_array.shape[0:2])


def add_to_array(array, frame):
    """Add a frame to a rolling array."""
    array = np.roll(array, -1, axis=2)
    # Add frame to end of buffer
    array[..., -1] = frame
    return array


def process_array(array):
    """Process an array."""
    # Start with just averaging - could be convolution
    return np.mean(array, axis=2)


class Buffer:
    """Object for a time buffer."""

    def __init__(self, rows, cols, length):
        """Initialise object.

        Assumes 8-bit values (for now).
        """
        # Set up an array to store a rolling window of inputs
        self.forward_array = np.zeros(
            shape=(rows, cols, length), dtype=np.uint8)
        # Initialise feedback mean
        self.backward_array = np.zeros(
            shape=(rows, cols, length), dtype=np.uint8)

    def feedforward(self, frame):
        """Add a frame to the buffer in FF mode."""
        self.forward_array = add_to_array(self.forward_array, frame)
        return None

    def feedback(self, frame):
        """Add a frame to the buffer in FB mode."""
        self.backward_array = add_to_array(self.backward_array, frame)
        return None

    @property
    def latest(self):
        """Return latest entry in buffer."""
        return self.forward_array[..., -1]

    @property
    def ff_output(self):
        """Provide feedforward output."""
        average = process_array(self.forward_array)
        # Convert to 8-bit for thresholding
        converted = (average*255).astype(np.uint8)
        # Add non-linearity
        binary = pb_threshold(converted)
        return binary

    @property
    def fb_output(self):
        """Provide feedback / reconstruction output."""
        average = process_array(self.backward_array)
        # Subtract input - this is just the latest frame
        difference = average - self.latest  # This is float64
        # Clip to 0 to 1
        clipped = np.clip(difference, 0, 1)
        # Convert to 8-bit integer
        converted = (clipped*255).astype(np.uint8)
        # Add non-linearity
        binary = pb_threshold(converted)
        return binary

    def iterate(self, input_frame, input_feedback):
        """Perform both a forward and backward pass."""
        self.feedforward(input_frame)
        self.feedback(input_feedback)
        return self.ff_output, self.fb_output
