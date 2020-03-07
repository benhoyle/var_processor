"""Object to model a set of buffers arranged in series in time."""

import numpy as np
from src.var_processor.time_buffer import Buffer


def check_size(self, frame):
    """Check size of input matches array shape."""
    return frame.shape == self.forward_array.shape[0:2]


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


class TimeSeries:
    """Generate a cascade of time buffers."""

    def __init__(self, rows, cols, length, stages):
        """Initialise list for buffer series.

        Args:
            rows - integer indicating input array (frame) height in rows.
            cols - integer indicating input array (frame) width in cols.
            length - integer indicating number of time steps to buffer.
            stages - integer indicating the number of buffer stages.
        """
        self.time_series = list()
        self.shape = (rows, cols, length)
        # Generate series of buffers
        self.time_series = [Buffer(*self.shape) for i in range(0, stages)]
        return None

    def add(self, frame):
        """Add frame for processing.

        This runs for each time iteration and passes information
        between buffers.
        """
        # Define variable to hold data passed forward
        feedforward = frame
        # Iterate through pairs of buffers in series
        for b_ff, b_fb in zip(self.time_series[:-1], self.time_series[1:]):
            feedback = b_fb.fb_output
            # Get feedforward and feedback for buffer
            feedforward, feedback = b_ff.iterate(feedforward, feedback)
        # Then feedforward to last buffer
        self.time_series[-1].iterate(feedforward, feedback)
        return None

    @property
    def ff_output(self):
        """Provide feedforward output of all arrays."""
        return np.asarray([buffer.ff_output for buffer in self.time_series])

    @property
    def fb_output(self):
        """Provide feedback / reconstruction output."""
        return np.asarray([buffer.fb_output for buffer in self.time_series])

    @property
    def latest(self):
        """Get last entry of each buffer as array."""
        return np.asarray([buffer.latest for buffer in self.time_series])

    def __repr__(self):
        """Output a string representation."""
        string_list = [
            "FF", np.array_repr(self.ff_output), "---",
            "FB", np.array_repr(self.fb_output), "---",
            "Latest", np.array_repr(self.latest), "---"]
        return "\n\n".join(string_list)

    def reconstruct(self, time_periods):
        """Reconstruct an input.

        Arg:
            time_periods - integer indicating no. of outputs to sum
        """
        buffer_sum = np.asarray(
            [self.fb_output for i in range(0, time_periods)])
        return buffer_sum
