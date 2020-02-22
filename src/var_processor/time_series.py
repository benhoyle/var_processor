"""Object to model a set of buffers arranged in series in time."""

import numpy as np
from src.var_processor.time_buffer import Buffer


class TimeSeries:
    """Generate a cascade of time buffers."""

    def __init__(self, rows, cols, length):
        """Initialise list for buffer series.

        Args:
            rows - integer indicating input array (frame) height in rows.
            cols - integer indicating input array (frame) width in cols.
            length - integer indicating number of time steps to buffer.
        """
        self.time_series = list()
        self.shape = (rows, cols, length)
        self.time_series.append(Buffer(*self.shape))
        # Initialise input iteration count
        self.count = 0
        return None

    def add(self, frame):
        """Add frame for processing.

        This runs for each time iteration. Individual buffers are updated
        at a lower clock rate due to the flags.
        """
        length = self.shape[2]
        # Iterate through buffers in series
        for i, buffer in enumerate(self.time_series):
            if (self.count % length**i) == 0:
                buffer.add(frame)
                # Get feedback from next buffer if not last buffer
                if i < (len(self.time_series)-1):
                    # Add here that self.count % length**(i+1)?
                    feedback_in = self.time_series[i+1].output
                    buffer.feedback(feedback_in)
                frame = buffer.output
        # Increment count
        self.count += 1
        # If we need to add a new buffer
        if self.count == length**len(self.time_series):
            self.time_series.append(
                Buffer(*self.shape)
            )
            # Reset count
            self.count = 0
        return None

    def output(self):
        """Get data as numpy array."""
        return np.asarray([buffer.output for buffer in self.time_series])
