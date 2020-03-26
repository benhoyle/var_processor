"""Sensor - a VPU Aggregator."""

import math
import numpy as np
from src.var_processor.time_stage import TimeStage


def resize(array, elem_num):
    """Linearly scale array.

    Arg:
        elem_num - integer number of new elements in array.
    """
    old_length = array.shape[0]
    x = np.linspace(0, old_length-1, elem_num)
    xp = np.linspace(0, old_length-1, old_length)
    return np.interp(x, xp, array.flatten()).reshape(-1, 1)


class Sensor:
    """Object to process a 1D sensor signal.

    For this to work well the data output by sensor_source should be a power
    of vec_len.
    """

    def __init__(self, sensor_source, vec_len, time_len, start=True):
        """Initialise sensor.

        Arg:
            sensor_source - SensorSource object that outputs a
            vector of sensor readings when iterated.
            vec_len - length of vector for VPU.
            time_len - length of time buffering.
        """
        self.source = sensor_source
        self.vec_len = vec_len
        self.time_len = time_len
        # Variable to store time stages
        self.stages = list()
        # Variable to store nearest power length
        self.power_len = None
        # Variable to store original sensor length
        self.sensor_len = None
        # Start sensor by default
        if start:
            self.start()

    def start(self):
        """Start sensor."""
        self.source.start()
        if not self.power_len:
            _, initial_frame = self.source.read()
            flattened = initial_frame.reshape(-1, 1)
            self.sensor_len = flattened.shape[0]
            num_stages = math.log(self.sensor_len, self.vec_len)
            self.num_stages = int(num_stages)
            self.power_len = self.vec_len**self.num_stages
        if not self.stages:
            # Build the time stages
            self.build_stages()

    def get_frame(self):
        """Get a 1D frame of data from the sensor."""
        # If the sensor is not started, start
        if not self.source.started:
            self.start()
        # Get frame and flatten to 1D array
        _, initial_frame = self.source.read()
        flattened = initial_frame.reshape(-1, 1)
        # Resize to nearest power of vec_len
        output = resize(flattened, self.power_len)
        return output

    def generate_stage(self, stage_len):
        """Generate a stage.

        Arg:
            stage_len - integer number of stages.
        """
        return TimeStage(self.vec_len, stage_len)

    def build_stages(self):
        """Build a set of stages."""
        self.stages = [
            self.generate_stage(
                int(self.power_len / self.vec_len**(i+1))
            )
            for i in range(0, self.num_stages)
        ]

    def iterate(self):
        """High level processing loop."""
        frame = self.get_frame()
        # Set feedforward as input data
        feedforward = frame
        # Iterate through pairs of timestages in series
        for ts_ff, ts_fb in zip(self.stages[:-1], self.stages[1:]):
            feedback = ts_fb.get_pred_inputs()
            # Get feedforward and feedback for buffer
            feedforward, feedback = ts_ff.iterate(feedforward, feedback)
        # Then feedforward to last stage with no feedback (for now)
        self.stages[-1].iterate(feedforward, None)
        return frame

    def get_causes(self):
        """Return causes as a list of arrays."""
        return [
            stage.get_causes() for stage in self.stages
        ]

    def get_pred_inputs(self):
        """Return predicted inputs as a list of arrays."""
        return [
            stage.get_pred_inputs() for stage in self.stages
        ]

    def get_lengths(self):
        """Return the vector lengths of the causes and predicted inputs."""
        causes = self.get_causes()
        pred_inputs = self.get_pred_inputs()
        cause_lengths = [cause.shape[0] for cause in causes]
        pred_lengths = [pred.shape[0] for pred in pred_inputs]
        return cause_lengths, pred_lengths

    def get_data_length(self):
        """Return vector length of initial data."""
        return self.power_len

    def stop(self):
        """Steop sensor thread."""
        if self.source.started:
            self.source.stop()
