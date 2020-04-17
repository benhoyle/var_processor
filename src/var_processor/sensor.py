"""Sensor - wraps an input source.

Can be thought of as an input pre-processor.

"""

import math
import numpy as np


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
    """
    Object to process an incoming sensor signal.

    Takes the length of the vector / receptive field and resizes
    the input signal to be a suitable length and converts to binary
    by applying a bias and probabilistic thresholding.
    """

    def __init__(self, sensor_source, vec_len, start=True):
        """Initialise sensor.

        Arg:
            sensor_source - SensorSource object that outputs a
                vector of sensor readings when iterated.
            vec_len - length of vector for VPU.
        """
        self.source = sensor_source
        self.vec_len = vec_len
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
            self.power_len = self.vec_len**int(num_stages)

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
        # PBThreshold - do we need to add bias? Added below
        rand_ints = 128+np.random.uniform(size=output.shape)*127
        binary_values = np.where(output > rand_ints, 1, 0)
        return binary_values

    def get_data_length(self):
        """Return vector length of initial data."""
        return self.power_len

    def stop(self):
        """Steop sensor thread."""
        if self.source.started:
            self.source.stop()
