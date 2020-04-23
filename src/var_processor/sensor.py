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


def signal_adjust(signal, mean, signal_max=255):
    """Remove mean and turn to ternary signal in range {-1, 0, 1}.

    Args:
        signal - numpy array representing an input signal.
        mean - numpy array representing the mean of the input signal.
        signal_max - value indicating a maximum value for the input
            signal - defaults to 255 (8-bit).
    """
    # Remove mean
    zero_mean = signal - mean
    # Get signs
    signs = np.sign(zero_mean)
    # PBThreshold - outputs values {-1, 0, 1}
    # We want symmetric mean for reconstruction
    signal_max = np.minimum(mean, (signal_max - mean))
    rand_vals = np.random.uniform(size=zero_mean.shape)*mean
    binary_values = np.where(np.abs(zero_mean) > rand_vals, 1, 0)
    # Re-add in signs
    signed = binary_values*signs
    return signed


class Sensor:
    """
    Object to process an incoming sensor signal.

    Takes the length of the vector / receptive field and resizes
    the input signal to be a suitable length and converts to binary
    by applying a bias and probabilistic thresholding.
    """

    def __init__(self, sensor_source, vec_len, start=True, m_batch=1000):
        """Initialise sensor.

        Arg:
            sensor_source - SensorSource object that outputs a
                vector of sensor readings when iterated.
            vec_len - length of vector for VPU.
            start - boolean to indicate whether to start on init.
            m_batch - number of reading to batch for mean estimate
        """
        self.source = sensor_source
        self.vec_len = vec_len
        # Variable to store time stages
        self.stages = list()
        # Variable to store nearest power length
        self.power_len = None
        # Variable to store original sensor length
        self.sensor_len = None
        # Add running signal mean measurement
        self.signal_mean = None
        self.sum = None
        # Add count for mean & batch size
        self.count = 0
        self.m_batch = m_batch
        # Start sensor by default
        if start:
            self.start()

    def start(self):
        """Start sensor."""
        self.source.start()
        if not self.power_len:
            _, initial_frame = self.source.read()
            # We might want to flatten video in local groups
            flattened = initial_frame.reshape(-1, 1)
            self.sensor_len = flattened.shape[0]
            num_stages = math.log(self.sensor_len, self.vec_len)
            self.power_len = self.vec_len**int(num_stages)
            self.sum = np.zeros(shape=(self.power_len, 1))

    def update_mean(self, signal):
        """Update running mean estimate.

        Args:
            signal - reszied & flattened 1D numpy array of data.
        """
        self.sum += signal
        self.count += 1
        if self.count >= self.m_batch:
            new_mean = self.sum/self.count
            # If mean is empty set, else take an average
            if self.signal_mean is None:
                self.signal_mean = new_mean
            else:
                self.signal_mean = (self.signal_mean + new_mean)/2
            # Reset sum
            self.sum.fill(0)
            self.count = 0
        # Maybe too slow to compute this every frame?
        return self.mean

    @property
    def mean(self):
        """Get mean."""
        if self.signal_mean is None:
            if self.count:
                output_mean = self.sum/self.count
            else:
                output_mean = None
        else:
            output_mean = self.signal_mean
        return output_mean

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
        mean = self.update_mean(output)
        signed = signal_adjust(output, mean)
        return signed

    def get_data_length(self):
        """Return vector length of initial data."""
        return self.power_len

    def stop(self):
        """Steop sensor thread."""
        if self.source.started:
            self.source.stop()
