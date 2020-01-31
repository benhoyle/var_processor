"""Probablistic Binary Thresholding."""
import numpy as np
from packaging import version


def get_rand_ints(bit_size, input_size):
    """Output a set of random integers of size input_size.

    These will be of the indicated bit size.
    """
    # Set highest integer value
    high = 2**bit_size

    # Code to account for different numpy versions
    if version.parse(np.version.version) < version.parse("1.17.0"):
        rand_ints = np.random.randint(high, size=input_size)
    else:
        # Setup a random number generator
        rng = np.random.default_rng()
        rand_ints = rng.integers(high, size=input_size)
    return rand_ints


def pb_threshold(input_values):
    """Apply a probablistic binary threshold to the input_values."""
    input_size = input_values.shape
    data_type = input_values.dtype
    bit_size = data_type.itemsize*8
    rand_ints = get_rand_ints(bit_size, input_size)
    binary_values = np.where(input_values > rand_ints, 1, 0)
    return binary_values
