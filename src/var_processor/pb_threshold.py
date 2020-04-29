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


def non_linearity(input_values):
    """Apply a non-linearity to the input.

    Args:
        input_values - numpy array.
    Returns:
        array of 8-bit binary values.

    """
    # Initially use tanh as per Rao paper - it's also faster than max/min norm
    # non_lin_output = np.tanh(input_values)
    # Or just clip at -1, 1? Do we need to do this at all?
    non_lin_output = input_values
    # Add bias here - moves -ve values into +ve space
    biased = non_lin_output + 1
    # Now we need to convert to int to threshold?
    # Why don't we define a float thresholder
    rand_vals = np.random.uniform(size=biased.shape)
    binary_values = np.where(input_values > rand_vals, 1, 0)
    return binary_values.astype(np.uint8)


def ternary_pbt(data_in, max_abs_value):
    """Apply probabilistic thresholding to data_array.

    Args:
        data_in: nD numpy array.
        max_abs_value: maximum absolute value in array.

    """
    # PBT
    # Get random integers - from 0 (inclusive) to max_abs_value (exclusive)
    rand_ints = np.random.randint(
        low=0,
        high=max_abs_value,
        size=data_in.shape
    )
    signs = np.sign(data_in)
    pbt_output = np.where(np.abs(data_in) > rand_ints, 1, 0)
    # Add to next stage (with signs returned)
    resigned = pbt_output*signs
    return resigned
