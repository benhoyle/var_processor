"""Power Iterator - 8bit version."""

import numpy as np


def normalise(array):
    """Scale array by L2 norm.

    Args:
        array - int32 numpy 1D array holding 8-bit values.
    """
    sq_root_sum = np.sqrt((array**2).sum())
    # Watch the below - we need to bring the 127
    # onto the top to keep everything in integer space
    scaled_array = (array*127//sq_root_sum)
    return scaled_array.astype(np.int32)


class PowerIterator:
    """Module to determine an eigenvector using power iteration.

    Operates on 8-bit values.
    """

    def __init__(self, length=4):
        """Initialise.

        Args:
            length: integer setting the 1D size of the eigenvector
            - needs to be a power of 2.
        """
        assert isinstance(length, int)
        self.length = length
        # Initialise eigenvector as random vector
        # Set range to -127 to 127 (to be symmetrical)
        # But generate as 16 bit value as we normalise to 8-bit
        self.ev = np.random.randint(
            low=-127, high=128, size=(length, 1), dtype=np.int32)
        # Loop if we get all zeros
        while not self.ev.any():
            self.ev = np.random.randint(
                low=-127, high=128, size=(length, 1), dtype=np.int32)
        # Normalise the eigenvector using the L2 norm
        self.ev = normalise(self.ev)
        # Define placeholder for covariance matrix -
        # values will be 8-bit but we need 32-bit for future calculations
        self.cov = np.zeros(shape=(length, length), dtype=np.int32)
        # Define eigenvalue
        self.rayleigh = np.zeros(1, dtype=np.uint16)

    def iterate(self):
        """One pass of iteration.

        Applies power iteration with power = 1.
        Casts cov and ev to 16-bit, matmuls then casts
        back to 8-bit after scaling.

        We could even do this in 32-bit space then apply
        the scaling afterwards. I.e. in this case we won't need
        to divide by the length or by 127 and
        could use the same normalise routine.
        """
        # Check cov is not all zero - if all 0 you get nan
        if self.cov.any():
            # Just do one multiplication per round
            temp_ev = np.matmul(self.cov, self.ev)
            # Divide by 127 (the max value)
            self.ev = normalise(temp_ev)
        return self.ev

    @property
    def eigenvector(self):
        """Return the top eigenvector."""
        return self.ev.copy().astype(np.int8)

    @property
    def eigenvalue(self):
        """Return associated eigenvalue."""
        if self.cov.any():
            # Compute in 32-bit space
            top_1 = np.matmul(self.ev.T, self.cov)
            bottom = np.matmul(self.ev.T, self.ev)
            rayleigh = np.matmul(top_1, self.ev) / bottom
            rayleigh = rayleigh.astype(np.uint16).ravel()
            self.rayleigh = rayleigh
        return self.rayleigh

    def load_covariance(self, cov):
        """Update the covariance matrix."""
        # Remember to convert the input cov to 32-bit variable
        self.cov = cov.astype(np.int32)
        return None

    def __repr__(self):
        """Generate printable representation of state."""
        string = (
            f"Power Iterator - length {self.length}\n"
            f"Eigenvector:\n{self.eigenvector}\n"
            f"Eigenvalue:\n{self.eigenvalue}\n"
            f"Covariance:\n{self.cov}\n"
        )
        return string


"""Old - this may be needed if we had max 16bit not 32.
def normalise(array):
    ""Scale 8-bit array by L2 norm.

    Args:
        array - int8 numpy 1D array.
    ""
    # Convert to 16-bit space
    temp_array = array.astype(np.int16)
    # Square
    squared = temp_array**2
    # Scale by L
    squared = squared // array.shape[0]
    # Sum
    array_sum = squared.sum()
    # Square root
    sq_root = np.sqrt(array_sum)  # We can keep this is 16-bit space
    # Scale by max_value / sqrt(length) and divide by norm
    scaled_array = (temp_array*127)//(sq_root*np.sqrt(array.shape[0]))
    return scaled_array.astype(np.int8)
"""
