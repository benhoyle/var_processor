"""Abstract Classes.

These are defined so we conserve common interfaces and use consistent
naming.
"""


class AbstractBase:
    """Original base class."""

    def __init__(self, vec_len):
        """Initialise.

        Args:
            vec_len: 8-bit unsigned integer setting
                the 1D size of an input.
        """
        assert isinstance(vec_len, int)
        self.vec_len = vec_len

    def update_cov(self, input_data):
        """Update covariance using input data.

        Args:
            input_data - 1D numpy array.
        """
        # Update covariance
        pass

    def reset(self):
        """Reset the object."""
        pass

    def __repr__(self):
        """Return a string representation."""
        string = f"{self.__class__.__name__} with vector length {self.vec_len}"
        return string


class AbstractSubUnit(AbstractBase):
    """A signal processing subunit that operates on short vectors."""

    @property
    def eigenvector(self):
        """Return the eigenvector."""
        pass

    @property
    def eigenvalue(self):
        """Return the eigenvalue."""
        pass

    @property
    def covariance(self):
        """Return the covariance."""
        pass

    def __repr__(self):
        """Generate printable representation of state."""
        string = super(AbstractSubUnit, self).__repr__()
        string += (
            f"Eigenvector:\n{self.eigenvector}\n"
            f"Eigenvalue:\n{self.eigenvalue}\n"
            f"Covariance:\n{self.covariance}\n"
        )
        return string


class AbstractSignalProcessor(AbstractBase):
    """Container for multiple signal processing subunits."""

    def __init__(self, vec_len, input_len):
        """Initialise.

        Args:
            vec_len: 8-bit unsigned integer setting
                the 1D size of a segment.
            input_len: 8-bit unsigned integer setting
                the 1D size of the input.
        """
        super(AbstractSignalProcessor, self).__init__(vec_len)
        self.input_len = input_len

    def get_eigenvectors(self):
        """Return eigenvectors."""
        pass

    def get_eigenvalues(self):
        """Return eigenvalues."""
        pass

    def get_covariances(self):
        """Return covariance matrices."""
        pass

    def get_causes(self):
        """Return causes."""
        pass

    def get_pred_inputs(self):
        """Return predicted inputs."""
        pass

    def __repr__(self):
        """Return a string representation."""
        string = (
            f"{self.__class__.__name__}:"
            f"Input length: {self.input_len}"
            f"Vector length: {self.vec_len}"
        )
        return string


class TransformMixin:
    """Mixin to add transformation functions."""

    def forward_pre_processing(self, forward_data):
        """Process input data for forward processing."""
        return forward_data

    def forward_post_processing(self, forward_output):
        """Process output data of forward processing."""
        return forward_output

    def backward_pre_processing(self, backward_data):
        """Process input data for backward processing."""
        return backward_data

    def backward_post_processing(self, backward_output):
        """Process output of backward processing."""
        return backward_output

    def forward(self, forward_data):
        """Forward pass to generate causes - r.

        Args:
            forward_data: 1D numpy array of length vec_len.
        Returns:
            forward_output: numpy array of output data

        """
        processed = self.forward_pre_processing(forward_data)
        forward_output = self.forward_post_processing(processed)
        return forward_output

    def backward(self, backward_data):
        """Backward pass to generate predictions - pred_inputs.

        Args:
            backward_data: 1D numpy array of causes - typically shape (1, 1)
        Returns:
            backward_output: numpy array of output

        """
        processed = self.backward_pre_processing(backward_data)
        backward_output = self.backward_post_processing(processed)
        return backward_output

    def iterate(self, forward_data, backward_data=None):
        """Iterate forward then backward."""
        forward_output = self.forward(forward_data)
        # If no feedback data perform a closed loop
        if backward_data is None:
            backward_data = forward_output
        backward_output = self.backward(backward_data)
        return forward_output, backward_output
