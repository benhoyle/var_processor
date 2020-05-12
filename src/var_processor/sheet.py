"""Sheet - a set of stacks."""

import numpy as np

from src.var_processor.abstract_classes import (
    AbstractSignalProcessor, TransformMixin
)
from src.var_processor.stack import Stack


class Sheet(AbstractSignalProcessor, TransformMixin):
    """Object to process a 1D sensor signal using mulitple stacks."""

    def __init__(self, vec_len, input_len, stack_len=None):
        """Initialise sensor.

        Arg:
            vec_len - length of vector for VPU.
            input_len - length of input sensor vector - needs to be
                a power of vec_len.
            stack_len - integer number of stacks.

        """
        super(Sheet, self).__init__(vec_len, input_len)
        if not stack_len:
            stack_len = vec_len
        # Define stacks
        self.stacks = [
            Stack(vec_len, input_len) for _ in range(0, stack_len)
        ]
        self.stack_len = stack_len
        # Create a blank array for the causes
        self.causes = np.zeros(shape=(1, stack_len), dtype=np.int8)
        # Create a blank array for the inputs for each stack
        self.stack_inputs = np.zeros(
            shape=(input_len, stack_len), dtype=np.int8)

    def iterate(self, forward_data, backward_data=None, update_cov=True):
        """Forward pass through the sheet.

        Needs forward and backward passes through each stack in series.

        Args:
            forward_data: 1D numpy array of ternary data representing
                the sensor signal.
            backward_data: optional 1D numpy array of scalar causes for each
                stack.
        Returns:
            causes - numpy 1 x stack_len array of causes from the stacks.
            stack_inputs - numpy input_len x stack_len array of inputs
                for each stack.

        """
        # Iterate forward through the stages
        for i, stack in enumerate(self.stacks):
            # Buffer stack input
            self.stack_inputs[:, i] = forward_data.ravel()
            # Update covariance if indicated
            if update_cov:
                stack.update_cov(forward_data)
            # Forward pass through i-th stack
            stack_output = stack.forward(forward_data)
            # Buffer current causes
            self.causes[:, i] = stack_output.ravel()
            # Backward pass through i-th stack
            if backward_data is not None:
                # Closed loop mode
                stack_output = backward_data[:, i]
            stack_preds = stack.backward(stack_output)
            # Calculate and clip residual
            stack_residual = np.clip(
                forward_data - stack_preds, -1, 1)
            # Set residual as next stack input
            forward_data = stack_residual
        # Return causes and inputs
        return self.causes, self.stack_inputs

    def get_causes(self):
        """Return causes as a list of arrays."""
        return [
            stack.get_causes() for stack in self.stacks
        ]

    def get_pred_inputs(self):
        """Return predicted inputs as a list of arrays."""
        return [
            stack.get_pred_inputs() for stack in self.stacks
        ]

    def get_outputs(self):
        """Return the outputs for the stack."""
        return [
            stack.get_outputs() for stack in self.stacks
        ]

    def get_lengths(self):
        """Return the vector lengths of the causes and predicted inputs."""
        stack_lengths = [stack.get_lengths for stack in self.stacks]
        return stack_lengths

    def get_eigenvectors(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [stack.get_eigenvectors() for stack in self.stacks]
        return evs

    def get_eigenvalues(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [stack.get_eigenvalues() for stack in self.stacks]
        return evs

    def get_covariances(self):
        """Return covariance matrices."""
        covs = [stack.get_covariances() for stack in self.stacks]
        return covs
