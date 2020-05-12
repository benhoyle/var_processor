"""Stack - Model for an area of cortex."""

import math
from src.var_processor.stage import Stage
from src.var_processor.abstract_classes import (
    AbstractSignalProcessor, TransformMixin
)


class Stack(AbstractSignalProcessor, TransformMixin):
    """Object to process a 1D sensor signal."""

    def __init__(self, vec_len, input_len):
        """Initialise sensor.

        Arg:
            vec_len - length of vector for VPU.
            input_len - length of input sensor vector - needs to be
                a power of vec_len.

        """
        super(Stack, self).__init__(vec_len, input_len)
        # self.time_len = time_len
        # Variable to store stages
        self.stages = list()
        num_stages = math.log(self.input_len, self.vec_len)
        self.num_stages = int(num_stages)
        self.build_stages()

    def build_stages(self):
        """Build a set of stages."""
        self.stages = [
            Stage(
                self.vec_len,
                int(self.input_len / self.vec_len**i)
            )
            for i in range(0, self.num_stages)
        ]

    def forward(self, forward_data, update_cov=True):
        """Forward pass through the stack.

        Args:
            input_data: 1D numpy array of ternary data.
        Returns:
            causes - numpy 1D array of causes.

        """
        # Iterate forward through the stages
        for stage in self.stages:
            if update_cov:
                stage.update_cov(forward_data)
            forward_data = stage.forward(forward_data)
        # Return scalar output from stack
        return forward_data

    def backward(self, backward_data):
        """Backward pass through the stack.

        Args:
            backward_data: feedback for last stage, scalar.
        """
        # Iterate through the stages backwards
        for stage in reversed(self.stages):
            backward_data = stage.backward(backward_data)
        # Return predicted data for stack
        return backward_data

    def update_cov(self, input_data):
        """Update the covariance matrices in a series of stages.

        We only have the input data for a next stage after a previous
        stage has finished.

        It makes more sense to update covariance as part of a forward
        pass?

        Args:
            input_data: 1D numpy array of ternary data.
        """
        # Update first stage
        self.stages[0].update_cov(input_data)
        # Perform a forward pass to get the causes
        self.forward(input_data)
        # Update each stage after the first using the causes
        for i in range(1, self.num_stages):
            # Get input data from previous stage
            input_data = self.stages[i-1].get_causes()
            # Pass to next stage to update_cov
            self.stages[i].update_cov(input_data)

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

    def get_outputs(self):
        """Return the outputs for the stack."""
        cause_output = self.stages[-1].get_causes()
        prediction_output = self.stages[0].get_pred_inputs()
        return cause_output, prediction_output

    def get_lengths(self):
        """Return the vector lengths of the causes and predicted inputs."""
        causes = self.get_causes()
        pred_inputs = self.get_pred_inputs()
        cause_lengths = [cause.shape[0] for cause in causes]
        pred_lengths = [pred.shape[0] for pred in pred_inputs]
        return cause_lengths, pred_lengths

    def get_eigenvectors(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [stage.get_eigenvectors() for stage in self.stages]
        return evs

    def get_eigenvalues(self):
        """Return a list of eigenvectors from the VPUs."""
        evs = [stage.get_eigenvalues() for stage in self.stages]
        return evs

    def get_covariances(self):
        """Return covariance matrices."""
        covs = [stage.get_covariances() for stage in self.stages]
        return covs
