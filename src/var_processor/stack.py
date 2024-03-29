"""Stack - Model for an area of cortex."""

import math
from src.var_processor.stage import Stage


class Stack:
    """Object to process a 1D sensor signal."""

    def __init__(self, sensor_len, vec_len):
        """Initialise sensor.

        Arg:
            sensor_len - length of input sensor vector - needs to be
                a power of vec_len.
            vec_len - length of vector for VPU.
        """
        self.sensor_len = sensor_len
        self.vec_len = vec_len
        # self.time_len = time_len
        # Variable to store time stages
        self.stages = list()
        num_stages = math.log(self.sensor_len, self.vec_len)
        self.num_stages = int(num_stages)
        self.build_stages()

    def generate_stage(self, stage_len):
        """Generate a stage.

        Arg:
            stage_len - integer number of stages.
        """
        return Stage(self.vec_len, stage_len)

    def build_stages(self):
        """Build a set of stages."""
        self.stages = [
            self.generate_stage(
                int(self.sensor_len / self.vec_len**(i+1))
            )
            for i in range(0, self.num_stages)
        ]

    def forward(self, input_data, update_cov=True):
        """Forward pass through the stack.

        Args:
            input_data: 1D numpy array of ternary data.
        Returns:
            causes - numpy 1D array of causes.

        """
        # Iterate forward through the stages
        for stage in self.stages:
            if update_cov:
                stage.update_cov(input_data)
            input_signal = stage.forward(input_data)
        # Return scalar output from stack
        return input_signal

    def backward(self, stack_feedback):
        """Backward pass through the stack.

        Args:
            stack_feedback: feedback for last stage, scalar.
        """
        feedback_data = stack_feedback
        # Iterate through the stages backwards
        for stage in reversed(self.stages):
            feedback_data = stage.backward(feedback_data)
        # Return predicted data for stack
        return feedback_data

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
