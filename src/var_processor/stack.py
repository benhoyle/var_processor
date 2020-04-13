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

    def forward(self, sw_residuals):
        """Forward pass through the stack.

        Args:
            sw_residuals: weighted residuals from switch, list of arrays.
        """
        pass

    def backward(self, stack_feedback):
        """Backward pass through the stack.

        Args:
            stack_feedback: feedback for last stage, scalar.
        """
        pass

    def update_cov(self, orig_input):
        """Update the covariance matrices.

        Run after computing the FF outputs in a forward pass.

        Args:
            orig_input: original un-switch-filtered input as array.
        """
        # for stage in stages:
        pass


    def iterate(self, orig_inputs, sw_residuals, stack_feedback):
        """High level processing loop.

        Args:
            orig_input: original un-switch-filtered input, list of
                arrays.
            sw_residuals: weighted residuals from switch, list of arrays.
            stack_feedback: feedback for last stage, scalar.

        Returns:
            ff_outputs: FF outputs from each stage.
            predictions: FB outputs from each stage.

        """
        # How do we get current prediction when we haven't iterated?
        # Get prediction from last time stamp?
        # Or do a forward pass first, then do a backard pass?
        for i in range(0, self.num_stages-1):
            # Get predicted inputs for current stage
            prediction = self.stages[i].get_pred_inputs()
            # Compute FF input by adding residuals to prediction
            stage_ff_input = sw_residuals[i] + prediction
            # Get FB input from next stage
            stage_fb_input = self.stages[i+1].get_pred_inputs()
            # Iterate current FF stage
            feedforward, _ = self.stages[i].iterate(
                orig_inputs[i],
                stage_ff_input,
                stage_fb_input
            )
        # Then feedforward to last stage with stack_feedback
        feedforward, _ = self.stages[-1].iterate(
            orig_inputs[-1],
            feedforward,
            stack_feedback
        )
        # Return r_out for last stage
        return feedforward

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
