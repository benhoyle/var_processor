"""Stack - Model for an area of cortex."""

import math
from src.var_processor.time_stage import TimeStage


class Stack:
    """Object to process a 1D sensor signal."""

    def __init__(self, sensor_len, vec_len, time_len):
        """Initialise sensor.

        Arg:
            sensor_len - length of input sensor vector - needs to be
                a power of vec_len.
            vec_len - length of vector for VPU.
            time_len - length of time buffering.
        """
        self.sensor_len = sensor_len
        self.vec_len = vec_len
        self.time_len = time_len
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
        return TimeStage(self.vec_len, stage_len)

    def build_stages(self):
        """Build a set of stages."""
        self.stages = [
            self.generate_stage(
                int(self.sensor_len / self.vec_len**(i+1))
            )
            for i in range(0, self.num_stages)
        ]

    def iterate(self, stack_feedforward, stack_feedback=None):
        """High level processing loop."""
        # Set feedforward as input data
        feedforward = stack_feedforward
        # Iterate through pairs of timestages in series
        for ts_ff, ts_fb in zip(self.stages[:-1], self.stages[1:]):
            feedback = ts_fb.get_pred_inputs()
            # Get feedforward and feedback for buffer
            feedforward, feedback = ts_ff.iterate(feedforward, feedback)
        # Then feedforward to last stage with no feedback (for now)
        feedforward, _ = self.stages[-1].iterate(
            feedforward, stack_feedback
        )
        # Return r_out for last stage and predicted inputs for first stage
        return feedforward, self.stages[0].get_pred_inputs()

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
