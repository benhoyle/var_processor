"""Stage Wrapper - Helper Class to wrap a stage for testing."""

import numpy as np
import matplotlib.pyplot as plt

from src.var_processor.stage import Stage


class StageWrapper:
    """Wrapper for a stage to help testing."""

    def __init__(self, size=256, vec_len=4, buf_length=1000):
        """Initialise.

        Args:
            size - integer representing size of input data.
                Needs to be a power of vec_len.
            vec_len - integer representing the length of each
                VPU segment.
            buf_length - integer representing the number of iterations
                to buffer when testing.
        """
        # Testing a stage
        self.size = size
        self.vec_len = vec_len
        self.buf_length = buf_length
        self.stage = Stage(vec_len, size//vec_len)
        # Generate buffers for testing
        self.input_buffer = np.zeros(shape=(size, buf_length))
        self.pred_buffer = np.zeros(shape=(size, buf_length))
        self.r_buffer = np.zeros(shape=(size//vec_len, buf_length))
        self.residual_buffer = np.zeros(shape=(size, buf_length))
        self.count = 0

    def iterate(self, input_signal):
        """Iterate.

        Args:
            input_signal - numpy 1D array of length size.
        """
        # Process stage
        self.stage.update_cov(input_signal)
        causes = self.stage.forward(input_signal)
        pred_inputs = self.stage.backward(causes)
        i = self.count % self.buf_length
        self.input_buffer[:, i] = input_signal.ravel()
        self.pred_buffer[:, i] = pred_inputs.ravel()
        self.r_buffer[:, i] = causes.ravel()
        self.residual_buffer[:, i] = (input_signal - pred_inputs).ravel()
        self.count += 1

    def reconstruct(self, mean):
        """Reconstruct an input signal.

        Args:
            mean - 1D numpy array of length size containing signal mean.
        """
        pred_average = self.pred_buffer.sum(axis=1)/self.buf_length
        pred_reconstruct = pred_average.reshape(-1, 1)*mean
        return pred_reconstruct

    def error(self, data_in, mean):
        """Get error between original signal and prediction.

        Args:
            data_in - 1D numpy array of length size containing original signal.
            mean - 1D numpy array of length size containing signal mean.
        """
        pred_reconstruct = self.reconstruct(mean)+mean
        error = data_in - pred_reconstruct
        mse = np.abs(error).astype(np.uint8).mean()
        return error, mse

    def plot_buffers(self):
        """Plot buffer contents for quick check."""
        fig, axes = plt.subplots(2, 2, sharex=True)
        axes[0, 0].set_title("Input Ternary Data")
        axes[0, 0].imshow(self.input_buffer, aspect='auto')
        axes[1, 0].set_title("Predicted Input")
        axes[1, 0].imshow(self.pred_buffer, aspect='auto')
        axes[0, 1].set_title("Residual Data")
        axes[0, 1].imshow(self.residual_buffer, aspect='auto')
        axes[1, 1].set_title("Causes")
        axes[1, 1].imshow(self.r_buffer, aspect='auto')
        fig.subplots_adjust(hspace=0.2)
        plt.show()
