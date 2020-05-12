"""Stack Wrapper - Helper Class to wrap a stack for testing.

Stack and stage have very similar methods - may be able to abstract.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.var_processor.stack import Stack


class StackWrapper:
    """Wrapper for a stack to help testing."""

    def __init__(self, vec_len=4, input_length=256, buf_length=1000):
        """Initialise.

        Args:
            input_length - integer representing size of input data.
                Needs to be a power of vec_len.
            vec_len - integer representing the length of each
                VPU segment.
            buf_length - integer representing the number of iterations
                to buffer when testing.
        """
        # Testing a stage
        self.input_length = input_length
        self.vec_len = vec_len
        self.buf_length = buf_length
        self.stack = Stack(vec_len, input_length)
        # Get lengths of causes & predictions
        cause_lengths, pred_lengths = self.stack.get_lengths()
        # Generate buffers for each
        self.input_buffers = [
            np.zeros(shape=(cl, buf_length), dtype=np.int8)
            for cl in cause_lengths]
        self.pred_buffers = [
            np.zeros(shape=(pl, buf_length), dtype=np.int8)
            for pl in pred_lengths]
        # Generate buffer for residuals (same size as prediction buffer)
        self.residual_buffers = [pb.copy() for pb in self.pred_buffers]
        # Also add an initial entry in the input buffers for the original input
        self.input_buffers.insert(
            0, np.zeros(shape=(input_length, buf_length), dtype=np.int8))
        # Set a looping counter
        self.count = 0

    def iterate(self, input_signal):
        """Iterate.

        Args:
            input_signal - numpy 1D array of length size.
        """
        i = self.count
        # Add to buffer
        self.input_buffers[0][:, i] = input_signal.ravel()
        # Forward pass
        r = self.stack.forward(input_signal)
        causes = self.stack.get_causes()
        # Store in buffers
        for c, cause in enumerate(causes):
            self.input_buffers[c+1][:, i] = cause.ravel()
        self.input_buffers[-1][:, i] = r
        # Perform backward pass for predictions
        _ = self.stack.backward(r)
        preds = self.stack.get_pred_inputs()
        # Store in buffers & compute residuals
        for p, pred in enumerate(preds):
            self.pred_buffers[p][:, i] = pred.ravel()
            # Compute and clamp residuals
            residuals = np.clip(
                self.input_buffers[p][:, i] - pred.ravel(), -1, 1)
            self.residual_buffers[p][:, i] = residuals
        # Increment count and wrap if over 500
        self.count = (self.count + 1) % self.buf_length
        # Return end cause and start predictions / residuals
        return r, preds[0], self.residual_buffers[0][:, i]

    @property
    def pred_estimate(self):
        """Return signal prediction."""
        est = self.pred_buffers[0].sum(axis=1)/self.buf_length
        return est.reshape(-1, 1)

    @property
    def residual(self):
        """Get last residual signal."""
        return self.residual_buffers[0][:, self.count]

    def reconstruct(self, mean):
        """Reconstruct an input signal.

        Args:
            mean - 1D numpy array of length size containing signal mean.
        """
        return self.pred_estimate*mean

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
        # Plot in greyscale for patent figures
        plt.style.use('default')
        # Plot inputs/preds/residuals
        fig, axes = plt.subplots(3, self.stack.num_stages+1, sharex=True)
        # Plot inputs
        for i, input_buffer in enumerate(self.input_buffers):
            axes[0, i].imshow(input_buffer, aspect='auto')
        # Plot predictions
        for i, pred_buffer in enumerate(self.pred_buffers):
            axes[1, i].imshow(pred_buffer, aspect='auto')
        # Plot residuals
        for i, residual_buffer in enumerate(self.residual_buffers):
            axes[2, i].imshow(residual_buffer, aspect='auto')
        # Define title text
        col_text = [f"Stage {i}" for i in range(0, self.stack.num_stages)]
        col_text.append("Final Cause")
        row_text = ["Input", "Predictions", "Residuals"]
        # Set column labels
        for ax, col in zip(axes[0], col_text):
            ax.set_title(col)
        # Set row labels
        for ax, row in zip(axes[:, 0], row_text):
            ax.set_ylabel(row, rotation=90, size='large')
        # Remove ticks and labels
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.subplots_adjust(hspace=0.2)
        # Hide unused subplots
        axes[1, 4].set_visible(False)
        axes[2, 4].set_visible(False)
        plt.show()

    def plot_eigenvectors(self, spacer_len=0):
        """Plot the eigenvectors in the stack.

        Args:
            spacer_len - integer setting a spacing between eigenvectors.
        """
        evs = self.stack.get_eigenvectors()
        # Define result arrays with extra spacing
        result_arrays = [
            np.zeros((self.vec_len+spacer_len)*len(stage))
            for stage in evs]
        for result_array, stage_vpus in zip(result_arrays, evs):
            i = 0
            for vpu in stage_vpus:
                result_array[i:i+self.vec_len] = vpu.ravel()
                i += self.vec_len+spacer_len
        fig, axes = plt.subplots(len(evs), 1)
        for ax, result_array in zip(axes, result_arrays):
            ax.bar(np.arange(0, result_array.shape[0]), result_array)
            ax.set_xticks(
                np.arange(-0.5, result_array.shape[0], self.vec_len+spacer_len)
            )
            ax.xaxis.grid(True)
            ax.set_xticklabels([])
        fig.subplots_adjust(hspace=0.2)
        plt.show()
