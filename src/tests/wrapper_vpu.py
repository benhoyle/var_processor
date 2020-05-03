"""Test Wrapper for VPU.

To test reconstruction.
"""
import numpy as np


def add_to_array(array, frame):
    """Add a frame to a rolling array."""
    array = np.roll(array, -1, axis=1)
    # Add frame to end of buffer
    array[..., -1] = frame.flatten()
    return array


def non_linearity(array):
    """Apply a non-linearity to array."""
    signs = np.sign(array)
    rand_vals = np.random.uniform(size=array.shape)
    binary_values = np.where(np.abs(array) > rand_vals, 1, 0)
    return binary_values*signs


class VPUWrapper:
    """Test wrapper for VPU.

    Proto - stage/stack. We store 1000 samples in the buffer.
    """

    def __init__(self, vpu, buf_length=1000):
        """Initialise wrapper.

        Args:
            vpu - VPU object
            buf_length - scalar length to buffer data.
        """
        self.vpu = vpu
        # Temp fields for output
        self.input_buffer = np.zeros(shape=(vpu.vec_len, buf_length))
        self.pred_buffer = np.zeros(shape=(vpu.vec_len, buf_length))
        # self.nl_pred_buffer = np.zeros(shape=(vpu.vec_len, buf_length))
        self.r_buffer = np.zeros(shape=(1, buf_length))
        self.residual_buffer = np.zeros(shape=(vpu.vec_len, buf_length))

    def iterate(self, input_signal):
        """Iterate VPU."""
        # Update covariance data of VPU & power iterate
        self.vpu.update_cov(input_signal, power_iterate=True)
        # Get r
        r = self.vpu.forward(input_signal)
        pred = self.vpu.backward(r)
        # pred_nl = non_linearity(pred).astype(np.int8)
        # Calculate residual
        residual = input_signal.astype(np.int8) - pred
        # Store last input and pred for output repr
        # Create rolling sum of inputs
        self.input_buffer = add_to_array(self.input_buffer, input_signal)
        # Add to rolling sum of predictions
        self.pred_buffer = add_to_array(self.pred_buffer, pred)
        # self.nl_pred_buffer = add_to_array(self.nl_pred_buffer, pred_nl)
        self.r_buffer = add_to_array(self.r_buffer, r)
        self.residual_buffer = add_to_array(self.residual_buffer, residual)
        return r, pred, residual

    @property
    def input_estimate(self):
        """Get real-valued estimate of input over buffer."""
        # How do we get the mean in here?
        return (self.input_buffer.sum(axis=1).T / 1000).reshape(-1, 1)

    @property
    def pred_estimate(self):
        """Get real-valued estimate of predictions over buffer."""
        # How do we get the mean in here?
        return (self.pred_buffer.sum(axis=1).T / 1000).reshape(-1, 1)

    @property
    def error(self):
        """Get difference between real input estimate and prediction."""
        return self.input_estimate - self.pred_estimate

    def __repr__(self):
        """Get string status of VPU.

            Optional - NL Predictions: {self.nl_pred_buffer[:,-1].T}
        """
        return f"""
                Input: {self.input_buffer[:,-1].T}
                r: {self.r_buffer[:,-1]}
                ev: {self.vpu.pi.eigenvector.T}
                lambda: {self.vpu.pi.eigenvalue}
                Scaled r: {self.r_buffer[:,-1]*np.sqrt(self.vpu.pi.eigenvalue)}
                NL r: {non_linearity(self.r_buffer[:,-1])}
                Predictions: {self.pred_buffer[:,-1].T}
                Residual: {self.residual_buffer[:,-1].T}
                ------------------------------
        """
