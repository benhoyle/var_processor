"""Visualisers."""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class VPUVisualizer:
    """Object to visualise VPU."""

    def __init__(self, VPU, input_generator):
        """Initialise.

        Args:
            VPU: VPU object.
            input_generator: generator object that outputs an input for
            each iteration.
        """
        self.VPU = VPU
        self.input_gen = input_generator
        # Get array size
        size = VPU.size
        # Initialise subplots with 3 figs
        self.figure, self.axes = plt.subplots(1, 4)
        # Initialising animation
        self.animation = FuncAnimation(
            self.figure, self.update, save_count=200)
        # Initialise arrays - need shape (size,)
        self.input_array = self.residual_array = np.zeros(shape=(size, ))
        # Initialise Array Element (X) Axis
        self.x_range = np.arange(0, size)
        self.binary_range = np.arange(0, 1)
        # Configuring subplots
        self.input_plot = self.axes[0].bar(
            self.x_range, self.input_array, color='y')
        self.axes[0].set_xlabel("Array Element")
        self.axes[0].set_title("Input Data")
        self.residual_plot = self.axes[1].bar(
            self.x_range, self.residual_array, color='r')
        self.axes[1].set_xlabel("Array Element")
        self.axes[1].set_title("Residual Data")
        # 3rd Plot shows R scalar output
        self.r_plot = self.axes[2].bar(
            self.binary_range, self.binary_range, color='k')
        self.axes[2].set_title("R")
        # Third subplot for the eigenvector
        self.ev_plot = self.axes[3].bar(
            self.x_range, self.input_array, color='b')
        self.axes[3].set_xlabel("Array Element")
        self.axes[3].set_title("Eigenvector")
        self.size = size
        for ax in self.axes:
            ax.xaxis.set_major_locator(plt.NullLocator())
            # ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(wspace=0.5)

        # Bed in VPU
        for i in range(0, 100):
            input_data = next(self.input_gen)
            self.VPU.update_cov(input_data)

    def update(self, frame):
        """Update the visualisations."""
        # Get input data
        # print("updating")
        input_data = next(self.input_gen)
        r, residual = self.VPU.iterate(input_data)
        self.input_plot.remove()
        self.input_plot = self.axes[0].bar(
            self.x_range,
            input_data.reshape(self.size,),
            color='y')
        self.residual_plot.remove()
        self.residual_plot = self.axes[1].bar(
            self.x_range,
            residual.reshape(self.size,),
            color='r')
        self.r_plot.remove()
        self.r_plot = self.axes[2].bar(
            self.binary_range,
            r.reshape(1,),
            color='k'
        )
        self.ev_plot.remove()
        self.ev_plot = self.axes[3].bar(
            self.x_range,
            self.VPU.pi.eigenvector.reshape(self.size,),
            color='b'
        )
        time.sleep(0.25)
        return self.figure

    def show(self):
        """Show the visualisations."""
        plt.show()
