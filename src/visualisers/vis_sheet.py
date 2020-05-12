"""Visualiser for a 1D Sheet."""
import time
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.var_processor.sheet import Sheet


class SheetVisualizer:
    """Object to visualise a set of stacks."""

    def __init__(self, sheet, sensor, buf_length=127):
        """Initialise.

        Args:
            sheet: a sheet object.
            sensor: a sensor object.
        """
        assert isinstance(sheet, Sheet)
        assert isinstance(sensor, Sensor)
        self.sheet = sheet
        self.sensor = sensor
        self.buf_length = buf_length
        # Start sensor if not started
        if not self.sensor.source.started:
            self.sensor.start()
        # Initialise single subplot
        self.figure, self.ax = plt.subplots()
        # Initialising animation
        self.animation = FuncAnimation(
            self.figure, self.update, save_count=200)
        # Initialise X Axis
        self.x_range = np.arange(0, sen_length)
        # Set buffer for inputs
        self.input_buffers = np.zeros(
            shape=(sheet.input_len, sheet.stack_len, self.buf_length),
            dtype=np.int8
        )
        # Set count
        self.count = 0
        # Initialise Labels, Colors and Linestyles
        self.labels = ["Raw Input", "Sensor Mean"]
        self.labels = self.labels + [f"Input for Stack {i}" for i in range(0, self.sheet.stack_len)]
        self.colors = cm.rainbow(np.linspace(0, 1, self.sheet.stack_len+2))
        self.linestyles = ["-", "--", ":", ":", ":", ":"]
        # Initialise Data Plots
        # Set variable to store lines to update
        self.lines = [None for i in range(0, self.sheet.stack_len+2)]
        # Start with blank data
        zero_y = np.zeros(shape=(sheet.stack_len+2, sheet.input_len))
        self.set_plots(zero_y)
        # Add legend
        self.ax.legend()
        self.ax.set_ylim(-127, 256)

    def set_plots(self, y_data):
        """Plot one or more lines on an axis.

        Args:
            y_data - list of 1D numpy arrays for y.
        """
        for i, y in enumerate(y_data):
            if self.lines[i] is None:
                self.lines[i], = self.ax.plot(
                    self.x_range,
                    y,
                    color=self.colors[i],
                    label=self.labels[i],
                    linestyle=self.linestyles[i]
                )
            else:
                self.lines[i].set_ydata(y)

    def update(self, frame):
        """Update the visualisations."""
        # Get mean-removed input data
        input_data, raw_frame = self.sensor.get_frame_plus_raw()
        causes, stack_inputs = self.sheet.iterate(input_data)
        self.input_buffers[:, :, self.count] = stack_inputs
        # Sum across samples in buffer
        sums = self.input_buffers.sum(axis=2)
        # Get y_data as list
        y_data = [raw_frame, self.sensor.mean] + list(sums.T)
        # Update plots
        self.set_plots(y_data)
        # Increment Count and wrap if necessary
        self.count = (self.count + 1) % self.buf_length
        # time.sleep(0.25)
        return self.figure

    def show(self):
        """Show the visualisations."""
        plt.show()
