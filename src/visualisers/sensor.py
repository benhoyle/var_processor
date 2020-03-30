"""Visualiser for a Sensor."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SensorVisualizer:
    """Object to visualise Sensor.

    We can likely make this simpler and more efficient once we
    get it working.

    """

    def __init__(self, sensor):
        """Initialise.

        Args:
            sensor - Sensor object.
        """
        self.sensor = sensor
        # Start sensor if not started
        if not self.sensor.source.started:
            self.sensor.start()
        # Get number of stages
        num_stages = self.sensor.num_stages
        # Initialise subplots for stages plus top level signal
        self.figure, self.axes = plt.subplots(num_stages+1, 2)
        # Initialising animation
        self.animation = FuncAnimation(
            self.figure, self.update, save_count=200)
        # Set up x ranges for each bar plot
        # Initialise raw data x range
        self.x_d_range = None
        # Initialise causes and residuals lengths for x axis
        self.x_c_ranges = list()
        self.x_r_ranges = list()
        # Set up containers for bar plots
        self.data_bar = None
        self.cause_bars = list()
        self.pred_input_bars = list()
        # Set Titles
        self.axes[0][0].set_title("Signal & Causes")
        self.axes[0][1].set_title("Residuals")
        # Setup variable to store bar plots
        # Clear ticks
        for ax in self.axes.ravel():
            ax.xaxis.set_major_locator(plt.NullLocator())
            # ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(wspace=0.5)

    def update(self, frame):
        """Update the visualisations."""
        # Iterate sensor
        frame = self.sensor.iterate()
        causes = self.sensor.get_causes()
        pred_inputs = self.sensor.get_pred_inputs()

        # For bar plots we need to iterate through previous plots
        # and remove then replot

        # Redraw sensor data if exists - else set x range
        if self.data_bar:
            self.data_bar.remove()
        else:
            self.x_d_range = np.linspace(0, frame.shape[0]-1, frame.shape[0])
        self.data_bar = self.axes[0][0].bar(
            self.x_d_range,
            frame.ravel(),
            color='r'
        )

        # Plot causes first  - iterate through axes[0, 1:]
        for i, cause in enumerate(causes):
            if i < len(self.cause_bars):
                self.cause_bars[i].remove()
            else:
                self.x_c_ranges.append(
                    np.linspace(0, cause.shape[0]-1, cause.shape[0])
                )
            bar_chart = self.axes[i+1][0].bar(
                self.x_c_ranges[i],
                cause.ravel(),
                color='b'
            )
            if i < len(self.cause_bars):
                self.cause_bars[i] = bar_chart
            else:
                self.cause_bars.append(bar_chart)
        # Then plot residuals - iterate through axes[1, 1:]
        for i, pred_input in enumerate(pred_inputs):
            if i < len(self.pred_input_bars):
                self.pred_input_bars[i].remove()
            else:
                self.x_r_ranges.append(
                    np.linspace(0, pred_input.shape[0]-1, pred_input.shape[0])
                )
            bar_chart = self.axes[i][1].bar(
                self.x_r_ranges[i],
                pred_input.ravel(),
                color='g'
            )
            if i < len(self.pred_input_bars):
                self.pred_input_bars[i] = bar_chart
            else:
                self.pred_input_bars.append(bar_chart)

        # time.sleep(0.1)
        return self.figure

    def show(self):
        """Show the visualisations."""
        plt.show()
