"""Some abstract Tkinter GUIs and objects."""

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from src.visualisers.spectrogram_gui import convert_to_chart


class AbstractGUI:

    def __init__(self, sensor):
        """Initiailisation method."""
        # Setup gui
        self.window = tk.Tk()

        # Setup FPS and Quit Button on First Row
        button_frame = tk.Frame(self.window)
        button_frame.pack(expand=True, fill=tk.BOTH)
        # quit button
        self.quit_button = tk.Button(
            button_frame, text='Quit', command=self.quit_)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Setup sensor
        self.add_sensor(sensor)

        # Create a list of panels for the GUI
        self.panels = list()
        self.canvas_content = list()

    def add_sensor(self, sensor):
        """Add a sensor to the GUI for display."""
        self.sensor = sensor
        # Start sensor if not already started
        if not self.sensor.started:
            self.sensor.start()

    def update(self):
        """Set here for update GUI method."""
        pass

    def run(self):
        """Run the GUI."""
        self.update()
        self.window.mainloop()

    def quit_(self):
        """Stop the GUI."""
        self.sensor.stop()
        self.window.destroy()


class SignalFrame:
    """Object to display signals."""

    def __init__(self, parent, subpanels=4, width=256, height=64, packing=tk.TOP):
        """Initialise.

        Args:
            parent - parent Tk object.
            subpanels - number of subpanels.
            width - subpanel width in pixels.
            height - subpanel height in pixels.
        """
        self.frame = tk.Frame(parent)
        self.frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.panels = list()
        self.canvas_content = list()
        self.width = width
        self.height = height
        for _ in range(subpanels):
            # Create a canvas for each panel
            canvas = tk.Canvas(self.frame, width=width, height=height)
            canvas.pack(side=packing)
            self.panels.append(canvas)
            self.canvas_content.append(None)

    def update(self, arrays, signed=True):
        """Update frame with image of array.

        Args:
            arrays - a list of numpy arrays.
        """
        # Display the mean values of each buffer
        for i, array in enumerate(arrays):
            chart_image = convert_to_chart(array)
            if signed:
                negative_chart_image = convert_to_chart(-1*array, flip=False)
                chart_image = np.concatenate(
                    [chart_image, negative_chart_image], axis=0)
            pil_image = Image.fromarray(chart_image)
            pil_image = pil_image.resize((self.width, self.height))
            photo_image = ImageTk.PhotoImage(image=pil_image)
            if self.canvas_content[i] is None:
                # print("Creating chart")
                self.canvas_content[i] = self.panels[i].create_image(
                    0, 0, image=photo_image, anchor=tk.NW)
            else:
                # print("Configuring chart")
                self.panels[i].itemconfig(
                    self.canvas_content[i], image=photo_image)
                self.panels[i].image = photo_image
