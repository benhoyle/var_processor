"""Some abstract Tkinter GUIs and objects."""

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from src.visualisers.spectrogram_gui import convert_to_chart


class AbstractGUI:

    def __init__(self, sensor):
        """Initialisation method."""
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
        self.sensor = None
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
        for panel in self.panels:
            pass

    def run(self):
        """Run the GUI."""
        self.update()
        self.window.mainloop()

    def quit_(self):
        """Stop the GUI."""
        self.sensor.stop()
        self.window.destroy()


class SignalPanels:
    """Object to display 1D signals."""

    def __init__(self, parent, sub_panels=4, width=256, height=64, packing=tk.TOP):
        """Initialise.

        Args:
            parent - parent Tk object.
            sub_panels - number of sub_panels.
            width - sub-panel width in pixels.
            height - sub-panel height in pixels.
        """
        self.frame = tk.Frame(parent)
        self.frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.panels = list()
        self.canvas_content = list()
        self.width = width
        self.height = height
        for _ in range(sub_panels):
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
                negative_chart_image = convert_to_chart(-1 * array, flip=False)
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


class ImagePanels:
    """Object to display a set of images."""

    def __init__(self, parent, sub_panels=4, width=256, height=256):
        """Initialise.

        Args:
            parent - parent Tk object.
            sub_panels - number of sub_panels.
            width - sub_panel width in pixels.
            height - sub_panel height in pixels.
        """
        # Create a parent frame
        self.frame = tk.Frame(parent)
        self.frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        # Create an empty list to hold panels to show
        self.panels = list()
        # Create an empty list to hold contents to show
        self.canvas_content = list()
        self.width = width
        self.height = height
        for _ in range(sub_panels):
            # Create a canvas for each panel
            canvas = tk.Canvas(self.frame, width=width, height=height)
            canvas.pack(side=tk.TOP)
            self.panels.append(canvas)
            self.canvas_content.append(None)

    def update(self, images):
        """Update frame with images.

        Args:
            images - a set of monochrome, RGB or RGBA arrays.
        """
        # Display the mean values of each buffer
        for i, image in enumerate(images):
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((self.width, self.height))
            photo_image = ImageTk.PhotoImage(image=pil_image)
            if self.canvas_content[i] is None:
                # print("Creating chart")
                self.canvas_content[i] = self.panels[i].create_image(0, 0, image=photo_image, anchor=tk.NW)
            else:
                # print("Configuring chart")
                self.panels[i].itemconfig(self.canvas_content[i], image=photo_image)
                self.panels[i].image = photo_image
