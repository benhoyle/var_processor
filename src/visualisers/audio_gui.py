"""Audio GUI."""
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


def convert_to_chart(array, max_value=256):
    """Convert 1D numpy array to a chart image."""
    # Create a "blank", i.e. white with 255, image array
    blank_image = np.ones(shape=(max_value, array.shape[0]))*255
    # Create a 2D mask from the data
    mask = array[:, None] >= np.arange(max_value)
    # Then flip the mask to align with the image and set to 0 (black)
    blank_image[mask.T] = 0
    # We then just need to flip the output again to get back to the original
    chart_image = np.flipud(blank_image)
    return chart_image


class AudioGUI:

    def __init__(self, sensor=None, run=True):
        """Initialise."""
        # Setup gui
        self.window = tk.Tk()

        # Setup FPS and Quit Button on First Row
        button_frame = tk.Frame(self.window)
        button_frame.pack(expand=True, fill=tk.BOTH)

        # quit button
        self.quit_button = tk.Button(
            button_frame, text='Quit', command=self.quit_)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Create a canvas
        self.canvas = tk.Canvas(self.window, width=512, height=256)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        # Define a variable to hold the canvas contents
        self.canvas_content = None

        # Setup sensor
        if sensor:
            self.add_sensor(sensor)

        # Run if indicated
        if run:
            self.run()

    def add_sensor(self, sensor):
        """Add a sensor to the GUI for display."""
        self.sensor = sensor
        # Start sensor if not already started
        if not self.sensor.started:
            self.sensor.start()

    def update(self):
        """Update the GUI."""
        # Get audio frame
        _, data = self.sensor.read()
        # Convert to image of histogram
        chart_image = convert_to_chart(data)
        # display(chart_image, self.original_fft)
        pil_image = Image.fromarray(chart_image)
        pil_image = pil_image.resize((512, 256))
        photo_image = ImageTk.PhotoImage(image=pil_image)
        if self.canvas_content is None:
            self.canvas_content = self.canvas.create_image(
                0, 0, image=photo_image, anchor=tk.NW)
        else:
            self.canvas.itemconfig(self.canvas_content, image=photo_image)
            self.canvas.image = photo_image
        self.window.after(10, self.update)

    def run(self):
        """Run the GUI."""
        self.update()
        self.window.mainloop()

    def quit_(self):
        """Quit GUI."""
        self.sensor.stop()
        self.window.destroy()
