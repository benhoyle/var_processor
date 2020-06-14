"""Spectogram GUI."""

import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

import math


def convert_to_chart(array, max_value=256, flip=True):
    """Convert 1D numpy array to a chart image.

    Args:
        array - 1D numpy with data to convert to plot.
        max_value - maximum value in the array.
        flip - boolean flag to correct the output when positive."""
    # Create a "blank", i.e. white with 255, image array
    blank_image = np.ones(shape=(max_value, array.shape[0]))*255
    # Create a 2D mask from the data
    mask = array[:, None] >= np.arange(max_value)
    # Then flip the mask to align with the image and set to 0 (black)
    blank_image[mask.T] = 0
    # We then just need to flip the output again to get back to the original
    if flip:
        chart_image = np.flipud(blank_image)
    else:
        chart_image = blank_image
    return chart_image


def decompose(image):
    """Decompose an image using the Hadamard transform."""
    H = image[1:, :] - image[:-1, :]
    V = image[:, 1:] - image[:, :-1]
    # Work out top TLHC + BRHC
    d_1 = image[:-1, :-1] + image[1:, 1:]
    # Then work out BLHC + TLHC
    d_2 = image[1:, :-1] + image[:-1, 1:]
    D = d_1 - d_2
    # Average is sum of four shifted versions
    A = image[:-1, :-1] + image[1:, :-1] + image[:1, -1] + image[1:, 1:]
    return [A, H, V, D]


def add_to_array(array, frame):
    """Add a frame to a rolling array."""
    array = np.roll(array, -1, axis=1)
    # Add frame to end of buffer
    array[..., -1] = frame
    return array


def norm_scale(a):
    """Normalise array to a scale of 0 to 1."""
    return (a - np.min(a))/np.ptp(a)


def mono_to_RGB(image, white_mode=True):
    """Convert a grayscale signed image to RGB.

    Red = positive, green = negative."""
    RGB = np.zeros(shape=(image.shape)+(3, ))
    # Red = positive
    RGB[:, :, 0] = np.maximum(image, 0)
    # Green
    RGB[:, :, 1] = np.maximum(-image, 0)
    if white_mode:
        # Get zero values
        mask = (RGB == 0)
        # set to white
        RGB[mask] = 255
    return RGB


def mono_to_RGBA(image):
    """Convert a grayscale signed image to RGB.

    Red = positive, green = negative."""
    RGBA = np.zeros(shape=(image.shape)+(4, ))
    # Red = positive
    RGBA[:, :, 0] = np.maximum(image, 0)
    # Green
    RGBA[:, :, 1] = np.maximum(-image, 0)
    # Set non-zero values to have alpha  = 255
    nz_rows, nz_cols, _ = RGBA.nonzero()
    RGBA[nz_rows, nz_cols, 3] = 255
    return RGBA


class Buffer:
    """Object for a time buffer."""

    def __init__(self, data_length, time_length):
        """Initialise object.

        Assumes 8-bit values (for now).
        """
        # Set up an array to store a rolling window of inputs
        self.array = np.zeros(
            shape=(data_length, time_length), dtype=np.int16)

    def add(self, frame):
        """Add a frame to the buffer in FF mode."""
        self.array = add_to_array(self.array, frame)
        return None

    @property
    def latest(self):
        """Return latest entry in buffer."""
        return self.array[..., -1]


class SpectrogramGUI:

    def __init__(self, sensor, vec_len=4, run=True):
        # Setup gui
        self.window = tk.Tk()

        # Setup FPS and Quit Button on First Row
        button_frame = tk.Frame(self.window)
        button_frame.pack(expand=True, fill=tk.BOTH)
        # quit button
        self.quit_button = tk.Button(
            button_frame, text='Quit', command=self.quit_)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Create a list of panels for the GUI
        self.panels = list()
        self.canvas_content = list()
        # We have an extra canvas for the original data
        for _ in range(vec_len+1):
            # Create a canvas for each panel
            canvas = tk.Canvas(self.window, width=256, height=256)
            canvas.pack(side=tk.LEFT)
            self.panels.append(canvas)
            self.canvas_content.append(None)

        # Setup sensor
        self.add_sensor(sensor)
        _, data = sensor.read()

        self.vec_len = vec_len
        # Determine the number of buffers based on data length and vec_len
        self.num_of_buffers = int(math.log(data.shape[0], vec_len))
        # We can crop the data to a factor of vec_len
        self.cropped_data_length = vec_len**self.num_of_buffers
        # Start with a single cuboid buffer that is decomposed
        self.buffer = Buffer(
            self.cropped_data_length, self.cropped_data_length)
        """
        # Create buffers - set as cuboid
        self.buffers = list()
        for i in range(num_of_buffers):
            buffer_size = self.cropped_data_length // (vec_len**i)
            # But
            self.buffers.append(Buffer((buffer_size, )*3))
        """
        if run:
            self.run()

    def add_sensor(self, sensor):
        """Add a sensor to the GUI for display."""
        self.sensor = sensor
        # Start sensor if not already started
        if not self.sensor.started:
            self.sensor.start()

    def update(self):
        # Get audio frame
        _, data = self.sensor.read()
        input_data = data[:self.cropped_data_length].reshape(-1, 1)
        # Add to buffer
        self.buffer.add(input_data.ravel())
        # Get buffer image
        image = 255-self.buffer.array
        # Decompose
        components = decompose(image)
        # Scale components

        components[0] = components[0] // self.vec_len
        components[1] = mono_to_RGB(components[1])
        components[2] = mono_to_RGB(components[2])
        # As we have two additions and a subtraction
        # we need to additionally scale
        components[3] = mono_to_RGB(components[3]//2)

        """
        # Iterate through buffers in reverse
        for buffer in self.buffers:
            buffer.add(input_data)
            # Get image from buffer
            image = buffer.array
            # Average is sum of four shifted versions - specific to vec_len = 4
            A = image[:-1,:-1] + image[1:,:-1] + image[:1,-1] + image[1:,1:]
            input_data = A[::self.vec_len]
        """
        # Group images in one list and scale components
        images_for_display = [image.astype(np.uint8)]
        images_for_display += [c.astype(np.uint8) for c in components]
        # images_for_display = [image] + components
        # Display the mean values of each buffer
        for i, image in enumerate(images_for_display):
            # print(i, image.max(), image.min(), image.dtype)
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize((256, 256))
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

        self.window.after(10, self.update)

    def run(self):
        self.update()
        self.window.mainloop()

    def quit_(self):
        self.sensor.stop()
        self.window.destroy()
