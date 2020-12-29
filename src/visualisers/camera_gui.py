"""Camera GUI using TKinter"""

from collections import deque
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
from src.sources.polar_mapping import (
    generateLUT, setup_reduced_res, reduce_resolution
)
from src.var_processor.surface_stage import decompose


def display(image_array, label):
    """Convert image array to image for display.

    Args:
        image_array - 2D numpy array.
        label - tkinter label for display.
    Returns:
        la
    """
    a = Image.fromarray(image_array)
    b = ImageTk.PhotoImage(image=a)
    label.configure(image=b)
    label._image_cache = b


class BasicCameraGUI:
    """Basic camera viewer for Y channel."""

    def __init__(self, src=0):
        # Set Up Camera
        self.cam = cv2.VideoCapture(src)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Setup gui
        self.window = tk.Tk()
        # label for the original video frame
        self.original_image = tk.Label(master=self.window)
        self.original_image.grid(row=1, column=0)

        # label for fps
        self.fps_label = tk.Label(master=self.window)
        self.fps_label.grid(row=0, column=0)
        self.fps_label._frame_times = deque([0] * 5)  # arbitrary 5 frame average FPS
        # quit button
        self.quit_button = tk.Button(master=self.window, text='Quit', command=lambda: self.quit_())
        self.quit_button.grid(row=0, column=1)
        # setup the update callback
        self.window.after(0, func=lambda: self.update_all())

    def update_image(self):
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        display(Y, self.original_image)
        return Y

    def update_fps(self):
        frame_times = self.fps_label._frame_times
        frame_times.rotate()
        frame_times[0] = time.time()
        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        try:
            fps = int(float(count_of_deltas) / sum_of_deltas)
        except ZeroDivisionError:
            fps = 0
        self.fps_label.configure(text=f'FPS: {fps}')

    def update_all(self):
        _ = self.update_image()
        # Update Window
        self.window.update()
        self.update_fps()
        self.window.after(20, func=lambda: self.update_all())

    def run(self):
        self.window.mainloop()

    def quit_(self):
        self.cam.release()
        self.window.destroy()


class CameraGUI:

    def __init__(self, src=0, phase_width=256):
        # Set Up Camera and LUT for Polar Mapping
        self.cam = cv2.VideoCapture(src)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Capture a frame to set image sizes
        _, frame = self.cam.read()
        Y = frame[:, :, 0]
        self.centre = np.asarray(Y.shape) // 2
        self.final_radius = self.centre.min()
        self.phase_width = phase_width
        # Generate LUT
        self.LUT = generateLUT(
            self.centre, self.final_radius, phase_width=phase_width)
        # Setup gui
        self.window = tk.Tk()
        # label for the original video frame
        self.original_image = tk.Label(master=self.window)
        self.original_image.grid(row=0, column=0)
        # label for the polar video frame
        self.polar_image = tk.Label(master=self.window)
        self.polar_image.grid(row=0, column=1)
        # Label for right image
        self.right_image = tk.Label(master=self.window)
        self.right_image.grid(row=0, column=2)
        # Label for left image
        self.left_image = tk.Label(master=self.window)
        self.left_image.grid(row=0, column=3)

        # label for fps
        self.fps_label = tk.Label(master=self.window)
        self.fps_label.grid(row=2, column=0)
        # arbitrary 5 frame average FPS
        self.fps_label._frame_times = deque([0] * 5)
        # quit button
        self.quit_button = tk.Button(
            master=self.window, text='Quit', command=lambda: self.quit_())
        self.quit_button.grid(row=2, column=1)
        # setup the update callback
        self.window.after(0, func=lambda: self.update_all())

    def update_image(self):
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        # Show original frame
        display(cv2.pyrDown(Y), self.original_image)
        # Can change this to internal method
        converted = img2polar(Y, self.centre, self.final_radius,
                              self.LUT, phase_width=self.phase_width)
        display(converted, self.polar_image)
        # Show left / right hemisphere images
        rows, cols = converted.shape
        # right_image = np.flipud(converted[:, :cols//2].T)
        # left_image = np.fliplr(converted[:, cols//2:].T)
        right_image = converted[:, :cols // 2].T
        left_image = np.flip(converted[:, cols // 2:].T)
        display(right_image, self.right_image)
        display(left_image, self.left_image)
        # Update Window
        self.window.update()

    def update_fps(self):
        frame_times = self.fps_label._frame_times
        frame_times.rotate()
        frame_times[0] = time.time()
        sum_of_deltas = frame_times[0] - frame_times[-1]
        count_of_deltas = len(frame_times) - 1
        try:
            fps = int(float(count_of_deltas) / sum_of_deltas)
        except ZeroDivisionError:
            fps = 0
        self.fps_label.configure(text=f'FPS: {fps}')

    def update_all(self):
        self.update_image()
        self.update_fps()
        self.window.after(20, func=lambda: self.update_all())

    def run(self):
        self.window.mainloop()

    def quit_(self):
        self.cam.release()
        self.window.destroy()


class CamGUIReduced(CameraGUI):

    def __init__(self, src=0, phase_width=256):
        # Call parent init
        super().__init__(src, phase_width)
        # Setup reduced resolution parameters
        self.precomputed = setup_reduced_res(self.phase_width)
        # label for the reduced resolution frame
        self.reduced_image = tk.Label(master=self.window)
        self.reduced_image.grid(row=0, column=4)

    def update_image(self):
        # Can maybe refactor top bit so we can call with a super()
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        # Show original frame
        display(cv2.pyrDown(Y), self.original_image)
        # Can change this to internal method
        converted = img2polar(Y, self.centre, self.final_radius,
                              self.LUT, phase_width=self.phase_width)
        display(converted, self.polar_image)
        # Show left / right hemisphere images
        rows, cols = converted.shape
        # right_image = np.flipud(converted[:, :cols//2].T)
        # left_image = np.fliplr(converted[:, cols//2:].T)
        right_image = converted[:, :cols // 2].T
        left_image = np.flip(converted[:, cols // 2:].T)
        display(right_image, self.right_image)
        display(left_image, self.left_image)
        # Show reduced image
        output_list, output_image = reduce_resolution(
            converted, output_display=True, precomputed=self.precomputed)
        display(output_image, self.reduced_image)
        # Update Window
        self.window.update()


class DecomposeFrame:
    """Object to display decomposed images."""

    def __init__(self, parent, subpanels=4, width=256, height=256):
        """Initialise.

        Args:
            parent - parent Tk object.
            sub_panels - number of sub_panels.
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


class DeComGUI:
    """GUI to look at decompositions."""

    def __init__(self, src=0, run=True):
        # Setup gui
        self.window = tk.Tk()

        # Setup FPS and Quit Button on First Row
        button_frame = tk.Frame(self.window)
        button_frame.pack(expand=True, fill=tk.BOTH)
        # quit button
        self.quit_button = tk.Button(button_frame, text='Quit', command=self.quit_)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # Set Up Camera
        self.cam = cv2.VideoCapture(src)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Capture a frame to set image sizes
        _, frame = self.cam.read()
        Y = frame[:, :, 0]

        # Hardcode decomposition stages for now
        self.num_of_stages = 3

        # Create a frame for each stage and pack vertically
        self.frames = [
            DecomposeFrame(self.window, width=128, height=128)
            for _ in range(self.num_of_stages + 4)]

        if run:
            self.run()

    def update(self):
        # Get frame
        _, frame = self.cam.read()
        Y = frame[:, :, 0]
        image = Y
        image_lists = [[Y]]
        # Iteratively decompose
        for _ in range(self.num_of_stages + 4 - 1):
            # Convert to 16-bit to avoid overflow
            images = decompose(image.astype(np.int16))
            # Scale and convert back to 8-bit
            # Convert A to unsigned 8 bit
            images[0] = (images[0]).astype(np.uint8)
            # For others shift to positive and apply colour map
            # heatmap = cv2.applyColorMap(image, cv2.COLORMAP_HOT)
            images = [images[0]] + [
                cv2.applyColorMap((i + 128).astype(np.uint8), cv2.COLORMAP_JET) for i in images[1:]
            ]
            image_lists.append(images)
            # Set the next image as the average of the set
            image = images[0]

        for frame, image_list in zip(self.frames, image_lists):
            frame.update(image_list)

        self.window.after(10, self.update)

    def run(self):
        self.update()
        self.window.mainloop()

    def quit_(self):
        self.cam.release()
        self.window.destroy()
