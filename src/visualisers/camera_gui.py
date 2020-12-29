"""Camera GUI using TKinter"""

from collections import deque
import time
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
from src.sources.polar_mapping import (
    generateLUT, setup_reduced_res, reduce_resolution, convert_image, forward_quad
)
from src.var_processor.surface_stage import decompose
from src.var_processor.pb_threshold import pb_threshold, pb_residual_threshold


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

    def __init__(self, src=0, run=False):
        # Set Up Camera
        self.cam = cv2.VideoCapture(src)
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Setup gui
        self.window = tk.Tk()

        # Setup FPS and Quit Button on First Row
        button_frame = tk.Frame(self.window)
        button_frame.pack(expand=True, fill=tk.BOTH)
        # label for fps
        self.fps_label = tk.Label(button_frame)
        self.fps_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.fps_label._frame_times = deque([0] * 5)  # arbitrary 5 frame average FPS
        # quit button
        self.quit_button = tk.Button(button_frame, text='Quit', command=self.quit_)
        self.quit_button.pack(side=tk.RIGHT, padx=5, pady=5)

        # label for the original video frame
        video_frame = tk.Frame(self.window)
        video_frame.pack(expand=True, fill=tk.BOTH)
        self.original_image = tk.Label(video_frame)
        self.original_image.pack(padx=5, pady=5)

        # setup the update callback
        self.window.after(0, func=lambda: self.update_all())

        # Run if indicated
        if run:
            self.run()

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


class PolarGUI(BasicCameraGUI):
    """Convert base camera into polar co-ordinates."""

    def __init__(self, src=0, phase_width=256):
        # Call parent init
        super().__init__(src)

        # Capture a frame to set image sizes
        _, frame = self.cam.read()
        Y = frame[:, :, 0]
        self.centre = np.asarray(Y.shape) // 2
        self.radius = self.centre.max()
        self.phase_width = phase_width
        # Generate LUT
        self.LUT = generateLUT(self.radius, phase_width=phase_width)

    def update_image(self):
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        converted = convert_image(Y, self.LUT)
        display(converted, self.original_image)
        return converted


class QuadPolar(PolarGUI):
    """Show different portions of visual field."""

    def __init__(self, src=0, phase_width=256):
        # Call parent init
        super().__init__(src, phase_width)

        # Add a decompose frame to show quad
        self.frame = DecomposeFrame(self.window, width=128, height=128)

    def update_image(self):
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        converted = convert_image(Y, self.LUT)
        display(converted, self.original_image)
        images = forward_quad(converted)
        self.frame.update(images)
        return converted


class CamGUIReduced(QuadPolar):

    def __init__(self, src=0, phase_width=256):
        # Call parent init
        super().__init__(src, phase_width)
        # Setup reduced resolution parameters
        self.precomputed = setup_reduced_res(self.phase_width)
        # frame and label for the reduced resolution frame
        reduced_frame = tk.Frame(self.window)
        reduced_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.reduced_image = tk.Label(reduced_frame)
        self.reduced_image.pack(padx=5, pady=5)

    def update_image(self):
        converted = super().update_image()
        # Show reduced image
        output_list, output_image = reduce_resolution(
            converted, output_display=True, precomputed=self.precomputed)
        display(output_image, self.reduced_image)
        return converted


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


class DeComGUI(BasicCameraGUI):
    """GUI to look at decompositions."""

    def __init__(self, src=0):
        # Call parent init
        super().__init__(src)

        # Hardcode decomposition stages for now
        self.num_of_stages = 8

        # Create a frame for each stage and pack vertically
        self.frames = [
            DecomposeFrame(self.window, width=128, height=128)
            for _ in range(self.num_of_stages)]

    def update_image(self):
        # Get frame
        (readsuccessful, frame) = self.cam.read()
        Y = frame[:, :, 0]
        image = Y
        image_lists = [[Y]]
        # Iteratively decompose
        for _ in range(self.num_of_stages - 1):
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
        return Y


class PBTDeComGUI(DeComGUI):
    """Apply PBT to the surfaces (separately to A and differences)."""

    def update_image(self):
        # Get frame
        _, frame = self.cam.read()
        Y = frame[:, :, 0]
        image = Y
        image_lists = [[Y]]
        # Iteratively decompose
        for _ in range(self.num_of_stages + 4 - 1):
            # Convert to 16-bit to avoid overflow
            images = decompose(image.astype(np.int16))
            # Set A as input for next stage
            image = images[0]
            # PBT A - remember to cast to 8-bit
            A_pbt = pb_threshold(images[0].astype(np.uint8)) * 255
            # PBT differences
            diffs = list()
            for i in images[1:]:
                thresholded = pb_residual_threshold(i)
                rescaled = (thresholded * 127) + 127
                color_mapped = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
                diffs.append(color_mapped)
            images = [A_pbt.astype(np.uint8)] + diffs
            image_lists.append(images)

        for frame, image_list in zip(self.frames, image_lists):
            frame.update(image_list)
        return Y


class PolarPBT(PolarGUI):
    """Perform PBT decomposition on a polar decomposed image."""
    def __init__(self, src=0, stages=7):
        # Call parent init
        super().__init__(src)

        # Hardcode decomposition stages for now
        self.num_of_stages = stages

        # Create a frame for each stage and pack vertically
        self.frames = [
            DecomposeFrame(self.window, width=128, height=128)
            for _ in range(self.num_of_stages)]

    def update_image(self):
        # Get frame
        converted = super().update_image()
        image = converted
        image_lists = [[image]]
        # Iteratively decompose
        for _ in range(self.num_of_stages + 4 - 1):
            # Convert to 16-bit to avoid overflow
            images = decompose(image.astype(np.int16))
            # Set A as input for next stage
            image = images[0]
            # PBT A - remember to cast to 8-bit
            A_pbt = pb_threshold(images[0].astype(np.uint8)) * 255
            # PBT differences
            diffs = list()
            for i in images[1:]:
                thresholded = pb_residual_threshold(i)
                rescaled = (thresholded * 127) + 127
                color_mapped = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
                diffs.append(color_mapped)
            images = [A_pbt.astype(np.uint8)] + diffs
            image_lists.append(images)

        for frame, image_list in zip(self.frames, image_lists):
            frame.update(image_list)
        return converted


class PBTPolarQuad(PolarGUI):
    """Show different portions of visual field."""

    def __init__(self, src=0, phase_width=256):
        # Call parent init
        super().__init__(src, phase_width)

        # Add a decompose frame to show quad
        self.quad_frame = DecomposeFrame(self.window, width=128, height=128)

        # Create a frame for each quad and pack vertically
        self.decom_frames = [
            DecomposeFrame(self.window, width=128, height=128)
            for _ in range(4)]

    def update_image(self):
        # Get frame
        converted = super().update_image()
        quad_images = forward_quad(converted)
        self.quad_frame.update(quad_images)
        for quad, frame in zip(quad_images, self.decom_frames):
            image = quad
            image_lists = [[image]]
            # Convert to 16-bit to avoid overflow
            images = decompose(image.astype(np.int16))
            # PBT A - remember to cast to 8-bit
            A_pbt = pb_threshold(images[0].astype(np.uint8)) * 255
            # PBT differences
            diffs = list()
            for i in images[1:]:
                thresholded = pb_residual_threshold(i)
                rescaled = (thresholded * 127) + 127
                color_mapped = cv2.applyColorMap(rescaled.astype(np.uint8), cv2.COLORMAP_JET)
                diffs.append(color_mapped)
            images = [A_pbt.astype(np.uint8)] + diffs
            frame.update(images)
        return converted
