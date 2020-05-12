"""Video Source.

Some more advanced pre-processing for video sources.
"""
import threading
import numpy as np

# Video
import cv2

from src.sources.abstract import SensorSource


class VideoSource(SensorSource):
    """Object for video using OpenCV."""

    def __init__(self, src=0, width=640, height=480):
        """Initialise video capture."""
        super().__init__()
        # width=640, height=480
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # Turn off RGB conversion (property 16) to get YUV
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Set width / height if passed (locks to nearest availabe resolution)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Grab a frame
        self.grabbed, self.frame = self.cap.read()
        self.read_lock = threading.Lock()

    def update(self):
        """Update based on new video data."""
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """Read video."""
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        """Stop daemon."""
        # Run parent stop
        super(VideoSource, self).stop()
        # Release camera
        self.cap.release()

    def __enter__(self):
        """Enter - dummy."""
        return self

    def __exit__(self, exec_type, exc_value, traceback):
        """Extra code to close camera."""
        self.cap.release()
