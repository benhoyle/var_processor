"""Video Source.

Some more advanced pre-processing for video sources.
"""
import threading

# Video
import cv2

from src.sources.abstract import SensorSource


def flatten_frame(frame, vec_len):
    """Flatten a frame of video to a 1D array."""
    rows = frame.shape[0] // vec_len
    cols = frame.shape[1] // vec_len
    flattened = frame.reshape(
        rows, vec_len, cols, vec_len).swapaxes(2, 1).flatten()
    return flattened


def reconstruct_frame(array, vec_len, rows, cols):
    """Reconstruct a 1D array of clustered blocks into a 2D image.

    rows, cols are original shape."""
    return array.reshape(
        rows//vec_len, cols//vec_len, vec_len, vec_len
    ).swapaxes(2, 1).reshape(rows, cols)


def separate_components(frame, square=True):
    """Separate frame into YUV components.

    square - boolean - if true subsample the colour images so they are square
    """
    if square:
        components = frame[:, :, 0], frame[::2, 1::2, 1], frame[::2, 0::2, 1]
    else:
        components = frame[:, :, 0], frame[:, 1::2, 1], frame[:, 0::2, 1]
    return components


class VideoSource(SensorSource):
    """Object for video using OpenCV."""

    def __init__(self, src=0, width=640, height=480):
        """Initialise video capture."""
        super(VideoSource, self).__init__()
        # width=640, height=480
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # Turn off RGB conversion (property 16) to get YUV
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Watch out - UV are interlaced
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


class YSource(VideoSource):
    """Version of video source that just returns the lightness channel."""

    def read(self):
        """Read video."""
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame[..., 0]


class FlatYSource(VideoSource):
    """Returns flattened Y video."""

    def __init__(self, src=0, width=640, height=480, vec_len=4):
        """Initialise video capture."""
        # Store extra vector length variable
        self.vec_len = vec_len
        super(FlatYSource, self).__init__(
            src=src, width=width, height=height)

    def read(self):
        """Read video."""
        with self.read_lock:
            # Get Y frame as 0th on last index
            frame = self.frame.copy()[..., 0]
            grabbed = self.grabbed
            # Flatten frame
            frame = flatten_frame(frame, self.vec_len)
        return grabbed, frame
