"""Code for capturing sensory data."""
import threading
import struct
import logging
from collections import deque
import numpy as np

# Video
import cv2
# Audio
import alsaaudio


class SensorSource:
    """Abstract object for a sensory modality."""

    def __init__(self):
        """Initialise object."""
        self.started = False
        self.thread = None

    def start(self):
        """Start capture source."""
        if self.started:
            print('[!] Asynchroneous capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(
            target=self.update,
            args=()
        )
        self.thread.start()
        return self

    def update(self):
        """Update data."""
        pass

    def read(self):
        """Read data."""
        pass

    def stop(self):
        """Stop daemon."""
        self.started = False
        self.thread.join()


class VideoSource(SensorSource):
    """Object for video using OpenCV."""

    def __init__(self, src=0):
        """Initialise video capture."""
        super().__init__()
        # width=640, height=480
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
        """Dummy enter."""
        return self

    def __exit__(self, exec_type, exc_value, traceback):
        """Extra code to close camera."""
        self.cap.release()


class AudioSource(SensorSource):
    """Object for audio using alsaaudio."""

    def __init__(self, sample_freq=44100, nb_samples=65536):
        """Initialise audio capture."""
        super().__init__()
        # Store variables
        self.sample_freq = sample_freq
        self.nb_samples = nb_samples
        # Initialise audio
        self.inp = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device="default"
        )
        # set attributes: Mono, frequency, 16 bit little endian samples
        self.inp.setchannels(1)
        self.inp.setrate(sample_freq)
        self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        # A period size of 512 means that there are 512 chunks read per s
        # see https://larsimmisch.github.io/
        # pyalsaaudio/terminology.html?
        # source=post_page-----d21d7b672305----------------------
        self.inp.setperiodsize(512)  # This is in Hz?
        self.read_lock = threading.Lock()
        # Create a FIFO structure for the data
        self._s_fifo = deque([0] * nb_samples, maxlen=nb_samples)
        self.length = 0
        self.read_lock = threading.Lock()

    def update(self):
        """Update based on new audio data."""
        while self.started:
            self.length, data = self.inp.read()
            if self.length > 0:
                # extract and format sample
                raw_smp_l = struct.unpack('h' * self.length, data)
                with self.read_lock:
                    self._s_fifo.extend(raw_smp_l)
            else:
                logging.error(
                    f'Sampler error occur'
                    '(l={self.length} and len data={len(data)})'
                )

    def read(self):
        """Read audio."""
        with self.read_lock:
            return self.length, np.asarray(self._s_fifo, dtype=np.int16)


class CombinedSource:
    """Object to combine multiple modalities."""

    def __init__(self):
        """Initialise."""
        self.sources = dict()

    def add_source(self, source, name=None):
        """Add a source object.

        source is a derived class from SensorSource
        name is an optional string name.
        """
        if not name:
            name = source.__class__.__name__
        self.sources[name] = source

    def start(self):
        """Start all sources."""
        for _, source in self.sources.items():
            source.start()

    def read(self):
        """Read from all sources.

        return as dict of tuples.
        """
        data = dict()
        for name, source in self.sources.items():
            data[name] = source.read()[1]
        return data

    def stop(self):
        """Stop all sources."""
        for _, source in self.sources.items():
            source.stop()

    def __del__(self):
        """Extra code to close camera."""
        for _, source in self.sources.items():
            if source.__class__.__name__ == "VideoSource":
                source.cap.release()

    def __enter__(self):
        """Dummy enter."""
        return self

    def __exit__(self, exec_type, exc_value, traceback):
        """Extra code to close camera."""
        for _, source in self.sources.items():
            if source.__class__.__name__ == "VideoSource":
                source.cap.release()


class AVCapture(CombinedSource):
    """Auto populate with audio and video."""

    def __init__(self):
        """Initialise."""
        super().__init__()
        audio = AudioSource()
        self.add_source(audio, "audio")
        video = VideoSource()
        self.add_source(video, "video")
