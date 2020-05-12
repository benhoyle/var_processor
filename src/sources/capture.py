"""Code for capturing sensory data."""
import threading
import struct
import logging
from collections import deque
import numpy as np

# Audio
import alsaaudio

# Abstract Class Import
from src.sources.abstract import SensorSource, CombinedSource

# Video Source Import
from src.sources.video import VideoSource


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


class AVCapture(CombinedSource):
    """Auto populate with audio and video."""

    def __init__(self):
        """Initialise."""
        super().__init__()
        audio = AudioSource()
        self.add_source(audio, "audio")
        video = VideoSource()
        self.add_source(video, "video")
