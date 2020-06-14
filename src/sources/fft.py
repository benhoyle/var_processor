"""FFT Source."""
import numpy as np
from scipy.fftpack import fft
from src.sources.capture import AudioSource
from src.sources.abstract import CombinedSource
from src.sources.video import VideoSource


def norm_scale(a):
    """Normalise array to a scale of 0 to 1."""
    return (a - np.min(a))/np.ptp(a)


def midi_tune(array, integer=False):
    """Convert an array of frequencies to a MIDI tuning."""
    array = (69 + 12*np.log2((array / 440)))
    if integer:
        array = array.astype(np.uint8)
    return array


class FFTSource(AudioSource):
    """Return frequency data transformed to a useable scale."""

    def __init__(self, sample_freq=44100, nb_samples=65536, res_factor=1):
        """Adding X Scale initialisation."""
        super().__init__(sample_freq, nb_samples)
        # Start at 10 for +ve values from midi tune
        sample_loc = np.linspace(
            10,
            sample_freq // 2,
            nb_samples // 2,
            dtype=np.uint16
        )
        # Use midi_tune to convert to log scale
        log_sample_loc = midi_tune(sample_loc)
        # Use resolution_factor to control resolution of log samples
        log_sample_loc = (log_sample_loc*res_factor)
        if res_factor == 1:
            self.x_scale = log_sample_loc.astype(np.uint8)
        else:
            self.x_scale = log_sample_loc.astype(np.uint16)

    def read(self):
        """Read fft."""
        with self.read_lock:
            samples = np.asarray(self._s_fifo, dtype=np.int16)
        y_freq = fft(samples)
        # level axe at each frequency:
        # yf between 0.0 and 1.0 for every xf step
        # This is also taking the first real half
        fft_y_data = (
            (1.0 / (self.nb_samples / 2)) *
            np.abs(y_freq[0:self.nb_samples // 2])
        )

        # Use bincount to get the sum for each unique x, and
        # divide each sum by the count of each unique value in x
        fft_y_data = (
            np.bincount(self.x_scale, weights=fft_y_data)
            / (np.bincount(self.x_scale) + 1)
        )
        # Convert to log scale - +1 to avoid log2(0)
        output = np.log2(fft_y_data + 1)
        # Normalise and convert to 8-bit - normalisation causes variation
        # integered = (norm_scale(output)*255).astype(np.uint8)
        # Clipping within a range and scaling to 255 is better
        integered = (np.clip(output, 2, 12)-2)*(25.5)
        # Convert to integer
        integered = integered.astype(np.uint8)
        return self.length, integered


class FFTAVCapture(CombinedSource):
    """Auto populate with audio and video."""

    def __init__(self):
        """Initialise."""
        super().__init__()
        audio = FFTSource()
        self.add_source(audio, "audio")
        video = VideoSource()
        self.add_source(video, "video")
