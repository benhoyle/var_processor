"""VPU and Covar Unit Tests.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.sources.capture import (
    AudioSource
)
from src.var_processor.sensor import Sensor, resize
from src.sources.fft import FFTSource
from src.visualisers.sensor import SensorVisualizer


def test_resize():
    """Test array resize function."""
    in_1 = np.linspace(0, 10-1, 10)
    expected_out = np.linspace(0, 10-1, 5).reshape(-1, 1)
    assert np.array_equal(
        expected_out,
        resize(in_1, 5)
    )


def test_sensor():
    """Test sensor object."""
    # Test Initialise
    sensor = Sensor(AudioSource(), 3, 3)
    assert len(sensor.stages) == 10
    # Test getting data
    data = sensor.get_frame()
    assert data.shape[0] == 3**10
    # Test iterating
    sensor.iterate()
    causes = sensor.get_causes()
    residuals = sensor.get_residuals()
    data_len = sensor.get_data_length()
    assert data_len == sensor.power_len
    c_lens, r_lens = sensor.get_lengths()
    assert c_lens[0] == causes[0].shape[0]
    assert r_lens[0] == residuals[0].shape[0]
    sensor.stop()
    _ = sensor.get_frame()
    sensor.stop()


def test_sensor_vis():
    """Test sensor visualisation."""
    audio = FFTSource()
    sensor = Sensor(audio, 4, 4)
    sensor.source.stop()
    sen_vis = SensorVisualizer(sensor)
    sen_vis.update(None)
    sen_vis.show()
    sensor.source.stop()

