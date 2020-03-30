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
    # Test getting data
    data = sensor.get_frame()
    assert data.shape[0] == 3**10
    assert data.max() <= 1


def test_stage():
    """Test stage object."""
    # Test iterating
    """
    sensor.iterate()
    causes = sensor.get_causes()
    pred_inputs = sensor.get_pred_inputs()
    data_len = sensor.get_data_length()
    assert data_len == sensor.power_len
    c_lens, p_lens = sensor.get_lengths()
    assert c_lens[0] == causes[0].shape[0]
    assert p_lens[0] == pred_inputs[0].shape[0]
    sensor.stop()
    _ = sensor.get_frame()
    sensor.stop()
    """
    pass


def test_sensor_vis():
    """Test sensor visualisation."""
    audio = FFTSource()
    sensor = Sensor(audio, 4, 4)
    sensor.source.stop()
    sen_vis = SensorVisualizer(sensor)
    sen_vis.update(None)
    sen_vis.show()
    sensor.source.stop()
