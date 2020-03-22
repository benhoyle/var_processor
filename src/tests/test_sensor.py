"""VPU and Covar Unit Tests.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.sources.capture import (
    AudioSource
)
from src.var_processor.sensor import Sensor, resize


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
    sensor.stop()
    sdata = sensor.get_frame()
    sensor.stop()
