"""VPU and Covar Unit Tests.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.sources.capture import AudioSource
from src.var_processor.sensor import Sensor, resize, signal_adjust
from src.sources.fft import FFTSource
from src.visualisers.vis_sensor import SensorVisualizer


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
    sensor = Sensor(AudioSource(), 3, 3, m_batch=10)
    # Check mean is None
    assert not sensor.mean.all()
    # Test getting data
    data = sensor.get_frame()
    mean_1 = sensor.mean
    assert mean_1 is not None
    assert data.shape[0] == 3**10
    assert data.max() <= 1
    # Test mean computation
    for _ in range(0, 25):
        data = sensor.get_frame()
    mean_2 = sensor.mean
    assert not np.array_equal(mean_1, mean_2)
    sensor.source.stop()


def test_signal_adjust():
    """Test signal adjustment."""
    # Test both +ve and -ve values
    data_in = np.asarray([220, 60])
    signal_mean = np.asarray([128, 128])
    # Repeat a number of times
    summed = signal_adjust(data_in, signal_mean)
    for _ in range(0, 3467):
        summed += signal_adjust(data_in, signal_mean)
    reconstructed = (summed / (3467 / signal_mean))+signal_mean
    assert np.allclose(data_in, reconstructed, rtol=0.1, atol=10)


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
    # sen_vis = SensorVisualizer(sensor)
    # sen_vis.update(None)
    # sen_vis.show()
    # sensor.source.stop()
