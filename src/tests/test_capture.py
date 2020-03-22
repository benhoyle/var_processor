"""Test routines for sensor wrappers.

Run: pytest --cov=src
"""
import time
import numpy as np
from src.sources.capture import (
    SensorSource, VideoSource, AudioSource, CombinedSource,
    AVCapture
)


def test_sensor_source():
    """Test the abstract sensor source object."""
    # Test initialisation
    source = SensorSource()
    assert not source.started
    assert source.thread is None
    # Test starting thread
    returned = source.start()
    assert source.started
    assert source.thread is not None
    assert returned is not None
    # Test no start duplication
    returned = source.start()
    assert source.started
    assert source.thread is not None
    assert returned is None
    # Test pass through methods
    source.update()
    source.read()
    # Test stopping
    source.stop()
    assert not source.started


def test_video_source():
    """Test video source."""
    # Test initialisation
    video = VideoSource()
    assert video.src == 0
    # Test read
    grabbed1, frame1 = video.read()
    assert grabbed1
    assert frame1.any()
    # Check starting and getting a frame
    video.start()
    time.sleep(0.25)
    grabbed2, frame2 = video.read()
    assert grabbed2
    assert not np.array_equal(frame1, frame2)
    # Test stopping
    video.stop()
    assert not video.started


def test_audio_source():
    """Test audio source."""
    # Test initialisation
    audio = AudioSource()
    audio.start()
    time.sleep(0.25)
    # Test read
    length1, samples1 = audio.read()
    assert length1
    assert samples1.any()
    # Check starting and getting a frame
    audio.start()
    time.sleep(0.25)
    length2, samples2 = audio.read()
    assert length2
    assert not np.array_equal(samples1, samples2)
    assert samples2.any()
    # Test stopping
    audio.stop()
    assert not audio.started


def test_combined_source():
    """Test abstract combined source object."""
    combined = CombinedSource()
    assert isinstance(combined.sources, dict)
    assert not combined.sources
    # Adding a source
    combined.add_source(SensorSource())
    assert len(combined.sources) == 1
    assert list(combined.sources.items())[0][0] == "SensorSource"
    assert list(combined.sources.items())[0][1]
    combined.add_source(SensorSource(), "SS2")
    assert len(combined.sources) == 2
    assert list(combined.sources.items())[1][0] == "SS2"
    assert list(combined.sources.items())[1][1]
    # Test starting and stopping
    combined.start()
    assert list(combined.sources.items())[0][1].started
    assert list(combined.sources.items())[1][1].started
    combined.stop()
    assert not list(combined.sources.items())[0][1].started
    assert not list(combined.sources.items())[1][1].started


def test_av_source():
    """Test combined AV Source."""
    av = AVCapture()
    av.start()
    time.sleep(0.25)
    data = av.read()
    assert data['audio'].any()
    assert data['video'].any()
    av.stop()
