"""Test Video Functions."""
import time
import numpy as np

from src.sources.video import (
    flatten_frame, reconstruct_frame, separate_components, VideoSource
)

def test_separate_components():
    """Test separate components method."""
    test_frame = np.zeros(shape=(4, 4, 2))
    test_frame[..., 0] = 1
    test_frame[:, 0::2, 1] = 3
    test_frame[:, 1::2, 1] = 2
    one, two, three = separate_components(test_frame, square=False)
    assert (one == 1).all()
    assert (two == 2).all()
    assert (three == 3).all()
    one, two, three = separate_components(test_frame, square=True)
    assert two.shape[0] == one.shape[0]//2
    assert (one == 1).all()
    assert (two == 2).all()
    assert (three == 3).all()


def test_flatten_reconstruct():
    # Build test array
    test = np.arange(64).reshape(8,8)
    # Test flattening in blocks of 4x4
    flat4 = flatten_frame(test, 4)
    assert flat4.shape[0] == 64
    assert len(flat4.shape) == 1
    assert np.array_equal(flat4[0:4], np.arange(4))
    assert np.array_equal(flat4[4:8], np.asarray([8,  9,  10,  11]))
    # Test flattening in blocks of 2x2
    flat2 = flatten_frame(test, 2)
    assert flat2.shape[0] == 64
    assert len(flat2.shape) == 1
    assert np.array_equal(flat2[0:4], np.asarray([0,  1,  8,  9]))
    assert np.array_equal(flat2[4:8], np.asarray([2,  3, 10, 11]))
    # Test reconstruction
    reconstructed4 = reconstruct_frame(flat4, 4, 8, 8)
    assert np.array_equal(reconstructed4, test)
    reconstructed2 = reconstruct_frame(flat2, 2, 8, 8)
    assert np.array_equal(reconstructed2, test)


def test_video_source():
    """Test video source."""
    # Test initialisation
    with VideoSource() as video:
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
