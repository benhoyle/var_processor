"""Test Video Functions."""
import time
import numpy as np

from src.sources.video import separate_components, VideoSource


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
