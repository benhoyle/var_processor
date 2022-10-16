"""Tests for video helper functions."""

import numpy as np
from src.sources.video_helper import create_pyramid, upsample,reduce_to_2


def test_create_pyramid():
    ones = np.ones(shape=(256, 256), dtype=np.uint8)
    pyramid = create_pyramid(ones, reduce=None)
    assert len(pyramid) == 9
    for i, level in enumerate(pyramid):
        assert level.shape[0] == 2 ** (8 - i)
        assert level.shape[0] == 2 ** (8 - i)
    ones = np.ones(shape=(256, 1))
    pyramid = create_pyramid(ones, reduce=None)
    assert len(pyramid) == 9
    for i, level in enumerate(pyramid):
        assert level.shape[0] == 2 ** (8 - i)
        assert level.shape[1] == 1


def test_upsample():
    ones = np.ones(shape=(256, 256), dtype=np.uint8)
    down_pyramid = create_pyramid(ones, reduce=None)
    up_pyramid = upsample(down_pyramid)
    assert len(down_pyramid) == len(up_pyramid)
    for d, u in zip(down_pyramid, reversed(up_pyramid)):
        assert d.shape == u.shape
    ones = np.ones(shape=(256, 1))
    pyramid = create_pyramid(ones, reduce=None)
    up_pyramid = upsample(down_pyramid)
    assert len(down_pyramid) == len(up_pyramid)
    for d, u in zip(down_pyramid, reversed(up_pyramid)):
        assert d.shape == u.shape


def test_reduce_to_2():
    test_array = np.ones(shape=(5, 7))
    reduced = reduce_to_2(test_array)
    assert reduced.shape == (4, 4)
