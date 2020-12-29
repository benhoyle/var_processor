"""Test Threshold Functions."""

import numpy as np
from src.var_processor.pb_threshold import ternary_pbt, pb_residual_threshold


def helper_for_pbt(shape=(4, 1)):
    """Helper function for PBT test below."""
    # 1D case
    max_value = 127
    one_d_array = np.ones(shape=shape, dtype=np.int8) * max_value
    output = ternary_pbt(one_d_array, max_value)
    # print(output, one_d_array//max_value)
    assert np.array_equal(output, one_d_array // max_value)
    # Test halving the input - we should get a different output
    one_d_array_2 = one_d_array // 2
    rolling_sum = np.zeros(shape=shape, dtype=np.int8)
    for _ in range(0, max_value):
        output_2 = ternary_pbt(one_d_array_2, max_value)
        rolling_sum += output_2
    # print(rolling_sum, one_d_array*(max_value//2),)
    # Check the sum is close to the original halved array
    assert np.allclose(rolling_sum, one_d_array * (max_value // 2), atol=20)
    # Repeat for negative values
    max_value = -127
    one_d_array = np.ones(shape=shape, dtype=np.int8) * max_value
    output = ternary_pbt(one_d_array, np.abs(max_value))
    # print(output, one_d_array//np.abs(max_value))
    assert np.array_equal(output, one_d_array // np.abs(max_value))
    # Test halving the input - we should get a different output
    one_d_array_2 = one_d_array // 2
    rolling_sum = np.zeros(shape=shape, dtype=np.int8)
    for _ in range(0, np.abs(max_value)):
        output_2 = ternary_pbt(one_d_array_2, np.abs(max_value))
        rolling_sum += output_2
    # print(rolling_sum, one_d_array*(max_value//2),)
    # Check the sum is close to the original halved array
    assert np.allclose(rolling_sum, one_d_array * (max_value // 2), atol=20)
    # Test with mixture of 1s and -1s and 0s
    random_array = np.random.randint(low=-127, high=127, size=shape)
    rolling_sum = np.zeros(shape=shape, dtype=np.int8)
    for _ in range(0, 127):
        output_3 = ternary_pbt(random_array, 127)
        rolling_sum += output_3
    # print(rolling_sum, random_array)
    # Check the sum is close to the original halved array
    assert np.allclose(rolling_sum, random_array, atol=20)


def test_ternary_ppt():
    """Test ternary PBT."""
    helper_for_pbt()
    helper_for_pbt(shape=(4, 4))
    helper_for_pbt(shape=(6, 2))


def test_pb_residual_threshold():
    # Get an array from -4 to 4
    test_residuals = np.arange(-4, 5, dtype=np.int8)
    # Repeat 100 times to see distribution
    binary_values = np.zeros(shape=test_residuals.shape)
    for i in range(100):
        binary_values += pb_residual_threshold(test_residuals)
    # Check middle is 0
    assert binary_values[4] == 0
    # Check minimum value
    # noinspection PyArgumentList
    assert binary_values.min() < -95
    # Check maximum value
    # noinspection PyArgumentList
    assert binary_values.max() > 95
    # Check monotonic
    assert binary_values[3] > binary_values[2]
    assert binary_values[6] > binary_values[5]
    # print(binary_values)
