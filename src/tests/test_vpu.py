"""VPU and Covar Unit Tests.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.vpu import VPU, VPUBinary, project
from src.var_processor.buffer_vpu import BufferVPU
from src.tests.vpu_wrapper import VPUWrapper, signal_pre_processor


def rand_same(size=2, negative=False):
    """Create 1D array of same binary values."""
    a = np.empty([size, 1])
    if not negative:
        # Choose value that is 0 or 1
        rand_int = np.random.randint(2)
    else:
        # Choose value that is -1, 0 or 1 with uniform distribution
        rand_int = np.random.randint(3)
    if rand_int == 0:
        a.fill(0)
    if rand_int == 1:
        a.fill(1)
    if rand_int == 2:
        a.fill(-1)
    return a


def rand_diff(size=2, negative=False):
    """Create 1D array with single 1 and rest 0."""
    a = np.zeros([size, 1])
    index = np.random.randint(size)
    if negative and np.random.randint(2):
        a[index] = -1
    else:
        a[index] = 1
    return a


def rand_opposite(size=2, negative=False):
    """Create a 1D array with opposite values."""
    # Create a random binary of size "size"
    rand_array = np.random.randint(2, size=(size, 1))
    if negative:
        rand_array = np.where(rand_array == 0, -1, 1)
    return rand_array


def test_rand():
    """Test random array generation as above."""
    # Test rand_same with negative numbers
    for i in range(0, 100):
        assert (rand_same() >= 0).all()
    neg_sum = 0
    for i in range(0, 100):
        if (rand_same(negative=True) < 0).all():
            neg_sum += 1
    # Check there are some negative values
    assert neg_sum > 0
    # Check zero mean
    rolling_sum = np.zeros(shape=(2, 1))
    for i in range(0, 1000):
        rolling_sum = rolling_sum + rand_same(negative=True)
    # Mean should be near 0
    assert np.allclose(rolling_sum/1000, np.zeros(shape=(2, 1)), atol=0.1)
    # Check opposites - for negative we should always have a sum of length
    rand_array = rand_opposite(size=2, negative=True)
    assert np.abs(rand_array).sum() == 2
    rand_array = rand_opposite(size=3, negative=True)
    assert np.abs(rand_array).sum() == 3


def test_project():
    """Test the VPU projection."""
    in_1 = np.ones(shape=(2, 1))
    in_2 = np.ones(shape=(2, 1))
    # Test vector multiplication
    out = project(in_1.T, in_2)
    assert np.array_equal(np.dot(in_1.T, in_2), out)
    assert out.shape == (1, 1)
    # Test scalar multiplication
    r = 1
    out = project(r, in_1)
    assert np.array_equal(in_1, out)


def test_vpu():
    """Test the VPU."""
    # Intialise VPU
    vpu = VPU(2)
    # Test Iteration
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(2, 1))
        r_backward = np.random.randint(2, size=(1, 1))
        vpu.update_cov(data_in)
        _ = vpu.iterate(data_in, r_backward)
    old_cov = vpu.cu.covariance
    assert old_cov.any()
    vpu.reset()
    new_cov = vpu.cu.covariance
    assert not np.array_equal(old_cov, new_cov)


def test_buffer_vpu():
    """Test the BufferVPU."""
    # Intialise VPU
    vpu = BufferVPU(2, 4)
    assert vpu.forward_buffer.shape == (2, 4)
    assert vpu.backward_buffer.shape == (1, 4)
    assert vpu.cu.covariance.shape == (8, 8)
    assert vpu.pi.ev.shape == (8, 1)
    # Test Iteration
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(2, 1))
        r_backward = np.random.randint(2, size=(1, 1))
        vpu.update_cov(data_in)
        _ = vpu.iterate(data_in, r_backward)
    old_cov = vpu.cu.covariance
    assert old_cov.any()
    vpu.reset()
    new_cov = vpu.cu.covariance
    assert not np.array_equal(old_cov, new_cov)


def test_vpu_binary():
    """Test the VPU with non linearity."""
    # Intialise VPU
    vpu = VPUBinary(2)
    # Test Iteration
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(2, 1))
        r_backward = np.random.randint(2, size=(1, 1))
        vpu.update_cov(data_in)
        cause, residual = vpu.iterate(data_in, r_backward)
    old_cov = vpu.cu.covariance
    assert old_cov.any()
    assert cause == 0 or cause == 1
    assert residual.shape == (2, 1)
    vpu.reset()
    new_cov = vpu.cu.covariance
    assert not np.array_equal(old_cov, new_cov)


def test_vpu_function_same():
    """More advanced testing of VPU function."""
    vpu = VPU(2)
    # Test with binary arrays of [0, 0] and [1, 1]
    for _ in range(0, 1000):
        vpu.update_cov(rand_same())
        vpu.pi.iterate(cov=vpu.cu.covariance)
    # Check all values of covariance matrix are the same
    assert np.allclose(vpu.cu.covariance, vpu.cu.covariance[0])
    # Check eigenvector has values of root 2
    assert np.allclose(vpu.pi.eigenvector, 1/np.sqrt(2))
    sample_1 = np.dot(
        vpu.pi.eigenvector, np.sqrt(vpu.pi.eigenvalue))+vpu.cu.mean
    sample_minus1 = -1*np.dot(
        vpu.pi.eigenvector, np.sqrt(vpu.pi.eigenvalue))+vpu.cu.mean
    assert np.allclose(sample_1, np.array([1, 1]), rtol=0.05, atol=0.05)
    assert np.allclose(
        sample_minus1, np.array([0, 0]), rtol=0.05, atol=0.05)


def test_vpu_function_diff():
    """Test random different 1D length 2 binary arrays."""
    vpu = VPU(2)
    for _ in range(0, 1000):
        vpu.update_cov(rand_diff())
        vpu.pi.iterate(cov=vpu.cu.covariance)
    # Check diagonal values of covariance matrix are the same
    # Use https://docs.scipy.org/doc/numpy/reference/
    # generated/numpy.diagonal.html
    print(vpu.cu.covariance[0], -1*vpu.cu.covariance[-1])
    assert np.allclose(vpu.cu.covariance[0], -1*vpu.cu.covariance[-1])
    # Check eigenvector has values of root 2
    print(vpu.pi.eigenvector, 1/np.sqrt(2))
    assert np.allclose(np.abs(vpu.pi.eigenvector), 1/np.sqrt(2))
    # Check different signs
    assert np.allclose(vpu.pi.eigenvector[0], -1*vpu.pi.eigenvector[1])
    sample_1 = np.dot(
        vpu.pi.eigenvector, np.sqrt(vpu.pi.eigenvalue))+vpu.cu.mean
    sample_minus1 = -1*np.dot(
        vpu.pi.eigenvector, np.sqrt(vpu.pi.eigenvalue))+vpu.cu.mean
    print(sample_1, np.flipud(sample_minus1))
    assert np.allclose(sample_1, np.flipud(sample_minus1), rtol=0.1, atol=0.1)


def test_recontruction():
    """Use the VPU Wrapper to test advanced function."""
    # Initialise two VPUs and wrappers
    data_in = np.random.randint(255, size=(2, 1))
    mean = np.asarray([128, 128]).reshape(-1, 1)
    vpu_1 = VPU(2)
    vpu_2 = VPU(2)
    wrapper_1 = VPUWrapper(vpu_1)
    wrapper_2 = VPUWrapper(vpu_2)
    for _ in range(0, 1000):
        # First VPU
        ternary_input = signal_pre_processor(data_in, mean)
        _, _, residual = wrapper_1.iterate(ternary_input)
        # Second VPU
        _ = wrapper_2.iterate(residual)
    est = (wrapper_1.pred_estimate*mean+mean)+(wrapper_2.pred_estimate*mean)
    assert np.allclose(data_in, est, rtol=0.10, atol=10)
