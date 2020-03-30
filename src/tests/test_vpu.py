"""VPU and Covar Unit Tests.
Run: pytest --cov=src --cov-report term-missing
"""
import numpy as np
from src.var_processor.vpu import (
    VPU, project, reconstruct, VPUNonLin, BufferVPU
)
from src.var_processor.time_stage import TimeStage


def test_project():
    """Test the VPU projection."""
    in_1 = np.ones(shape=(2, 1))
    in_2 = np.ones(shape=(2, 1))
    out = project(in_1, in_2)
    assert np.array_equal(np.dot(in_1.T, in_2), out)
    assert out.shape == (1, 1)


def test_reconstruct():
    """Test the VPU reconstruction."""
    assert reconstruct(2, 2) == 4
    in_1 = np.ones(shape=(2, 1))
    assert np.array_equal(reconstruct(in_1, 1), in_1)


def test_vpu():
    """Test the VPU."""
    # Intialise VPU
    vpu = VPU(2)
    # Test Iteration
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(2, 1))
        _ = vpu.iterate(data_in)
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
        _ = vpu.iterate(data_in, r_backward)
    old_cov = vpu.cu.covariance
    assert old_cov.any()
    vpu.reset()
    new_cov = vpu.cu.covariance
    assert not np.array_equal(old_cov, new_cov)


def test_vpu_nonlin():
    """Test the VPU with non linearity."""
    # Intialise VPU
    vpu = VPUNonLin(2, 4)
    # Test Iteration
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(2, 1))
        r_backward = np.random.randint(2, size=(1, 1))
        cause, residual = vpu.iterate(data_in, r_backward)
    old_cov = vpu.cu.covariance
    assert old_cov.any()
    assert cause == 0 or cause == 1
    assert residual.shape == (2, 1)
    vpu.reset()
    new_cov = vpu.cu.covariance
    assert not np.array_equal(old_cov, new_cov)


def test_time_stage():
    """Test time stage."""
    # Initialise time stage
    stages = TimeStage(3, 10)
    assert len(stages.vpus) == 10
    assert not stages.causes.any()
    assert not stages.pred_inputs.any()
    assert "10" in stages.__repr__()
    # Check data in
    for _ in range(0, 10):
        data_in = np.random.randint(2, size=(stages.size, 1))
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes1, pred_inputs1 = stages.iterate(data_in, r_backwards)
    assert stages.causes.any()
    assert stages.pred_inputs.any()
    for _ in range(0, 100):
        data_in = np.random.randint(2, size=(stages.size, 1))
        r_backwards = np.random.randint(2, size=(stages.stage_len, 1))
        causes2, pred_inputs2 = stages.iterate(data_in, r_backwards)
    assert not np.array_equal(causes1, causes2)
    assert not np.array_equal(pred_inputs1, pred_inputs2)
    # Check different sizes
    data_in = np.random.randint(2, size=(stages.size+10, 1))
    _ = stages.iterate(data_in, r_backwards)
    data_in = np.random.randint(2, size=(stages.size-10, 1))
    _ = stages.iterate(data_in, r_backwards)
