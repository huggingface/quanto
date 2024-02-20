import functools

import pytest
import torch
from packaging import version

from quanto import QBitsTensor, QTensor, absmax_scale, qint4, qint8


def torch_min_version(v):
    def torch_min_version_decorator(test):
        @functools.wraps(test)
        def test_wrapper(*args, **kwargs):
            if version.parse(torch.__version__) < version.parse(v):
                pytest.skip(f"Requires pytorch >= {v}")
            test(*args, **kwargs)

        return test_wrapper

    return torch_min_version_decorator


def device_eq(a, b):
    if a.type != b.type:
        return False
    a_index = a.index if a.index is not None else 0
    b_index = b.index if b.index is not None else 0
    return a_index == b_index


def random_tensor(shape, dtype=torch.float32):
    # Return a random tensor between -1. and 1.
    return torch.rand(shape, dtype=dtype) * 2 - 1


def random_qtensor(shape, qtype=qint8, dtype=torch.float32, axis=None):
    t = random_tensor(shape, dtype)
    scale = absmax_scale(t, qtype=qtype, axis=axis)
    return QTensor.quantize(t, qtype=qtype, scale=scale)


def random_qbitstensor(shape, qtype=qint4, dtype=torch.float32, axis=None):
    t = random_tensor(shape, dtype)
    return QBitsTensor.quantize(t, qtype=qtype, axis=axis)


def assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = None, rtol: float = None):
    # Please refer to that discussion for default rtol values based on the float type:
    # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
    assert a.dtype == b.dtype
    if atol is None:
        # We use torch finfo resolution
        atol = torch.finfo(a.dtype).resolution
    atol = torch.tensor(atol)
    # We cannot expect an absolute error lower than the quantization scales
    if isinstance(a, QTensor):
        atol = torch.maximum(a._scale, atol)
        a = a.dequantize()
    if isinstance(b, QTensor):
        atol = torch.maximum(b._scale, atol)
        b = b.dequantize()
    if rtol is None:
        # Please refer to that discussion for default rtol values based on the float type:
        # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
        rtol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-1}[a.dtype]
    abs_error = torch.abs(a - b)
    closeness = atol + rtol * torch.abs(a)
    if not torch.all(abs_error <= closeness):
        max_error_index = torch.argmax(abs_error - closeness)
        max_error_a = a.flatten()[max_error_index]
        max_rel_error = abs_error.flatten()[max_error_index] / torch.abs(max_error_a) * 100
        max_error_b = b.flatten()[max_error_index]
        raise ValueError(
            f"Error exceeds tolerance (max: {max_error_b:.8f} instead of {max_error_a:.8f} ({max_rel_error:.4f} %)."
        )


def assert_similar(a, b, atol=1e-6, rtol=1e-5):
    """Verify that the cosine similarity of the two inputs is close to 1.0 everywhere"""
    assert a.shape == b.shape
    sim = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    if not torch.allclose(sim, torch.tensor(1.0, dtype=sim.dtype), atol=atol, rtol=rtol):
        max_deviation = torch.min(sim)
        raise ValueError(f"Alignment {max_deviation:.8f} deviates too much from 1.0 with atol={atol}, rtol={rtol}")
