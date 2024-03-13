import os

import torch
from torch.library import impl
from torch.utils.cpp_extension import load


__all__ = []


_ext = None


def ext():
    """Helper to load the MPS extension only when it is required"""
    global _ext
    if _ext is None:
        module_path = os.path.dirname(__file__)
        _ext = load(
            name="quanto_mps",
            sources=[
                f"{module_path}/unpack.mm", 
                f"{module_path}/udqmm.cpp",
                f"{module_path}/pybind_module.cpp"
            ],
            extra_cflags=["-std=c++17"],
        )
    return _ext


@impl("quanto_ext::unpack", "MPS")
def unpack_mps(t: torch.Tensor, bits: int):
    return ext().unpack(t, bits)


@impl("quanto_ext::udqmm", "MPS")
def udqmm_cpp(input: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, bits: int):
    return ext().udqmm(input, weight, scales, bits)
