from typing import Tuple
import torch
from .symmetric_optimizer import SymmetricOptimizer
from .affine_optimizer import AffineOptimizer
from torch import Tensor
from quanto.tensor.core import axis_to_dim


class MseSymmetricOptimizer(SymmetricOptimizer):
    def __init__(self, p=2.0, iters=80) -> None:
        self.iters = iters
        self.p = p
        assert p >= 1.0
        assert 0 < iters < 100

    def optimize(self, base: torch.Tensor, bits: int, axis: int):
        dim = None if axis is None else axis_to_dim(base, axis)
        rmax = torch.amax(torch.abs(base), dim=dim, keepdim=True)
        qmax = 2**bits - 1
        best_score = float("inf")
        best_rmax = rmax
        for i in range(self.iters):
            new_rmax = best_rmax * (1 - (i * 0.001))
            scale = new_rmax / qmax
            if axis is None:
                fq_base = torch.fake_quantize_per_tensor_affine(base, scale.item(), 0, -qmax, qmax)
            else:
                fq_base = torch.fake_quantize_per_channel_affine(
                    base, scale, torch.zeros_like(scale), axis, -qmax, qmax
                )
            score = (fq_base - base).abs().pow(self.p).mean(dim) if axis else (fq_base - base).abs().pow(self.p).mean()
            if score.item() < best_score:
                best_score = score.item()
                best_rmax = new_rmax
        scale = best_rmax / qmax
        return scale


class MseAffineOptimizer(AffineOptimizer):
    def __init__(self, p=2.0, iters=80) -> None:
        self.iters = iters
        self.p = p
        assert p >= 1.0
        assert 0 < iters < 100

    def optimize(self, base: Tensor, bits: int, axis: int) -> Tuple[Tensor, Tensor]:
        best_score = float("inf")
        dim = None if axis is None else axis_to_dim(base, axis)
        rmin, rmax = torch.aminmax(base, dim=dim, keepdim=True)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        best_rmin, best_rmax = rmin, rmax
        for i in range(self.iters):
            new_rmin = best_rmin * (1 - (i * 0.001))
            new_rmax = best_rmax * (1 - (i * 0.001))
            scale = (new_rmax - new_rmin) / (qmax - qmin)
            zeropoint = torch.round(-rmin / scale).to(torch.int8)
            if axis is None:
                fq_base = torch.fake_quantize_per_tensor_affine(base, scale.item(), zeropoint.item(), qmin, qmax)
            else:
                fq_base = torch.fake_quantize_per_channel_affine(base, scale, zeropoint, axis, qmin, qmax)
            score = (fq_base - base).abs().pow(self.p).mean(dim) if axis else (fq_base - base).abs().pow(self.p).mean()
            if score.item() < best_score:
                best_score = score.item()
                best_rmin, best_rmax = new_rmin, new_rmax
        scale = (best_rmax - best_rmin) / (qmax - qmin)
        zeropoint = torch.round(-best_rmin / scale).to(torch.int8)
        return scale, zeropoint
