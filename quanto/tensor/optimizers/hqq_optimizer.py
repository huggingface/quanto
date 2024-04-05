from typing import Optional, Tuple, Union

import torch

from ..qtype import qint2, qint4
from ..quantizers import AffineQuantizer
from .max_optimizer import MaxOptimizer


__all__ = ["HqqOptimizer"]


# Shrinking operator
def shrink_lp_op(x: torch.Tensor, beta: float, lp_norm: float) -> torch.Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1)
        )


class HqqOptimizer(MaxOptimizer):

    def __init__(
        self,
        lp_norm: Optional[float] = 0.7,
        beta: Optional[int] = 1e1,
        kappa: Optional[float] = 1.01,
        iters: Optional[int] = 20,
        verbose: Optional[bool] = False,
    ) -> None:
        self.lp_norm = lp_norm
        self.beta = beta
        self.kappa = kappa
        self.iters = iters
        self.verbose = verbose

    def optimize(
        self, base: torch.Tensor, bits: int, axis: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        scale, zeropoint = super().optimize(base, bits, axis)
        best_error = 1e4
        beta = self.beta
        qtype = qint2 if bits == 2 else qint4
        for i in range(self.iters):
            base_q = AffineQuantizer.apply(base, qtype, axis, None, scale, zeropoint)
            error = base - base_q
            e = shrink_lp_op(error, beta, self.lp_norm)
            mean_axis = 0 if axis == -1 else -1
            zeropoint = torch.mean(base_q._data - (base - e) / scale, axis=mean_axis, keepdim=True)
            zeropoint = torch.round(zeropoint).to(torch.int8)
            beta *= self.kappa

            current_error = float(torch.abs(base - base_q).mean())
            if self.verbose:
                print(f"HQQ error at it #{i}: {current_error:.6f}")
            if current_error < best_error:
                best_error = current_error
            else:
                break
        return scale, zeropoint
