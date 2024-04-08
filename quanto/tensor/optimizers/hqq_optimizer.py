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


def optimize_weights_proximal_legacy(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    min_max: list,
    axis: int = 0,
    device: str = "cuda",
    opt_params: dict = {"lp_norm": 0.7, "beta": 1e1, "kappa": 1.01, "iters": 20},
    verbose: bool = False,
) -> tuple:
    lp_norm, beta, kappa, iters = (
        opt_params["lp_norm"],
        opt_params["beta"],
        opt_params["kappa"],
        opt_params["iters"],
    )

    device = torch.device(device)
    dtype = torch.float16 if (device.type == "cuda") else torch.float32
    W_f = tensor.to(dtype).to(device)
    scale = scale.to(dtype).to(device)
    zero = zero.to(dtype).to(device)

    best_error = 1e4
    for i in range(iters):
        W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
        W_r = (W_q - zero) / scale
        W_e = shrink_lp_op(W_f - W_r, beta, lp_norm)
        zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
        beta *= kappa

        current_error = float(torch.abs(W_f - W_r).mean())
        if verbose:
            print(f"{i} {current_error:.6f}")
        if current_error < best_error:
            best_error = current_error
        else:
            break

    scale = scale.to(tensor.device)
    zero = zero.to(tensor.device)
    del W_f, W_q, W_r, W_e

    return scale, zero


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
        _, zeropoint = optimize_weights_proximal_legacy(
            base, 1 / scale, zeropoint, min_max=[0, 2**bits - 1], device="cpu", verbose=True
        )
        zeropoint = torch.round(zeropoint).to(torch.int8)
        return scale, zeropoint
        best_error = 1e4
        beta = self.beta
        qtype = qint2 if bits == 2 else qint4
        for i in range(self.iters):
            base_q = AffineQuantizer.apply(base, qtype, axis, None, scale, zeropoint)
            error = base - base_q
            e = shrink_lp_op(error, beta, self.lp_norm)
            mean_axis = 0 if axis == -1 else -1
            zeropoint = torch.mean(base_q._data - (base - e) / scale, axis=mean_axis, keepdim=True)
            beta *= self.kappa

            current_error = float(torch.abs(base - base_q).mean())
            if self.verbose:
                print(f"HQQ error at it #{i}: {current_error:.6f}")
            if current_error < best_error:
                best_error = current_error
            else:
                break
        zeropoint = torch.round(zeropoint).to(torch.int8)
        return scale, zeropoint
