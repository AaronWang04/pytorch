"""CuteDSL BatchNorm adapter.

Adapts the CuTE DSL kernels to the ATen ``native_batch_norm`` signatures.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cutedsl_batchnorm_fwd(
    input: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    from ._batchnorm_kernels import _batchnorm_eval, _batchnorm_fwd

    C = input.size(1)
    S = input.numel() // (input.size(0) * C)
    x = input.reshape(input.size(0), C, S)
    out = torch.empty_like(input).reshape_as(x)
    save_mean = torch.empty(C, device=input.device, dtype=torch.float32)
    save_invstd = torch.empty(C, device=input.device, dtype=torch.float32)

    if training:
        _batchnorm_fwd(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            out,
            save_mean,
            save_invstd,
            momentum,
            eps,
        )
    else:
        assert running_mean is not None
        assert running_var is not None
        _batchnorm_eval(
            x, weight, bias, running_mean, running_var, out, save_mean, save_invstd, eps
        )

    return out.view_as(input), save_mean, save_invstd


def cutedsl_batchnorm_bwd(
    grad_out: Tensor,
    input: Tensor,
    weight: Tensor | None,
    save_mean: Tensor,
    save_invstd: Tensor,
    output_mask: list[bool],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    from ._batchnorm_kernels import _batchnorm_bwd

    C = input.size(1)
    S = input.numel() // (input.size(0) * C)
    x = input.reshape(input.size(0), C, S)
    dout = grad_out.reshape_as(x)

    dx = torch.empty_like(x) if output_mask[0] else None
    dw = (
        torch.empty(C, device=input.device, dtype=weight.dtype)
        if output_mask[1] and weight is not None
        else None
    )
    db = (
        torch.empty(C, device=input.device, dtype=weight.dtype)
        if output_mask[2] and weight is not None
        else None
    )

    _batchnorm_bwd(x, weight, dout, save_mean, save_invstd, dx, dw, db)

    return dx.view_as(input) if dx is not None else None, dw, db
