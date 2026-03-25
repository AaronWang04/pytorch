"""Adaptor for quack's 2-D RMSNorm kernel interface to match ATen op signatures.

These functions handle tensor reshaping and memory allocation.
"""

from __future__ import annotations

import importlib
import math
from functools import cache

import torch


@cache
def _quack_rmsnorm():  # type: ignore[no-untyped-def]
    return importlib.import_module("quack.rmsnorm")


def _reshape_2d(t: torch.Tensor, M: int, N: int) -> torch.Tensor:
    if t.ndim == 2 and t.shape[0] == M and t.shape[1] == N and t.is_contiguous():
        return t
    return t.reshape(M, N).contiguous()


def _flatten_rstd(t: torch.Tensor, M: int) -> torch.Tensor:
    if t.ndim == 1 and t.shape[0] == M:
        return t
    if t.is_contiguous() and t.numel() == M:
        return t.detach().view(M)
    return t.reshape(M).contiguous()


def quack_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _rmsnorm_fwd = _quack_rmsnorm()._rmsnorm_fwd

    input_shape = input.shape
    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N)

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    if weight is not None and weight.ndim != 1:
        weight = weight.view(N)
    _rmsnorm_fwd(x, weight, out, None, rstd, None, None, None, eps)

    out = out.reshape(input_shape)
    return out, rstd


def quack_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mod = _quack_rmsnorm()
    _get_sm_count, _rmsnorm_bwd = mod._get_sm_count, mod._rmsnorm_bwd

    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = _reshape_2d(input, M, N)
    dout = _reshape_2d(grad_out, M, N)
    rstd_flat = _flatten_rstd(rstd, M)

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(N, x.device)
    dw_partial: torch.Tensor | None = None
    if weight is not None:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    _rmsnorm_bwd(x, weight, dout, rstd_flat, dx, dw_partial, None, None, None, sm_count)

    dx = dx.reshape(input.shape)
    dw = (
        dw_partial.sum(dim=0, dtype=weight.dtype)  # pyrefly: ignore[missing-attribute]
        if weight is not None
        else None
    )
    return dx, dw
