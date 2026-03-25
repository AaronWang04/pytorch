"""Quack-backed RMSNorm overrides for aten fused RMSNorm operators.

Requires the `quack-kernels` package (https://github.com/Dao-AILab/quack)
When quack is not installed the overrides are silently skipped
"""
# mypy: allow-untyped-defs

from __future__ import annotations

import os
import functools
import importlib
import logging
from collections.abc import Callable

import torch

from ...common_utils import check_native_jit_disabled
from ...registry import _register_op_override


log = logging.getLogger(__name__)


def _quack_available() -> bool:
    try:
        # Disable quack's .o disk cache before first import — loading
        # cached objects can segfault due to a quack jit_cache bug.
        # Aaron: will try and fix this on quack side
        os.environ.setdefault("QUACK_CACHE_ENABLED", "0")
        importlib.import_module("quack.rmsnorm")
        return True
    except ModuleNotFoundError:
        return False


_RMSNormFwdFallback = Callable[
    [torch.DispatchKeySet, torch.Tensor, list[int], torch.Tensor | None, float | None],
    tuple[torch.Tensor, torch.Tensor],
]
_RMSNormBwdFallback = Callable[
    [
        torch.DispatchKeySet,
        torch.Tensor,
        torch.Tensor,
        list[int],
        torch.Tensor,
        torch.Tensor | None,
        list[bool],
    ],
    tuple[torch.Tensor | None, torch.Tensor | None],
]


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


def _support_error(
    input: torch.Tensor,
    name: str,
) -> str | None:
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input dtype must be float16, bfloat16, or float32"
    if _get_device_major(input.device) not in (9, 10):
        return f"CuTeDSL {name} requires compute capability 9.0 or 10.0"
    return None


def _fused_rms_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
    *,
    fallback_kernel: _RMSNormFwdFallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    error = _support_error(input, "RMSNorm")
    if error is not None:
        return fallback_kernel(dispatch_keys, input, normalized_shape, weight, eps)

    if eps is None:
        eps = torch.finfo(input.dtype).eps

    from .norms import quack_rmsnorm_fwd

    return quack_rmsnorm_fwd(input, weight, normalized_shape, eps)


def _fused_rms_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    normalized_shape: list[int],
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    output_mask: list[bool],
    *,
    fallback_kernel: _RMSNormBwdFallback,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    error = _support_error(input, "RMSNorm backward")
    if error is not None:
        return fallback_kernel.call_boxed(  # pyrefly: ignore[missing-attribute]
            dispatch_keys,
            grad_out,
            input,
            normalized_shape,
            rstd,
            weight,
            output_mask,
        )

    from .norms import quack_rmsnorm_bwd

    grad_input, grad_weight = quack_rmsnorm_bwd(
        grad_out, input, rstd, weight, normalized_shape,
        dw_mask=output_mask[1],
    )

    if not output_mask[0]:
        grad_input = None
    return grad_input, grad_weight


def register_rmsnorm_overrides() -> None:
    if not _quack_available():
        log.debug("quack-kernels not installed, skipping RMSNorm overrides")
        return

    if check_native_jit_disabled():
        return

    if not torch.cuda.is_available():
        return

    fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
    bwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm_backward", "CUDA")

    fwd_impl = functools.partial(
        _fused_rms_norm_impl,
        fallback_kernel=fwd_fallback,
    )
    bwd_impl = functools.partial(
        _fused_rms_norm_backward_impl,
        fallback_kernel=bwd_fallback,
    )

    _register_op_override("aten", "_fused_rms_norm", "CUDA", fwd_impl)
    _register_op_override("aten", "_fused_rms_norm_backward", "CUDA", bwd_impl)


register_rmsnorm_overrides()
