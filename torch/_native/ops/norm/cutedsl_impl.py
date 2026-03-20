"""CuTeDSL RMSNorm overrides for aten fused RMSNorm operators.
print("\nTrace exported to 'trace.json'. Open chrome://tracing/ in your browser and load this file.")

Registers flat Python impls directly on the aten::_fused_rms_norm and
aten::_fused_rms_norm_backward CUDA dispatch keys. The compiled TVM FFI
kernels are called directly from these impls, bypassing the
@torch.library.custom_op layer to reduce CPU dispatch overhead.
"""

# mypy: allow-untyped-defs

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable

import torch

from ... import cutedsl_utils as cu


log = logging.getLogger(__name__)

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


def _collect_tensors(*tensors: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    return tuple(t for t in tensors if t is not None)


def _support_error(
    input: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
    name: str,
) -> str | None:
    if not all(t.is_cuda for t in tensors):
        return "inputs must be CUDA tensors"
    if len({t.device for t in tensors}) != 1:
        return "inputs must share device"
    if input.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return "input dtype must be float16, bfloat16, or float32"
    if not torch.cuda.is_available():
        return "CUDA not available"
    if _get_device_major(input.device) not in (9, 10):
        return f"CuTeDSL {name} requires compute capability 9.0 or 10.0"
    return None


def _stat_shape(input: torch.Tensor, n_norm: int) -> tuple[int, ...]:
    return input.shape[: input.dim() - n_norm] + (1,) * n_norm


def _get_compiled_fwd(dtype, weight_dtype, N):
    """JIT-compile and cache the forward RMSNorm kernel for the given config."""
    from ._rmsnorm_kernels import (
        _TORCH2CUTE_DTYPE,
        RMSNorm,
    )
    from ._cute_utils import make_fake_tensor as fake_tensor

    import cutlass.cute as cute
    from cutlass import Float32

    cute_dtype = _TORCH2CUTE_DTYPE[dtype]
    cute_weight_dtype = _TORCH2CUTE_DTYPE[weight_dtype] if weight_dtype is not None else None
    div = math.gcd(N, *(
        128 // dt.width
        for dt in [cute_dtype, cute_dtype, cute_weight_dtype]
        if dt is not None
    ))
    batch_sym = cute.sym_int()
    x_cute = fake_tensor(cute_dtype, (batch_sym, N), div)
    out_cute = fake_tensor(cute_dtype, (batch_sym, N), div)
    weight_cute = fake_tensor(cute_weight_dtype, (N,), div)
    rstd_cute = fake_tensor(Float32, (batch_sym,))
    return cute.compile(
        RMSNorm(cute_dtype, N),
        x_cute,
        weight_cute,
        None,  # bias
        None,  # residual
        out_cute,
        None,  # residual_out
        rstd_cute,
        Float32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _get_compiled_bwd(dtype, weight_dtype, N):
    """JIT-compile and cache the backward RMSNorm kernel for the given config."""
    from ._rmsnorm_kernels import (
        _TORCH2CUTE_DTYPE,
        RMSNormBackward,
    )
    from ._cute_utils import make_fake_tensor as fake_tensor

    import cutlass.cute as cute
    from cutlass import Float32

    cute_dtype = _TORCH2CUTE_DTYPE[dtype]
    cute_weight_dtype = _TORCH2CUTE_DTYPE[weight_dtype] if weight_dtype is not None else None
    div = math.gcd(N, *(
        128 // dt.width
        for dt in [cute_dtype, cute_dtype, cute_dtype]
        if dt is not None
    ))
    batch_sym = cute.sym_int()
    batch_partial_sym = cute.sym_int()
    x_cute = fake_tensor(cute_dtype, (batch_sym, N), div)
    dout_cute = fake_tensor(cute_dtype, (batch_sym, N), div)
    dx_cute = fake_tensor(cute_dtype, (batch_sym, N), div)
    weight_cute = fake_tensor(cute_weight_dtype, (N,), div)
    rstd_cute = fake_tensor(Float32, (batch_sym,))
    dw_partial_cute = (
        fake_tensor(Float32, (batch_partial_sym, N), div)
        if cute_weight_dtype is not None
        else None
    )
    return cute.compile(
        RMSNormBackward(cute_dtype, N),
        x_cute,
        weight_cute,
        dout_cute,
        None,  # dresidual_out
        rstd_cute,
        dx_cute,
        dw_partial_cute,
        None,  # dresidual
        None,  # db_partial
        0,  # sm_count (symbolic)
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


_fwd_compile_cache: dict = {}
_bwd_compile_cache: dict = {}


def _cutedsl_fused_rms_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None,
    eps: float | None,
    *,
    fallback_kernel: _RMSNormFwdFallback,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_norm = len(normalized_shape)
    N = normalized_shape[0] if n_norm == 1 else math.prod(normalized_shape)
    M = input.numel() // N
    needs_reshape = input.dim() != 2 or input.shape[0] != M
    x = input.reshape(M, N) if needs_reshape else input

    w = weight if weight is not None and weight.dim() == 1 else (
        weight.reshape(N) if weight is not None else None
    )
    key = (input.dtype, weight.dtype if weight is not None else None, N)
    compiled = _fwd_compile_cache.get(key)
    if compiled is None:
        compiled = _get_compiled_fwd(*key)
        _fwd_compile_cache[key] = compiled

    if eps is None:
        eps = 1e-5

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=input.device, dtype=torch.float32)
    compiled(x, w, None, None, out, None, rstd, eps)

    if needs_reshape:
        out = out.reshape(input.shape)
    if n_norm == 1 and input.dim() == 2:
        rstd = rstd.unsqueeze(1)
    else:
        rstd = rstd.view(_stat_shape(input, n_norm))
    return out, rstd


def _cutedsl_fused_rms_norm_backward_impl(
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
    error = _support_error(
        input, _collect_tensors(grad_out, input, rstd, weight), "RMSNorm"
    )
    if error is not None:
        return fallback_kernel(
            dispatch_keys,
            grad_out,
            input,
            normalized_shape,
            rstd,
            weight,
            output_mask,
        )

    from ._rmsnorm_kernels import _get_sm_count

    N = math.prod(normalized_shape)
    M = input.numel() // N
    x = input.reshape(M, N).contiguous()
    dout = grad_out.reshape(M, N).contiguous()
    rstd_flat = rstd.reshape(M).contiguous()
    w = weight.reshape(N) if weight is not None else None

    key = (x.dtype, w.dtype if w is not None else None, N)
    compiled = _bwd_compile_cache.get(key)
    if compiled is None:
        compiled = _get_compiled_bwd(*key)
        _bwd_compile_cache[key] = compiled

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(N, x.device)
    dw_partial: torch.Tensor | None = None
    if w is not None:
        dw_partial = torch.empty(sm_count, N, device=x.device, dtype=torch.float32)

    compiled(x, w, dout, None, rstd_flat, dx, dw_partial, None, None, sm_count)

    grad_input: torch.Tensor | None = dx.reshape(input.shape)
    grad_weight: torch.Tensor | None = (
        dw_partial.sum(dim=0).to(weight.dtype).reshape(weight.shape)  # pyrefly: ignore[missing-attribute]
        if weight is not None
        else torch.Tensor()
    )

    if not output_mask[0]:
        grad_input = None
    if not output_mask[1]:
        grad_weight = None
    return grad_input, grad_weight


def register_cutedsl_rmsnorm_overrides() -> None:
    if torch.cuda.is_available():
        fwd_fallback = torch.library.get_kernel("aten::_fused_rms_norm", "CUDA")
        bwd_fallback = torch.library.get_kernel(
            "aten::_fused_rms_norm_backward", "CUDA"
        )
    else:
        return

    fwd_impl = functools.partial(
        _cutedsl_fused_rms_norm_impl,
        fallback_kernel=fwd_fallback,
    )
    bwd_impl = functools.partial(
        _cutedsl_fused_rms_norm_backward_impl,
        fallback_kernel=bwd_fallback,
    )

    cu.register_op_override("aten", "_fused_rms_norm", "CUDA", fwd_impl)
    cu.register_op_override("aten", "_fused_rms_norm_backward", "CUDA", bwd_impl)


register_cutedsl_rmsnorm_overrides()
