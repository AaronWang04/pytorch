"""CuTeDSL override for aten::native_batch_norm{,_backward}."""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools

import torch

from ... import cutedsl_utils as cu


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _is_cow_tensor(t: torch.Tensor | None) -> bool:
    return t is not None and torch._C._is_cow_tensor(t)


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


@functools.cache
def _get_batchnorm_kernels():
    from .norms import cutedsl_batchnorm_bwd, cutedsl_batchnorm_fwd

    return cutedsl_batchnorm_fwd, cutedsl_batchnorm_bwd


def _is_compatible_param(t: torch.Tensor | None, C: int) -> bool:
    return t is None or (
        t.is_cuda
        and t.dtype in _SUPPORTED_DTYPES
        and t.dim() == 1
        and t.numel() == C
        and t.is_contiguous()
        and not _is_cow_tensor(t)
    )


def _is_supported_input(input: torch.Tensor) -> bool:
    return (
        input.is_cuda
        and input.dtype in _SUPPORTED_DTYPES
        and input.dim() >= 2
        and input.numel() > 0
        and input.is_contiguous()
        and not _is_cow_tensor(input)
        and _get_device_major(input.device) in (9, 10)
    )


def _is_supported_common(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
) -> bool:
    if input.dim() < 2:
        return False

    C = input.size(1)
    if C <= 0:
        return False

    reduction_size = input.numel() // C
    return (
        _is_supported_input(input)
        and reduction_size >= 2
        and reduction_size <= 1024 * 1024
        and _is_compatible_param(weight, C)
        and _is_compatible_param(running_mean, C)
        and _is_compatible_param(running_var, C)
    )


def _cutedsl_native_batch_norm_impl(
    dispatch_keys: torch.DispatchKeySet,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
    *,
    fallback_kernel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C = input.size(1) if input.dim() >= 2 else 0
    use_fallback = (
        not _is_supported_common(input, weight, running_mean, running_var)
        or not _is_compatible_param(bias, C)
        or (not training and (running_mean is None or running_var is None))
    )
    if use_fallback:
        return fallback_kernel.call_boxed(
            dispatch_keys,
            input,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
        )

    cutedsl_batchnorm_fwd, _ = _get_batchnorm_kernels()
    return cutedsl_batchnorm_fwd(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )


def _cutedsl_native_batch_norm_backward_impl(
    dispatch_keys: torch.DispatchKeySet,
    grad_out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    save_mean: torch.Tensor | None,
    save_invstd: torch.Tensor | None,
    train: bool,
    eps: float,
    output_mask: list[bool],
    *,
    fallback_kernel,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    use_fallback = (
        not train
        or not _is_supported_common(input, weight, running_mean, running_var)
        or input.dtype == torch.bfloat16
        or not grad_out.is_cuda
        or grad_out.dtype != input.dtype
        or grad_out.shape != input.shape
        or not grad_out.is_contiguous()
        or _is_cow_tensor(grad_out)
        or save_mean is None
        or save_invstd is None
        or not save_mean.is_cuda
        or not save_invstd.is_cuda
        or save_mean.dim() != 1
        or save_invstd.dim() != 1
        or save_mean.numel() != input.size(1)
        or save_invstd.numel() != input.size(1)
        or not save_mean.is_contiguous()
        or not save_invstd.is_contiguous()
        or _is_cow_tensor(save_mean)
        or _is_cow_tensor(save_invstd)
        or save_mean.dtype != torch.float32
        or save_invstd.dtype != torch.float32
        or (weight is None and (output_mask[1] or output_mask[2]))
    )
    if use_fallback:
        return fallback_kernel.call_boxed(
            dispatch_keys,
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            train,
            eps,
            output_mask,
        )

    _, cutedsl_batchnorm_bwd = _get_batchnorm_kernels()
    return cutedsl_batchnorm_bwd(
        grad_out, input, weight, save_mean, save_invstd, output_mask
    )


def _register_for_dispatch_key(dispatch_key: str) -> None:
    fwd_fallback = torch.library.get_kernel("aten::native_batch_norm", dispatch_key)
    bwd_fallback = torch.library.get_kernel(
        "aten::native_batch_norm_backward", dispatch_key
    )

    cu.register_op_override(
        "aten",
        "native_batch_norm",
        dispatch_key,
        functools.partial(_cutedsl_native_batch_norm_impl, fallback_kernel=fwd_fallback),
        allow_multiple_override=True,
    )
    cu.register_op_override(
        "aten",
        "native_batch_norm_backward",
        dispatch_key,
        functools.partial(
            _cutedsl_native_batch_norm_backward_impl, fallback_kernel=bwd_fallback
        ),
        allow_multiple_override=True,
    )


def register_to_dispatch() -> None:
    if not cu.runtime_available():
        return

    _register_for_dispatch_key("CUDA")
