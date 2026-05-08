"""CuTeDSL override for aten::native_group_norm{,_backward}."""
# mypy: allow-untyped-defs

from __future__ import annotations

import functools

import torch

from ... import cutedsl_utils as cu


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.cache
def _get_device_major(device: torch.device) -> int:
    major, _ = torch.cuda.get_device_capability(device)
    return major


@functools.cache
def _get_groupnorm_kernels():
    from .norms import cutedsl_groupnorm_bwd, cutedsl_groupnorm_fwd

    return cutedsl_groupnorm_fwd, cutedsl_groupnorm_bwd


def _has_cow_tensor(*tensors: torch.Tensor | None) -> bool:
    is_cow = torch._C._is_cow_tensor  # pyrefly: ignore[missing-attribute]
    return any(t is not None and is_cow(t) for t in tensors)


def _groupnorm_supported(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    C: int,
    HxW: int,
    group: int,
    *optional_tensors: torch.Tensor | None,
) -> bool:
    if group <= 0 or C % group != 0:
        return False

    K = (C // group) * HxW
    if K < 32 or K > 1024 * 1024:
        return False

    if _get_device_major(input.device) not in (9, 10):
        return False

    for tensor in (input, weight, *optional_tensors):
        if tensor is not None and tensor.dtype not in _SUPPORTED_DTYPES:
            return False

    return True


def _native_group_norm_cond(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> bool:
    if _has_cow_tensor(input, weight, bias):
        return False
    return _groupnorm_supported(input, weight, C, HxW, group, bias)


def _native_group_norm_impl(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cutedsl_groupnorm_fwd, _ = _get_groupnorm_kernels()
    return cutedsl_groupnorm_fwd(input, weight, bias, N, C, HxW, group, eps)


def _native_group_norm_backward_cond(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> bool:
    if _has_cow_tensor(grad_out, input, mean, rstd, weight):
        return False
    if not _groupnorm_supported(input, weight, C, HxW, group, grad_out):
        return False
    if (
        output_mask[0]
        and not output_mask[1]
        and not output_mask[2]
        and weight is not None
        and weight.requires_grad
    ):
        return False
    K = (C // group) * HxW
    return input.dtype != torch.float32 or K <= 128 * 1024


def _native_group_norm_backward_impl(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    _, cutedsl_groupnorm_bwd = _get_groupnorm_kernels()
    return cutedsl_groupnorm_bwd(
        grad_out, input, mean, rstd, weight, N, C, HxW, group, output_mask
    )


def _register_for_dispatch_key(dispatch_key: str) -> None:
    cu.register_op_override(
        "aten",
        "native_group_norm",
        dispatch_key,
        cond=_native_group_norm_cond,
        impl=_native_group_norm_impl,
        allow_multiple_override=True,
    )
    cu.register_op_override(
        "aten",
        "native_group_norm_backward",
        dispatch_key,
        cond=_native_group_norm_backward_cond,
        impl=_native_group_norm_backward_impl,
        allow_multiple_override=True,
    )


def register_groupnorm_overrides() -> None:
    if not torch.cuda.is_available():
        return

    _register_for_dispatch_key("CUDA")
