"""GroupNorm CuTE DSL forward kernel.

Computes row-wise LayerNorm (mean subtraction, variance normalization, affine)
on a [M, K] tensor where M = N * group and K = (C // group) * HxW.
Weight and bias are pre-expanded to [M, K] by the adapter and tiled per-row.
"""

# pyre-ignore-all-errors
# pyrefly: ignore-errors
# ruff: noqa: S101

import math
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64

import torch
from torch import Tensor

from ._cute_utils import (
    allocate_reduction_buffer_and_mbar,
    copy,
    expand,
    get_tiled_copy,
    initialize_cluster,
    make_fake_tensor as fake_tensor,
    predicate_k,
    row_reduce,
)


_TORCH2CUTE_DTYPE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


class GroupNormFwd:
    """Row-wise LayerNorm kernel for GroupNorm forward with fused affine."""

    def __init__(self, dtype: type[cutlass.Numeric], K: int):
        self.dtype = dtype
        self.K = K
        self.stage = 1
        self.reduction_dtype = Float32
        self.reload_from = None if K <= 8192 else "smem"

    def _threads_per_row(self):
        K = self.K
        for limit, threads in [
            (64, 8),
            (128, 16),
            (3072, 32),
            (6144, 64),
            (16384, 128),
        ]:
            if K <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.K <= 16384 else 256

    def _set_cluster_n(self):
        K = self.K
        if const_expr(self.dtype.width == 16):
            thresholds = [
                (16 * 1024, 1),
                (32 * 1024, 2),
                (64 * 1024, 4),
                (128 * 1024, 8),
            ]
        else:
            thresholds = [
                (32 * 1024, 1),
                (64 * 1024, 2),
                (128 * 1024, 4),
                (256 * 1024, 8),
            ]
        for limit, cluster in thresholds:
            if K <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in [mX, mW, mB, mO]
                    if t is not None
                )
            )
        )
        vecsize = math.gcd(self.K, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = get_tiled_copy(
            self.dtype,
            self.K,
            self.cluster_n,
            self._threads_per_row(),
            self._num_threads(),
            vecsize=vecsize,
        )
        num_threads = tiled_copy.size
        if const_expr(mMean is not None):
            mMean = expand(mMean, dim=1, size=self.K)
        if const_expr(mRstd is not None):
            mRstd = expand(mRstd, dim=1, size=self.K)
        self.kernel(
            mX,
            mW,
            mB,
            mO,
            mMean,
            mRstd,
            eps,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor | None,
        mRstd: cute.Tensor | None,
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )
        reduction_buffer, mbar_ptr = allocate_reduction_buffer_and_mbar(
            smem, self.reduction_dtype, self.stage, self.cluster_n, tv_layout
        )

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, gMean, gRstd, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mO, mMean, mRstd, idX)
        ]
        # Weight/bias are 2D [M, K], tiled per-row like input
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mW, mB)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrMean = (
            thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        )
        tXrRstd = (
            thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        )
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_fragment_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_fragment_like(t) for t in (tXgX, tXgO)]

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        initialize_cluster(tidx, mbar_ptr, num_warps, self.cluster_n, self.stage)

        is_even_K = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_K
            else None
        )
        copy_ = partial(copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy_(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        # Load weight/bias while waiting for async copy of X
        if const_expr(mW is not None):
            copy_(tXgW, tXrW)
        if const_expr(mB is not None):
            copy_(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # Compute mean via row reduction
        sum_x = row_reduce(
            x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        mean_val = sum_x / shape[1]

        x_centered = x - mean_val

        # Compute variance: E[(x - mean)^2]
        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        sum_sq = row_reduce(
            x_centered * x_centered,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(sum_sq / shape[1] + eps, fastmath=True)

        # Store mean and rstd statistics
        if const_expr(mMean is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrMean[0] = mean_val
        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd

        # Reload x from smem if needed (large K)
        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
            x_centered = x - mean_val

        # Normalize and apply affine: y = (x - mean) * rstd * w + b
        y = x_centered * rstd
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy_(tXrO, tXgO)


def _groupnorm_fwd(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    out: Tensor,
    mean: Tensor,
    rstd: Tensor,
    eps: float = 1e-5,
) -> None:
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    assert x.dim() == 2, "Input must be 2D [M, K]"

    _, K = x.shape
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    compile_key = (dtype, out_dtype, weight_dtype, bias_dtype, K)
    if compile_key not in _groupnorm_fwd.compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, out_dtype, weight_dtype, bias_dtype]
        div = math.gcd(K, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, out_cute = [
            fake_tensor(dt, (batch_sym, K), div)
            for dt in [dtype, out_dtype]
        ]
        weight_cute, bias_cute = [
            fake_tensor(dt, (batch_sym, K), div) for dt in [weight_dtype, bias_dtype]
        ]
        mean_cute = fake_tensor(Float32, (batch_sym,))
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        _groupnorm_fwd.compile_cache[compile_key] = cute.compile(
            GroupNormFwd(dtype, K),
            x_cute,
            weight_cute,
            bias_cute,
            out_cute,
            mean_cute,
            rstd_cute,
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _groupnorm_fwd.compile_cache[compile_key](
        x, weight, bias, out, mean, rstd, eps
    )


_groupnorm_fwd.compile_cache = {}
