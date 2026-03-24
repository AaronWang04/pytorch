"""GroupNorm CuTE DSL forward and backward kernels.

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
    fill_oob,
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

        # Compute variance as E[x^2] - mean^2 to avoid padding contamination.
        # Padding positions are zero (from predicated cp.async), so x^2 = 0
        # contributes nothing. Using (x - mean)^2 would be wrong because
        # padding positions become (-mean)^2 = mean^2.
        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        sum_x_sq = row_reduce(
            x * x,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(
            sum_x_sq / shape[1] - mean_val * mean_val + eps, fastmath=True
        )

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

        # Normalize and apply affine: y = (x - mean) * rstd * w + b
        y = (x - mean_val) * rstd
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


class GroupNormBackward:
    """Persistent backward kernel for GroupNorm on [M, K] tensors.

    Like RMSNormBackward but with an extra mean-subtraction correction term.
    dx = (wdy - x_hat * mean(x_hat * wdy) - mean(wdy)) * rstd
    where x_hat = (x - mean) * rstd.
    """

    def __init__(self, dtype: type[cutlass.Numeric], K: int):
        self.dtype = dtype
        self.K = K
        self.stage = 2
        self.reduction_dtype = Float32
        self.reload_wdy = None if K <= 16 * 1024 else "smem"
        if self.K > 128 * 1024 and self.dtype.width >= 32:
            raise ValueError(
                "GroupNormBackward does not support K > 128k with dtype >= 32 bits"
            )

    def _num_threads(self):
        return 128 if self.K <= 4096 else 256

    def _threads_per_row(self):
        K = self.K
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if K <= limit:
                return threads
        return 256

    def _set_cluster_n(self):
        K = self.K
        for limit, cluster in [
            (8 * 1024, 1),
            (16 * 1024, 2),
            (32 * 1024, 4),
            (64 * 1024, 8),
        ]:
            if K <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        sm_count: Int32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(
                *(
                    t.element_type.width
                    for t in [mX, mW, mdO, mdX]
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
        num_blocks = sm_count
        self.kernel(
            mX,
            mW,
            mdO,
            mMean,
            mRstd,
            mdX,
            mdW,
            mdB,
            tiler_mn,
            tiled_copy,
            threads_per_row,
        ).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mRstd: cute.Tensor,
        mdX: cute.Tensor,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = (
            const_expr(0)
            if const_expr(self.cluster_n == 1)
            else cute.arch.block_idx()[1]
        )
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M, _K = shape[0], shape[1]
        is_even_K = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout(
            (tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2)
        )
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        reduction_buffer, mbar_ptr = allocate_reduction_buffer_and_mbar(
            smem,
            self.reduction_dtype,
            self.stage,
            self.cluster_n,
            tv_layout,
            is_persistent=True,
        )
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y))
            for mT in (mX, mdO, mdX, idX)
        ]
        # Weight is [M, K] pre-expanded, tiled per-row like input
        gW = (
            cute.local_tile(mW, tiler_mn, (None, cluster_y))
            if const_expr(mW is not None)
            else None
        )
        gdW, gdB = [
            cute.local_tile(mT, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mT is not None)
            else None
            for mT in (mdW, mdB)
        ]

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgdO = thr_copy_X.partition_S(gdO)
        tXsdO = thr_copy_X.partition_D(sdO)
        tXgdX = thr_copy_X.partition_D(gdX)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None, None]

        tXrX, tXrdO, tXrdX = [
            cute.make_fragment_like(thr[None, None, None, 0])
            for thr in (tXgX, tXgdO, tXgdX)
        ]

        tXpX = (
            None
            if is_even_K
            else predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        copy_ = partial(copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        tXgdB, tXrdB = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)
        if const_expr(mdB is not None):
            tXgdB = thr_copy_X.partition_S(gdB)
            tXrdB = cute.make_fragment_like(tXgdB, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        initialize_cluster(
            tidx,
            mbar_ptr,
            num_warps,
            self.cluster_n,
            self.stage,
            is_persistent=True,
        )

        # Load weight for this row (persistent: weight changes per row for GroupNorm)
        tXrW = None
        tXgW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW[None, None, None, 0])

        # Prefetch the first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            copy_(
                tXgX[None, None, None, bidx_start],
                tXsX[None, None, None, 0],
                is_async=True,
            )
            copy_(
                tXgdO[None, None, None, bidx_start],
                tXsdO[None, None, None, 0],
                is_async=True,
            )
        else:
            if const_expr(tiler_mn[0] > 1):
                fill_oob(
                    tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero
                )
                fill_oob(
                    tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero
                )
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        if const_expr(mdB is not None):
            tXrdB.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                copy_(
                    tXgX[None, None, None, bidx + gdim],
                    tXsX[None, None, None, stage ^ 1],
                    is_async=True,
                )
                copy_(
                    tXgdO[None, None, None, bidx + gdim],
                    tXsdO[None, None, None, stage ^ 1],
                    is_async=True,
                )
            else:
                if const_expr(tiler_mn[0] > 1):
                    fill_oob(
                        tXsX[None, None, None, stage ^ 1],
                        None,
                        fill_value=mX.element_type.zero,
                    )
                    fill_oob(
                        tXsdO[None, None, None, stage ^ 1],
                        None,
                        fill_value=mdO.element_type.zero,
                    )
            cute.arch.cp_async_commit_group()
            mean_val = cutlass.Float.zero
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                mean_val = mMean[row]
                rstd = mRstd[row]

            # Load weight for this row (weight is [M, K], varies per row)
            if const_expr(mW is not None):
                if const_expr(not is_even_K):
                    tXrW.fill(0.0)
                copy_(tXgW[None, None, None, bidx], tXrW)

            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = (x - mean_val) * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)

            # Two reductions needed: mean(wdy) and mean(x_hat * wdy)
            # Use mbar protocol for first, cluster_wait between, mbar for second
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_wdy = (
                row_reduce(
                    wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                cute.arch.cluster_wait()

            mean_xhat_wdy = (
                row_reduce(
                    x_hat * wdy,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    (mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None),
                    phase=consumer_phase,
                    init_val=0.0,
                    hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
                )
                / shape[1]
            )

            # Signal buffer is free for next iteration
            if const_expr(self.cluster_n > 1):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    copy_(tXgW[None, None, None, bidx], tXrW)
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy - mean_wdy) * rstd
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy_(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)
            if const_expr(mdB is not None):
                tXrdB.store(tXrdB.load() + dout)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        if const_expr(tiler_mn[0] > 1):
            if const_expr(mdW is not None):
                sdW = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdW = thr_copy_X.partition_D(sdW)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdW, tXsdW)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdW_other = cute.make_fragment_like(tXrdW)
                        tXsdW_other = cute.make_tensor(
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy_(tXrdW, tXgdW)
                cute.arch.barrier()
            if const_expr(mdB is not None):
                sdB = cute.make_tensor(
                    cute.recast_ptr(sX.iterator, dtype=cute.Float32),
                    cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                )
                tXsdB = thr_copy_X.partition_D(sdB)
                cute.arch.barrier()
                row = tXcX[None, None, None, 0][0][0]
                if row > 0:
                    cute.autovec_copy(tXrdB, tXsdB)
                cute.arch.barrier()
                if row == 0:
                    for i in cutlass.range_constexpr(1, const_expr(tiler_mn[0])):
                        tXrdB_other = cute.make_fragment_like(tXrdB)
                        tXsdB_other = cute.make_tensor(
                            tXsdB.iterator + i * sdB.stride[0], tXsdB.layout
                        )
                        cute.autovec_copy(tXsdB_other, tXrdB_other)
                        tXrdB.store(tXrdB.load() + tXrdB_other.load())
                    copy_(tXrdB, tXgdB)
        else:
            if const_expr(mdW is not None):
                copy_(tXrdW, tXgdW)
            if const_expr(mdB is not None):
                copy_(tXrdB, tXgdB)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


def _get_sm_count(K: int, device: torch.device) -> int:
    sm_count_multiple = (
        16
        if K <= 256
        else (8 if K <= 1024 else (4 if K <= 2048 else (2 if K <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm_count = (
        sm_count * sm_count_multiple
        if K <= 8192
        else sm_count // 2
        if K <= 16384
        else sm_count * 2
    )
    return sm_count


def _groupnorm_bwd(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    mean: Tensor,
    rstd: Tensor,
    dx: Tensor,
    dw_partial: Tensor | None,
    db_partial: Tensor | None = None,
    sm_count: int | None = None,
) -> None:
    assert x.dim() == 2, "Input must be 2D"
    assert x.is_cuda, "Input tensor must be on CUDA device"
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    assert x.dtype in supported_types, "Unsupported dtype"
    if weight is not None:
        assert weight.dim() == 2, "Weight must be 2D [M, K]"
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dtype in supported_types

    K = x.size(1)
    if dw_partial is None and db_partial is None:
        assert sm_count is not None
    else:
        sm_count = (
            dw_partial.shape[0] if dw_partial is not None else db_partial.shape[0]
        )
    dtype, dout_dtype, dx_dtype, weight_dtype = [
        _TORCH2CUTE_DTYPE[t.dtype] if t is not None else None
        for t in [x, dout, dx, weight]
    ]
    compile_key = (
        K,
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        dw_partial is not None,
        db_partial is not None,
    )
    if compile_key not in _groupnorm_bwd.compile_cache:
        batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
        all_dtypes = [dtype, dout_dtype, dx_dtype]
        div = math.gcd(K, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute, dout_cute, dx_cute = [
            fake_tensor(dt, (batch_sym, K), div)
            for dt in [dtype, dout_dtype, dx_dtype]
        ]
        weight_cute = fake_tensor(weight_dtype, (batch_sym, K), div)
        mean_cute = fake_tensor(Float32, (batch_sym,))
        rstd_cute = fake_tensor(Float32, (batch_sym,))
        dw_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, K), div)
            if dw_partial is not None
            else None
        )
        db_partial_cute = (
            fake_tensor(Float32, (batch_partial_sym, K), div)
            if db_partial is not None
            else None
        )
        _groupnorm_bwd.compile_cache[compile_key] = cute.compile(
            GroupNormBackward(dtype, K),
            x_cute,
            weight_cute,
            dout_cute,
            mean_cute,
            rstd_cute,
            dx_cute,
            dw_partial_cute,
            db_partial_cute,
            sm_count,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _groupnorm_bwd.compile_cache[compile_key](
        x,
        weight,
        dout,
        mean,
        rstd,
        dx,
        dw_partial,
        db_partial,
        sm_count,
    )


_groupnorm_bwd.compile_cache = {}
