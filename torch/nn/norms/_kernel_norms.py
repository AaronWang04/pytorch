"""CuteDSL norm kernels (RMSNorm forward + backward).

Ported from quack (github.com/Dao-AILab/quack), inlined to avoid
external utility modules.  Only the RMSNorm path is implemented;
LayerNorm stubs remain NotImplementedError.
"""

from __future__ import annotations

import math
import operator
from functools import partial
from typing import Optional
from functools import cache

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

import torch

# Dtype mapping
_TORCH2CUTE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: Float32,
}

# Utils
def _tiled_copy_2d(dtype, threads_per_row, num_threads, vecsize=1):
    num_copy_bits = vecsize * dtype.width
    copy_op = cpasync.CopyG2SOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0),
    )
    val_layout = cute.make_layout((1, vecsize))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@dsl_user_op
def _copy(src, dst, *, pred=None, is_async=False, loc=None, ip=None, **kwargs):
    num_copy_elems = src.shape[0][0]
    num_copy_bits = const_expr(min(128, num_copy_elems * src.element_type.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, src.element_type, num_bits_per_copy=num_copy_bits)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@cute.jit
def _predicate_k(tAcA, limit):
    from cutlass import Boolean
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


def _expand(tensor, *, dim, size):
    shape = (*tensor.shape[:dim], size, *tensor.shape[dim:])
    stride = (*tensor.layout.stride[:dim], 0, *tensor.layout.stride[dim:])
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


def _make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1):
    if leading_dim < 0:
        leading_dim = len(shape) + leading_dim
    if dtype is None:
        return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8,
    )


@dsl_user_op
def _elem_pointer(x, coord, *, loc=None, ip=None):
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def _set_block_rank(smem_ptr, peer_cta_rank_in_cluster, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote(val, smem_ptr, mbar_ptr, peer_cta_rank_in_cluster, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm
    remote_smem = _set_block_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    remote_mbar = _set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    suffix = {Float32: "f32", Int32: "s32", Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem, val.ir_value(loc=loc, ip=ip), remote_mbar],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _fill_oob(tXsX, tXpX, fill_value):
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])

# Row reduction
@cute.jit
def _block_reduce(val, op, reduction_buffer, init_val=0.0):
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def _cluster_reduce(val, op, reduction_buffer, mbar_ptr, init_val=0.0, phase=None):
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        _store_shared_remote(
            val,
            _elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def _row_reduce(x, op, threads_per_row, reduction_buffer=None, mbar_ptr=None,
                phase=None, init_val=0.0, hook_fn=None):
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(
        val, warp_op, threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            if const_expr(mbar_ptr is None):
                val = _block_reduce(val, warp_op, reduction_buffer, init_val=init_val)
            else:
                val = _cluster_reduce(
                    val, warp_op, reduction_buffer, mbar_ptr,
                    phase=phase, init_val=init_val,
                )
    return val


# RMSNorm forward kernel
class _RMSNormFwd:
    """CuTE DSL RMSNorm forward kernel (is_layernorm=False only)."""

    def __init__(self, dtype, N):
        self.dtype = dtype
        self.N = N
        self.stage = 1
        self.reduction_dtype = Float32
        self.reload_from = None if N <= 8192 else "smem"

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _set_cluster_n(self):
        N = self.N
        if const_expr(self.dtype.width == 16):
            thresholds = [(16384, 1), (32768, 2), (65536, 4), (131072, 8)]
        else:
            thresholds = [(32768, 1), (65536, 2), (131072, 4), (262144, 8)]
        for limit, cluster in thresholds:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _get_tiled_copy(self, vecsize=1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = _tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, self.cluster_n), self.stage),
            order=(1, 0, 2),
        )

    @cute.jit
    def __call__(self, mX, mW, mO, mRstd, eps, stream):
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mW, mO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = _expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None
        mRstd = _expand(mRstd, dim=1, size=self.N) if const_expr(mRstd is not None) else None
        self.kernel(mX, mW, mO, mRstd, eps, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX, mW, mO, mRstd, eps, tiler_mn, tiled_copy, threads_per_row):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16,
        )
        # Reduction buffer
        red_layout = self._get_reduction_buffer_layout(tv_layout)
        reduction_buffer = smem.allocate_tensor(self.reduction_dtype, red_layout, byte_alignment=8)
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage)
        else:
            mbar_ptr = None

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, gRstd, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mO, mRstd, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if const_expr(mW is not None) else None

        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXgO = thr_copy_X.partition_D(gO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        tXrW = cute.make_fragment_like(tXgW) if const_expr(mW is not None) else None
        tXrX = cute.make_fragment_like(tXgX)
        tXrO = cute.make_fragment_like(tXgO)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        # Initialize cluster barriers
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            _predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N else None
        )
        copy = partial(_copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(mW is not None):
            copy(tXgW, tXrW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)

        # RMSNorm: sum of squares
        sum_sq_x = _row_reduce(
            x * x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        rstd = cute.math.rsqrt(sum_sq_x / shape[1] + eps, fastmath=True)

        if const_expr(mRstd is not None):
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd

        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)

        y = x * rstd
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


# RMSNorm backward kernel
class _RMSNormBwd:
    """CuTE DSL RMSNorm backward kernel."""

    def __init__(self, dtype, N):
        self.dtype = dtype
        self.N = N
        self.stage = 2
        self.reduction_dtype = Float32
        self.reload_wdy = None if N <= 16384 else "smem"
        if N > 131072 and dtype.width >= 32:
            raise ValueError("RMSNormBackward does not support N > 128k with dtype >= 32 bits")

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (256, 32), (512, 64), (4096, 128)]:
            if N <= limit:
                return threads
        return 256

    def _num_threads(self):
        return 128 if self.N <= 4096 else 256

    def _set_cluster_n(self):
        N = self.N
        for limit, cluster in [(8192, 1), (16384, 2), (32768, 4), (65536, 8)]:
            if N <= limit:
                self.cluster_n = cluster
                return
        self.cluster_n = 16

    def _get_tiled_copy(self, vecsize=1):
        threads_per_row = self._threads_per_row()
        num_threads = self._num_threads()
        num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * self.cluster_n)
        tiler_mn = (num_threads // threads_per_row, vecsize * num_blocks_N * threads_per_row)
        tiled_copy = _tiled_copy_2d(self.dtype, threads_per_row, num_threads, vecsize)
        return tiled_copy, tiler_mn, threads_per_row

    def _get_reduction_buffer_layout(self, tv_layout):
        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        warps_per_row = (
            num_warps
            if cute.rank(tv_layout.shape[0]) == 1
            else max(tv_layout.shape[0][0] // cute.arch.WARP_SIZE, 1)
        )
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, self.cluster_n), self.stage),
            order=(1, 0, 2),
        )

    @cute.jit
    def __call__(self, mX, mW, mdO, mRstd, mdX, mdW, sm_count, stream):
        self._set_cluster_n()
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mW, mdO, mdX] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        mW = _expand(mW, dim=0, size=tiler_mn[0]) if const_expr(mW is not None) else None
        num_blocks = sm_count
        self.kernel(mX, mW, mdO, mRstd, mdX, mdW, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[num_blocks, self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if self.cluster_n > 1 else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX, mW, mdO, mRstd, mdX, mdW, tiler_mn, tiled_copy, threads_per_row):
        tidx, _, _ = cute.arch.thread_idx()
        bidx_start, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        shape = mX.shape
        M, N = shape[0], shape[1]
        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)

        idX = cute.make_identity_tensor(shape)

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_ordered_layout((tiler_mn[0], tiler_mn[1], 2), order=(1, 0, 2))
        sX = smem.allocate_tensor(mX.element_type, smem_layout, byte_alignment=16)
        sdO = smem.allocate_tensor(mdO.element_type, smem_layout, byte_alignment=16)
        red_layout = self._get_reduction_buffer_layout(tv_layout)
        reduction_buffer = smem.allocate_tensor(self.reduction_dtype, red_layout, byte_alignment=8)
        if const_expr(self.cluster_n > 1):
            mbar_ptr = smem.allocate_array(Int64, num_elems=self.stage * 2)
        else:
            mbar_ptr = None
        if const_expr(mbar_ptr is not None):
            mbar_full_ptr, mbar_empty_ptr = mbar_ptr, mbar_ptr + 2
        else:
            mbar_full_ptr, mbar_empty_ptr = None, None

        thr_copy_X = tiled_copy.get_slice(tidx)

        gX, gdO, gdX, cX = [
            cute.local_tile(mT, tiler_mn, (None, cluster_y)) if mT is not None else None
            for mT in (mX, mdO, mdX, idX)
        ]
        gW = cute.local_tile(mW, tiler_mn, (0, cluster_y)) if mW is not None else None
        gdW = (
            cute.local_tile(mdW, (1, tiler_mn[1]), (bidx_start, cluster_y))
            if const_expr(mdW is not None) else None
        )

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
            None if is_even_N
            else _predicate_k(thr_copy_X.partition_S(cX[None, None, 0]), limit=shape[1])
        )
        copy = partial(_copy, pred=tXpX)

        tXgdW, tXrdW = None, None
        if const_expr(mdW is not None):
            tXgdW = thr_copy_X.partition_S(gdW)
            tXrdW = cute.make_fragment_like(tXgdW, Float32)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE

        # Initialize cluster barriers (persistent mode)
        if const_expr(self.cluster_n > 1):
            if tidx < self.stage:
                cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
                cute.arch.mbarrier_init(mbar_ptr + self.stage + tidx, num_warps * self.cluster_n)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()

        tXrW = None
        if const_expr(mW is not None):
            tXgW = thr_copy_X.partition_S(gW)
            tXrW = cute.make_fragment_like(tXgW)
            if const_expr(not is_even_N):
                tXrW.fill(0.0)
            copy(tXgW, tXrW)

        # Prefetch first batch
        row = tXcX[None, None, None, bidx_start][0][0]
        if row < M:
            copy(tXgX[None, None, None, bidx_start], tXsX[None, None, None, 0], is_async=True)
            copy(tXgdO[None, None, None, bidx_start], tXsdO[None, None, None, 0], is_async=True)
        else:
            if const_expr(tiler_mn[0] > 1):
                _fill_oob(tXsX[None, None, None, 0], None, fill_value=mX.element_type.zero)
                _fill_oob(tXsdO[None, None, None, 0], None, fill_value=mdO.element_type.zero)
        cute.arch.cp_async_commit_group()

        if const_expr(self.cluster_n > 1):
            cute.arch.cluster_wait()

        if const_expr(mdW is not None):
            tXrdW.fill(0.0)
        stage = Int32(0)
        producer_phase = Int32(1)
        consumer_phase = Int32(0)
        for bidx in cutlass.range(bidx_start, cute.ceil_div(M, tiler_mn[0]), gdim):
            row = tXcX[None, None, None, bidx][0][0]
            if row + gdim * tiler_mn[0] < M:
                copy(tXgX[None, None, None, bidx + gdim], tXsX[None, None, None, stage ^ 1], is_async=True)
                copy(tXgdO[None, None, None, bidx + gdim], tXsdO[None, None, None, stage ^ 1], is_async=True)
            else:
                if const_expr(tiler_mn[0] > 1):
                    _fill_oob(tXsX[None, None, None, stage ^ 1], None, fill_value=mX.element_type.zero)
                    _fill_oob(tXsdO[None, None, None, stage ^ 1], None, fill_value=mdO.element_type.zero)
            cute.arch.cp_async_commit_group()
            rstd = cutlass.Float.zero
            if row < M or tiler_mn[0] == 1:
                rstd = mRstd[row]
            cute.arch.cp_async_wait_group(1)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            x = tXrX.load().to(cute.Float32)
            cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
            dout = tXrdO.load().to(cute.Float32)
            x_hat = x * rstd
            wdy = dout
            if const_expr(mW is not None):
                wdy *= tXrW.load().to(Float32)
            if const_expr(self.cluster_n > 1):
                cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)
            mean_xhat_wdy = (
                _row_reduce(
                    x_hat * wdy,
                    cute.ReductionOp.ADD,
                    threads_per_row,
                    reduction_buffer[None, None, stage],
                    mbar_full_ptr + stage if const_expr(self.cluster_n > 1) else None,
                    phase=consumer_phase,
                    init_val=0.0,
                )
                / shape[1]
            )

            if const_expr(self.cluster_n > 1):
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                lane_idx = cute.arch.lane_idx()
                if lane_idx < self.cluster_n:
                    cute.arch.mbarrier_arrive(
                        mbar_empty_ptr + stage, peer_cta_rank_in_cluster=lane_idx,
                    )

            if const_expr(self.reload_wdy == "smem"):
                cute.autovec_copy(tXsdO[None, None, None, stage], tXrdO)
                dout = tXrdO.load().to(cute.Float32)
                wdy = dout
                if const_expr(mW is not None):
                    wdy *= tXrW.load().to(Float32)

            dx = (wdy - x_hat * mean_xhat_wdy) * rstd
            tXrdX.store(dx.to(tXrdX.element_type))
            if row < M or tiler_mn[0] == 1:
                copy(tXrdX, tXgdX[None, None, None, bidx])
            if const_expr(mdW is not None):
                tXrdW.store(tXrdW.load() + dout * x_hat)

            stage ^= 1
            if stage == 0:
                consumer_phase ^= 1
                producer_phase ^= 1

        # Reduce partial dW across rows within the threadblock
        if const_expr(mdW is not None):
            if const_expr(tiler_mn[0] > 1):
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
                            tXsdW.iterator + i * sdW.stride[0], tXsdW.layout,
                        )
                        cute.autovec_copy(tXsdW_other, tXrdW_other)
                        tXrdW.store(tXrdW.load() + tXrdW_other.load())
                    copy(tXrdW, tXgdW)
            else:
                copy(tXrdW, tXgdW)

        if const_expr(self.cluster_n > 1):
            stage ^= 1
            if stage == 0:
                producer_phase ^= 1
            cute.arch.mbarrier_wait(mbar_empty_ptr + stage, producer_phase)


# Compilation caches
_fwd_compile_cache: dict = {}
_bwd_compile_cache: dict = {}

@cache
def _get_sm_count(N: int, device: torch.device) -> int:
    sm_count_multiple = (
        16 if N <= 256 else (8 if N <= 1024 else (4 if N <= 2048 else (2 if N <= 4096 else 1)))
    )
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sm_count = (
        sm_count * sm_count_multiple if N <= 8192 else sm_count // 2 if N <= 16384 else sm_count * 2
    )
    return sm_count


# Public API
def cutedsl_rmsnorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = input.reshape(-1, normalized_shape[-1])
    M, N = x.shape

    dtype = _TORCH2CUTE[x.dtype]
    out_dtype = _TORCH2CUTE[x.dtype]
    weight_dtype = _TORCH2CUTE[weight.dtype] if weight is not None else None

    compile_key = (dtype, out_dtype, weight_dtype, N)
    if compile_key not in _fwd_compile_cache:
        batch_sym = cute.sym_int()
        all_dtypes = [dtype, out_dtype, weight_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute = _make_fake_tensor(dtype, (batch_sym, N), div)
        out_cute = _make_fake_tensor(out_dtype, (batch_sym, N), div)
        weight_cute = _make_fake_tensor(weight_dtype, (N,), div)
        rstd_cute = _make_fake_tensor(Float32, (batch_sym,))
        _fwd_compile_cache[compile_key] = cute.compile(
            _RMSNormFwd(dtype, N),
            x_cute,
            weight_cute,
            out_cute,
            rstd_cute,
            Float32(0),  # eps placeholder
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    out = torch.empty_like(x)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)

    _fwd_compile_cache[compile_key](x, weight, out, rstd, eps)

    return out.view_as(input), rstd


def cutedsl_rmsnorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    N = normalized_shape[-1]
    x = input.reshape(-1, N)
    dout = grad_out.reshape(-1, N)
    M = x.shape[0]
    device = x.device

    dtype = _TORCH2CUTE[x.dtype]
    dout_dtype = _TORCH2CUTE[dout.dtype]
    dx_dtype = _TORCH2CUTE[x.dtype]
    weight_dtype = _TORCH2CUTE[weight.dtype] if weight is not None else None

    compile_key = (N, dtype, dout_dtype, dx_dtype, weight_dtype)
    if compile_key not in _bwd_compile_cache:
        batch_sym, batch_partial_sym = cute.sym_int(), cute.sym_int()
        all_dtypes = [dtype, dout_dtype, dx_dtype]
        div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
        x_cute = _make_fake_tensor(dtype, (batch_sym, N), div)
        dout_cute = _make_fake_tensor(dout_dtype, (batch_sym, N), div)
        dx_cute = _make_fake_tensor(dx_dtype, (batch_sym, N), div)
        weight_cute = _make_fake_tensor(weight_dtype, (N,), div)
        rstd_cute = _make_fake_tensor(Float32, (batch_sym,))
        dw_partial_cute = (
            _make_fake_tensor(Float32, (batch_partial_sym, N), div)
            if weight is not None else None
        )
        _bwd_compile_cache[compile_key] = cute.compile(
            _RMSNormBwd(dtype, N),
            x_cute,
            weight_cute,
            dout_cute,
            rstd_cute,
            dx_cute,
            dw_partial_cute,
            0,  # sm_count placeholder
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    dx = torch.empty_like(x)
    sm_count = _get_sm_count(N, device)
    dw_partial = (
        torch.empty(sm_count, N, device=device, dtype=torch.float32)
        if weight is not None else None
    )

    _bwd_compile_cache[compile_key](x, weight, dout, rstd, dx, dw_partial, sm_count)

    dw = dw_partial.sum(dim=0).to(weight.dtype) if weight is not None else None
    return dx.view_as(grad_out), dw


# LayerNorm stubs (if we want to add mayb)
def cutedsl_layernorm_fwd(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("cutedsl_layernorm_fwd")


def cutedsl_layernorm_bwd(
    grad_out: torch.Tensor,
    input: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    normalized_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("cutedsl_layernorm_bwd")
