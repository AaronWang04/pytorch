"""BatchNorm CuTE DSL forward and backward kernels.

The supported fast path assumes contiguous NCHW/NCDHW-style storage and treats
the input as [N, C, S], where S is the flattened spatial size. One CTA owns one
channel and reduces over N * S.
"""

# pyre-ignore-all-errors
# pyrefly: ignore-errors
# ruff: noqa: S101

import operator

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64

from quack.compile_utils import make_fake_tensor as fake_tensor

import torch
from torch import Tensor


_TORCH2CUTE_DTYPE = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


class _BatchNormBase:
    def __init__(self, S: int, num_threads: int = 256):
        self.S = S
        self.num_threads = num_threads
        self.num_warps = num_threads // cute.arch.WARP_SIZE

    @cute.jit
    def _cta_reduce_sum(
        self,
        val: Float32,
        acc: cute.Tensor,
    ) -> Float32:
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        val = cute.arch.warp_reduction(val, operator.add)
        if lane_idx == 0:
            acc[warp_idx] = val
        cute.arch.barrier()

        block_val = Float32(0.0)
        if warp_idx == 0:
            if lane_idx < self.num_warps:
                block_val = acc[lane_idx]
            block_val = cute.arch.warp_reduction(block_val, operator.add)
            if lane_idx == 0:
                acc[self.num_warps] = block_val
        cute.arch.barrier()
        return acc[self.num_warps]

    @cute.jit
    def _offset_to_ns(self, r: Int32) -> tuple[Int32, Int32]:
        n = r // self.S
        s = r - n * self.S
        return n, s


class BatchNormFwd(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor | None,
        mRunningVar: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        momentum: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mX,
            mW,
            mB,
            mRunningMean,
            mRunningVar,
            mO,
            mMean,
            mInvstd,
            momentum,
            eps,
        ).launch(
            grid=[mX.shape[1], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor | None,
        mRunningVar: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        momentum: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        R = mX.shape[0] * self.S

        smem = cutlass.utils.SmemAllocator()
        acc_sum = smem.allocate_tensor(Float32, self.num_warps + 1)
        acc_sumsq = smem.allocate_tensor(Float32, self.num_warps + 1)

        sum_x = Float32(0.0)
        sum_x_sq = Float32(0.0)
        for r in cutlass.range(tidx, R, self.num_threads):
            n, s = self._offset_to_ns(r)
            x = mX[n, c, s].to(Float32)
            sum_x += x
            sum_x_sq += x * x

        sum_x = self._cta_reduce_sum(sum_x, acc_sum)
        sum_x_sq = self._cta_reduce_sum(sum_x_sq, acc_sumsq)

        mean = sum_x / R
        var = sum_x_sq / R - mean * mean
        var = cute.arch.fmax(var, Float32(0.0))
        invstd = cute.math.rsqrt(var + eps, fastmath=True)

        if tidx == 0:
            mMean[c] = mean
            mInvstd[c] = invstd
            if const_expr(mRunningMean is not None):
                old_mean = mRunningMean[c].to(Float32)
                mRunningMean[c] = (
                    old_mean * (Float32(1.0) - momentum) + mean * momentum
                ).to(mRunningMean.element_type)
            if const_expr(mRunningVar is not None):
                old_var = mRunningVar[c].to(Float32)
                unbiased_var = var * R / (R - 1)
                mRunningVar[c] = (
                    old_var * (Float32(1.0) - momentum) + unbiased_var * momentum
                ).to(mRunningVar.element_type)

        gamma = Float32(1.0)
        beta = Float32(0.0)
        if const_expr(mW is not None):
            gamma = mW[c].to(Float32)
        if const_expr(mB is not None):
            beta = mB[c].to(Float32)

        for r in cutlass.range(tidx, R, self.num_threads):
            n, s = self._offset_to_ns(r)
            y = (mX[n, c, s].to(Float32) - mean) * invstd * gamma + beta
            mO[n, c, s] = y.to(mO.element_type)


class BatchNormStats(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mRunningMean: cute.Tensor | None,
        mRunningVar: cute.Tensor | None,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        momentum: Float32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mX, mRunningMean, mRunningVar, mMean, mInvstd, momentum, eps
        ).launch(
            grid=[mX.shape[1], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mRunningMean: cute.Tensor | None,
        mRunningVar: cute.Tensor | None,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        momentum: Float32,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        R = mX.shape[0] * self.S

        smem = cutlass.utils.SmemAllocator()
        acc_sum = smem.allocate_tensor(Float32, self.num_warps + 1)
        acc_sumsq = smem.allocate_tensor(Float32, self.num_warps + 1)

        sum_x = Float32(0.0)
        sum_x_sq = Float32(0.0)
        for r in cutlass.range(tidx, R, self.num_threads):
            n, s = self._offset_to_ns(r)
            x = mX[n, c, s].to(Float32)
            sum_x += x
            sum_x_sq += x * x

        sum_x = self._cta_reduce_sum(sum_x, acc_sum)
        sum_x_sq = self._cta_reduce_sum(sum_x_sq, acc_sumsq)

        if tidx == 0:
            mean = sum_x / R
            var = sum_x_sq / R - mean * mean
            var = cute.arch.fmax(var, Float32(0.0))
            invstd = cute.math.rsqrt(var + eps, fastmath=True)
            mMean[c] = mean
            mInvstd[c] = invstd
            if const_expr(mRunningMean is not None):
                old_mean = mRunningMean[c].to(Float32)
                mRunningMean[c] = (
                    old_mean * (Float32(1.0) - momentum) + mean * momentum
                ).to(mRunningMean.element_type)
            if const_expr(mRunningVar is not None):
                old_var = mRunningVar[c].to(Float32)
                unbiased_var = var * R / (R - 1)
                mRunningVar[c] = (
                    old_var * (Float32(1.0) - momentum) + unbiased_var * momentum
                ).to(mRunningVar.element_type)


class BatchNormApply(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        stream: cuda.CUstream,
    ):
        R = mX.shape[0] * self.S
        self.kernel(mX, mW, mB, mO, mMean, mInvstd).launch(
            grid=[mX.shape[1], cute.ceil_div(R, self.num_threads), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, rbidx, _ = cute.arch.block_idx()
        _, grid_y, _ = cute.arch.grid_dim()
        R = mX.shape[0] * self.S

        mean = mMean[c]
        invstd = mInvstd[c]
        gamma = Float32(1.0)
        beta = Float32(0.0)
        if const_expr(mW is not None):
            gamma = mW[c].to(Float32)
        if const_expr(mB is not None):
            beta = mB[c].to(Float32)

        start = rbidx * self.num_threads + tidx
        stride = grid_y * self.num_threads
        for r in cutlass.range(start, R, stride):
            n, s = self._offset_to_ns(r)
            y = (mX[n, c, s].to(Float32) - mean) * invstd * gamma + beta
            mO[n, c, s] = y.to(mO.element_type)


class BatchNormEval(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor,
        mRunningVar: cute.Tensor,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mX, mW, mB, mRunningMean, mRunningVar, mO, mMean, mInvstd, eps
        ).launch(
            grid=[mX.shape[1], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor,
        mRunningVar: cute.Tensor,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        R = mX.shape[0] * self.S

        mean = mRunningMean[c].to(Float32)
        invstd = cute.math.rsqrt(mRunningVar[c].to(Float32) + eps, fastmath=True)
        if tidx == 0:
            mMean[c] = mean
            mInvstd[c] = invstd

        gamma = Float32(1.0)
        beta = Float32(0.0)
        if const_expr(mW is not None):
            gamma = mW[c].to(Float32)
        if const_expr(mB is not None):
            beta = mB[c].to(Float32)

        for r in cutlass.range(tidx, R, self.num_threads):
            n, s = self._offset_to_ns(r)
            y = (mX[n, c, s].to(Float32) - mean) * invstd * gamma + beta
            mO[n, c, s] = y.to(mO.element_type)


class BatchNormEvalApply(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor,
        mRunningVar: cute.Tensor,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        R = mX.shape[0] * self.S
        self.kernel(
            mX, mW, mB, mRunningMean, mRunningVar, mO, mMean, mInvstd, eps
        ).launch(
            grid=[mX.shape[1], cute.ceil_div(R, self.num_threads), 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mB: cute.Tensor | None,
        mRunningMean: cute.Tensor,
        mRunningVar: cute.Tensor,
        mO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        eps: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, rbidx, _ = cute.arch.block_idx()
        _, grid_y, _ = cute.arch.grid_dim()
        R = mX.shape[0] * self.S

        mean = mRunningMean[c].to(Float32)
        invstd = cute.math.rsqrt(mRunningVar[c].to(Float32) + eps, fastmath=True)
        if rbidx == 0 and tidx == 0:
            mMean[c] = mean
            mInvstd[c] = invstd

        gamma = Float32(1.0)
        beta = Float32(0.0)
        if const_expr(mW is not None):
            gamma = mW[c].to(Float32)
        if const_expr(mB is not None):
            beta = mB[c].to(Float32)

        start = rbidx * self.num_threads + tidx
        stride = grid_y * self.num_threads
        for r in cutlass.range(start, R, stride):
            n, s = self._offset_to_ns(r)
            y = (mX[n, c, s].to(Float32) - mean) * invstd * gamma + beta
            mO[n, c, s] = y.to(mO.element_type)


class BatchNormBackward(_BatchNormBase):
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        mdX: cute.Tensor | None,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mW, mdO, mMean, mInvstd, mdX, mdW, mdB).launch(
            grid=[mX.shape[1], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor | None,
        mdO: cute.Tensor,
        mMean: cute.Tensor,
        mInvstd: cute.Tensor,
        mdX: cute.Tensor | None,
        mdW: cute.Tensor | None,
        mdB: cute.Tensor | None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        c, _, _ = cute.arch.block_idx()
        R = mX.shape[0] * self.S

        mean = mMean[c]
        invstd = mInvstd[c]

        smem = cutlass.utils.SmemAllocator()
        acc_sum_dy = smem.allocate_tensor(Float32, self.num_warps + 1)
        acc_sum_dy_xmu = smem.allocate_tensor(Float32, self.num_warps + 1)

        sum_dy = Float32(0.0)
        sum_dy_xmu = Float32(0.0)
        for r in cutlass.range(tidx, R, self.num_threads):
            n, s = self._offset_to_ns(r)
            dout = mdO[n, c, s].to(Float32)
            xmu = mX[n, c, s].to(Float32) - mean
            sum_dy += dout
            sum_dy_xmu += dout * xmu

        sum_dy = self._cta_reduce_sum(sum_dy, acc_sum_dy)
        sum_dy_xmu = self._cta_reduce_sum(sum_dy_xmu, acc_sum_dy_xmu)

        gamma = Float32(1.0)
        if const_expr(mW is not None):
            gamma = mW[c].to(Float32)

        if tidx == 0:
            if const_expr(mdW is not None):
                mdW[c] = (sum_dy_xmu * invstd).to(mdW.element_type)
            if const_expr(mdB is not None):
                mdB[c] = sum_dy.to(mdB.element_type)

        if const_expr(mdX is not None):
            mean_dy = sum_dy / R
            proj_scale = sum_dy_xmu * invstd * invstd / R
            grad_scale = invstd * gamma
            for r in cutlass.range(tidx, R, self.num_threads):
                n, s = self._offset_to_ns(r)
                dout = mdO[n, c, s].to(Float32)
                xmu = mX[n, c, s].to(Float32) - mean
                dx = (dout - mean_dy - xmu * proj_scale) * grad_scale
                mdX[n, c, s] = dx.to(mdX.element_type)


def _shape_key(x: Tensor) -> tuple[int, int]:
    return x.size(1), x.numel() // (x.size(0) * x.size(1))


def _num_threads_for_spatial(S: int) -> int:
    return 512 if S >= 64 else 256


def _use_split_eval_apply(x: Tensor, S: int) -> bool:
    return x.size(0) * S >= 96 * 1024


def _use_split_train_apply(x: Tensor, S: int) -> bool:
    return x.size(0) * S >= 96 * 1024


def _batchnorm_stats(
    x: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    mean: Tensor,
    invstd: Tensor,
    momentum: float,
    eps: float,
) -> None:
    C, S = _shape_key(x)
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    running_mean_dtype = (
        _TORCH2CUTE_DTYPE[running_mean.dtype] if running_mean is not None else None
    )
    running_var_dtype = (
        _TORCH2CUTE_DTYPE[running_var.dtype] if running_var is not None else None
    )
    compile_key = (dtype, running_mean_dtype, running_var_dtype, C, S)
    if compile_key not in _batchnorm_stats.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        running_mean_cute = (
            fake_tensor(running_mean_dtype, (C,)) if running_mean_dtype else None
        )
        running_var_cute = (
            fake_tensor(running_var_dtype, (C,)) if running_var_dtype else None
        )
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        _batchnorm_stats.compile_cache[compile_key] = cute.compile(
            BatchNormStats(S, _num_threads_for_spatial(S)),
            x_cute,
            running_mean_cute,
            running_var_cute,
            mean_cute,
            invstd_cute,
            Float32(0),
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_stats.compile_cache[compile_key](
        x, running_mean, running_var, mean, invstd, momentum, eps,
    )


_batchnorm_stats.compile_cache = {}


def _batchnorm_apply(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    out: Tensor,
    mean: Tensor,
    invstd: Tensor,
) -> None:
    C, S = _shape_key(x)
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    compile_key = (dtype, out_dtype, weight_dtype, bias_dtype, C, S)
    if compile_key not in _batchnorm_apply.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        out_cute = fake_tensor(out_dtype, (batch_sym, C, S))
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        bias_cute = fake_tensor(bias_dtype, (C,)) if bias_dtype else None
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        _batchnorm_apply.compile_cache[compile_key] = cute.compile(
            BatchNormApply(S, _num_threads_for_spatial(S)),
            x_cute,
            weight_cute,
            bias_cute,
            out_cute,
            mean_cute,
            invstd_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_apply.compile_cache[compile_key](
        x, weight, bias, out, mean, invstd,
    )


_batchnorm_apply.compile_cache = {}


def _batchnorm_fwd(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    out: Tensor,
    mean: Tensor,
    invstd: Tensor,
    momentum: float,
    eps: float,
) -> None:
    C, S = _shape_key(x)
    if _use_split_train_apply(x, S):
        _batchnorm_stats(
            x, running_mean, running_var, mean, invstd, momentum, eps,
        )
        _batchnorm_apply(x, weight, bias, out, mean, invstd)
        return

    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    running_mean_dtype = (
        _TORCH2CUTE_DTYPE[running_mean.dtype] if running_mean is not None else None
    )
    running_var_dtype = (
        _TORCH2CUTE_DTYPE[running_var.dtype] if running_var is not None else None
    )
    compile_key = (
        dtype,
        out_dtype,
        weight_dtype,
        bias_dtype,
        running_mean_dtype,
        running_var_dtype,
        C,
        S,
    )
    if compile_key not in _batchnorm_fwd.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        out_cute = fake_tensor(out_dtype, (batch_sym, C, S))
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        bias_cute = fake_tensor(bias_dtype, (C,)) if bias_dtype else None
        running_mean_cute = (
            fake_tensor(running_mean_dtype, (C,)) if running_mean_dtype else None
        )
        running_var_cute = (
            fake_tensor(running_var_dtype, (C,)) if running_var_dtype else None
        )
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        _batchnorm_fwd.compile_cache[compile_key] = cute.compile(
            BatchNormFwd(S, _num_threads_for_spatial(S)),
            x_cute,
            weight_cute,
            bias_cute,
            running_mean_cute,
            running_var_cute,
            out_cute,
            mean_cute,
            invstd_cute,
            Float32(0),
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_fwd.compile_cache[compile_key](
        x,
        weight,
        bias,
        running_mean,
        running_var,
        out,
        mean,
        invstd,
        momentum,
        eps,
    )


_batchnorm_fwd.compile_cache = {}


def _batchnorm_eval_apply(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor,
    running_var: Tensor,
    out: Tensor,
    mean: Tensor,
    invstd: Tensor,
    eps: float,
) -> None:
    C, S = _shape_key(x)
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    running_mean_dtype = _TORCH2CUTE_DTYPE[running_mean.dtype]
    running_var_dtype = _TORCH2CUTE_DTYPE[running_var.dtype]
    compile_key = (
        dtype,
        out_dtype,
        weight_dtype,
        bias_dtype,
        running_mean_dtype,
        running_var_dtype,
        C,
        S,
    )
    if compile_key not in _batchnorm_eval_apply.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        out_cute = fake_tensor(out_dtype, (batch_sym, C, S))
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        bias_cute = fake_tensor(bias_dtype, (C,)) if bias_dtype else None
        running_mean_cute = fake_tensor(running_mean_dtype, (C,))
        running_var_cute = fake_tensor(running_var_dtype, (C,))
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        _batchnorm_eval_apply.compile_cache[compile_key] = cute.compile(
            BatchNormEvalApply(S, _num_threads_for_spatial(S)),
            x_cute,
            weight_cute,
            bias_cute,
            running_mean_cute,
            running_var_cute,
            out_cute,
            mean_cute,
            invstd_cute,
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_eval_apply.compile_cache[compile_key](
        x, weight, bias, running_mean, running_var, out, mean, invstd, eps,
    )


_batchnorm_eval_apply.compile_cache = {}


def _batchnorm_eval(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    running_mean: Tensor,
    running_var: Tensor,
    out: Tensor,
    mean: Tensor,
    invstd: Tensor,
    eps: float,
) -> None:
    C, S = _shape_key(x)
    if _use_split_eval_apply(x, S):
        _batchnorm_eval_apply(
            x, weight, bias, running_mean, running_var, out, mean, invstd, eps,
        )
        return

    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    out_dtype = _TORCH2CUTE_DTYPE[out.dtype]
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    bias_dtype = _TORCH2CUTE_DTYPE[bias.dtype] if bias is not None else None
    running_mean_dtype = _TORCH2CUTE_DTYPE[running_mean.dtype]
    running_var_dtype = _TORCH2CUTE_DTYPE[running_var.dtype]
    compile_key = (
        dtype,
        out_dtype,
        weight_dtype,
        bias_dtype,
        running_mean_dtype,
        running_var_dtype,
        C,
        S,
    )
    if compile_key not in _batchnorm_eval.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        out_cute = fake_tensor(out_dtype, (batch_sym, C, S))
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        bias_cute = fake_tensor(bias_dtype, (C,)) if bias_dtype else None
        running_mean_cute = fake_tensor(running_mean_dtype, (C,))
        running_var_cute = fake_tensor(running_var_dtype, (C,))
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        _batchnorm_eval.compile_cache[compile_key] = cute.compile(
            BatchNormEval(S, _num_threads_for_spatial(S)),
            x_cute,
            weight_cute,
            bias_cute,
            running_mean_cute,
            running_var_cute,
            out_cute,
            mean_cute,
            invstd_cute,
            Float32(0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_eval.compile_cache[compile_key](
        x, weight, bias, running_mean, running_var, out, mean, invstd, eps,
    )


_batchnorm_eval.compile_cache = {}


def _batchnorm_bwd(
    x: Tensor,
    weight: Tensor | None,
    dout: Tensor,
    mean: Tensor,
    invstd: Tensor,
    dx: Tensor | None,
    dw: Tensor | None,
    db: Tensor | None,
) -> None:
    C, S = _shape_key(x)
    dtype = _TORCH2CUTE_DTYPE[x.dtype]
    dout_dtype = _TORCH2CUTE_DTYPE[dout.dtype]
    dx_dtype = _TORCH2CUTE_DTYPE[dx.dtype] if dx is not None else None
    weight_dtype = _TORCH2CUTE_DTYPE[weight.dtype] if weight is not None else None
    dw_dtype = _TORCH2CUTE_DTYPE[dw.dtype] if dw is not None else None
    db_dtype = _TORCH2CUTE_DTYPE[db.dtype] if db is not None else None
    compile_key = (
        dtype,
        dout_dtype,
        dx_dtype,
        weight_dtype,
        dw_dtype,
        db_dtype,
        C,
        S,
    )
    if compile_key not in _batchnorm_bwd.compile_cache:
        batch_sym = cute.sym_int()
        x_cute = fake_tensor(dtype, (batch_sym, C, S))
        dout_cute = fake_tensor(dout_dtype, (batch_sym, C, S))
        dx_cute = fake_tensor(dx_dtype, (batch_sym, C, S)) if dx_dtype else None
        weight_cute = fake_tensor(weight_dtype, (C,)) if weight_dtype else None
        mean_cute = fake_tensor(Float32, (C,))
        invstd_cute = fake_tensor(Float32, (C,))
        dw_cute = fake_tensor(dw_dtype, (C,)) if dw_dtype else None
        db_cute = fake_tensor(db_dtype, (C,)) if db_dtype else None
        _batchnorm_bwd.compile_cache[compile_key] = cute.compile(
            BatchNormBackward(S, _num_threads_for_spatial(S)),
            x_cute,
            weight_cute,
            dout_cute,
            mean_cute,
            invstd_cute,
            dx_cute,
            dw_cute,
            db_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _batchnorm_bwd.compile_cache[compile_key](
        x, weight, dout, mean, invstd, dx, dw, db,
    )


_batchnorm_bwd.compile_cache = {}
