from __future__ import annotations

import argparse
import math
import time
from functools import partial
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

import _C
import _gelu_bwd
import _linear_bwd_fused
from tk_bench import input_group_count, profile_groups, print_bench, uniform_bf16


def _is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _register_custom_op(name: str, mutates_args=()):
    try:
        return torch.library.custom_op(name, mutates_args=mutates_args)
    except Exception:
        def decorator(fn):
            return fn
        return decorator


def _register_autograd(op, backward, setup_context) -> None:
    try:
        op.register_autograd(backward, setup_context=setup_context)
    except Exception:
        pass


@_register_custom_op("tk_dit::layernorm_adaln", mutates_args=())
def tk_layernorm_adaln_op(
    flat: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(flat)
    mean = torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    if flat.dtype == torch.bfloat16 and flat.shape[1] == 1024:
        _C.layernorm_adaln_warp4(flat, shift.contiguous(), scale.contiguous(), out, mean, rstd, tokens, eps)
    else:
        _C.layernorm_adaln(flat, shift.contiguous(), scale.contiguous(), out, mean, rstd, tokens, eps)
    return out, mean, rstd


@tk_layernorm_adaln_op.register_fake
def _tk_layernorm_adaln_fake(flat, shift, scale, tokens: int, eps: float):
    return (
        torch.empty_like(flat),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
    )


@_register_custom_op("tk_dit::layernorm_adaln_backward", mutates_args=())
def tk_layernorm_adaln_backward_op(
    grad: torch.Tensor,
    flat: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(flat)
    dshift = torch.empty_like(scale, dtype=torch.float32)
    dscale = torch.empty_like(scale, dtype=torch.float32)
    if flat.dtype == torch.bfloat16 and flat.shape[1] == 1024:
        _C.layernorm_adaln_backward_warp4(grad, flat, scale, mean, rstd, dx, dshift, dscale, tokens)
    else:
        _C.layernorm_adaln_backward(grad, flat, scale, mean, rstd, dx, dshift, dscale, tokens)
    return dx, dshift, dscale


@tk_layernorm_adaln_backward_op.register_fake
def _tk_layernorm_adaln_backward_fake(grad, flat, scale, mean, rstd, tokens: int):
    return (
        torch.empty_like(flat),
        torch.empty_like(scale, dtype=torch.float32),
        torch.empty_like(scale, dtype=torch.float32),
    )


@_register_custom_op("tk_dit::gated_residual", mutates_args=())
def tk_gated_residual_op(
    flat_x: torch.Tensor,
    flat_h: torch.Tensor,
    gate: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    out = torch.empty_like(flat_x)
    _C.gated_residual(flat_x, flat_h, gate.contiguous(), out, tokens)
    return out


@tk_gated_residual_op.register_fake
def _tk_gated_residual_fake(flat_x, flat_h, gate, tokens: int):
    return torch.empty_like(flat_x)


@_register_custom_op("tk_dit::gated_residual_backward_no_dx", mutates_args=())
def tk_gated_residual_backward_no_dx_op(
    grad: torch.Tensor,
    flat_h: torch.Tensor,
    gate: torch.Tensor,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dh = torch.empty_like(grad)
    dgate = torch.empty_like(gate, dtype=torch.float32)
    _C.gated_residual_backward_no_dx(grad, flat_h, gate, dh, dgate, tokens)
    return dh, dgate


@tk_gated_residual_backward_no_dx_op.register_fake
def _tk_gated_residual_backward_no_dx_fake(grad, flat_h, gate, tokens: int):
    return torch.empty_like(grad), torch.empty_like(gate, dtype=torch.float32)


@_register_custom_op("tk_dit::gelu_backward", mutates_args=())
def tk_gelu_backward_op(grad_out: torch.Tensor, preact: torch.Tensor) -> torch.Tensor:
    grad_input = torch.empty_like(preact)
    _gelu_bwd.gelu_backward(grad_out.contiguous(), preact.contiguous(), grad_input)
    return grad_input


@tk_gelu_backward_op.register_fake
def _tk_gelu_backward_fake(grad_out, preact):
    return torch.empty_like(preact)


@_register_custom_op("tk_dit::gelu_bwd_bias", mutates_args=())
def tk_gelu_bwd_bias_op(grad_out: torch.Tensor, preact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dz = torch.empty_like(grad_out)
    db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
    _linear_bwd_fused.gelu_bwd_bias(grad_out.contiguous(), preact.contiguous(), dz, db)
    return dz, db


@tk_gelu_bwd_bias_op.register_fake
def _tk_gelu_bwd_bias_fake(grad_out, preact):
    return torch.empty_like(grad_out), torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)


@_register_custom_op("tk_dit::bias_reduce", mutates_args=())
def tk_bias_reduce_op(grad_out: torch.Tensor) -> torch.Tensor:
    db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
    _linear_bwd_fused.bias_reduce(grad_out.contiguous(), db)
    return db


@tk_bias_reduce_op.register_fake
def _tk_bias_reduce_fake(grad_out):
    return torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)


@_register_custom_op("tk_dit::dw_gemm", mutates_args=())
def tk_dw_gemm_op(grad_out: torch.Tensor, x: torch.Tensor, w_like: torch.Tensor) -> torch.Tensor:
    dw = torch.empty_like(w_like)
    _linear_bwd_fused.dw_gemm(grad_out.contiguous(), x.contiguous(), dw)
    return dw


@tk_dw_gemm_op.register_fake
def _tk_dw_gemm_fake(grad_out, x, w_like):
    return torch.empty_like(w_like)


@_register_custom_op("tk_dit::dx_gemm_native", mutates_args=())
def tk_dx_gemm_native_op(grad_out: torch.Tensor, w: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
    dx = torch.empty_like(x_like)
    _linear_bwd_fused.dx_gemm_native(grad_out.contiguous(), w.contiguous(), dx)
    return dx


@tk_dx_gemm_native_op.register_fake
def _tk_dx_gemm_native_fake(grad_out, w, x_like):
    return torch.empty_like(x_like)


@_register_custom_op("tk_dit::linear_native", mutates_args=())
def tk_linear_native_op(flat: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    _C.gemm_linear_native(flat.contiguous(), w.contiguous(), out, b.contiguous())
    return out


@tk_linear_native_op.register_fake
def _tk_linear_native_fake(flat, w, b):
    return torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)


def _tk_linear_native_setup(ctx, inputs, output) -> None:
    flat, w, b = inputs
    ctx.save_for_backward(flat, w, b)


def _tk_linear_native_backward(ctx, grad_out):
    flat, w, b = ctx.saved_tensors
    grad = grad_out.contiguous()
    dx = tk_dx_gemm_native_op(grad, w.contiguous(), flat)
    dw = tk_dw_gemm_op(grad, flat, w)
    db = tk_bias_reduce_op(grad)
    return dx, dw, db.to(b.dtype)


_register_autograd(
    tk_linear_native_op,
    _tk_linear_native_backward,
    _tk_linear_native_setup,
)


@_register_custom_op("tk_dit::linear_gelu_native", mutates_args=())
def tk_linear_gelu_native_op(flat: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    preact = torch.empty_like(out)
    _C.gemm_custom_native(flat.contiguous(), w.contiguous(), out, b.contiguous(), preact)
    return out, preact


@tk_linear_gelu_native_op.register_fake
def _tk_linear_gelu_native_fake(flat, w, b):
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    return out, torch.empty_like(out)


def _tk_linear_gelu_native_setup(ctx, inputs, output) -> None:
    flat, w, b = inputs
    _out, preact = output
    ctx.save_for_backward(flat, w, b, preact)


def _tk_linear_gelu_native_backward(ctx, grad_out, _grad_preact):
    flat, w, b, preact = ctx.saved_tensors
    grad = grad_out.contiguous()
    dz, db = tk_gelu_bwd_bias_op(grad, preact)
    dx = tk_dx_gemm_native_op(dz, w.contiguous(), flat)
    dw = tk_dw_gemm_op(dz, flat, w)
    return dx, dw, db.to(b.dtype)


_register_autograd(
    tk_linear_gelu_native_op,
    _tk_linear_gelu_native_backward,
    _tk_linear_gelu_native_setup,
)


@_register_custom_op("tk_dit::gated_residual_backward", mutates_args=())
def tk_gated_residual_backward_op(
    grad: torch.Tensor,
    projected: torch.Tensor,
    gate: torch.Tensor,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dresidual = torch.empty_like(grad)
    dprojected = torch.empty_like(grad)
    dgate = torch.empty_like(gate, dtype=torch.float32)
    _C.gated_residual_backward(grad.contiguous(), projected.contiguous(), gate.contiguous(), dresidual, dprojected, dgate, tokens)
    return dresidual, dprojected, dgate


@tk_gated_residual_backward_op.register_fake
def _tk_gated_residual_backward_fake(grad, projected, gate, tokens: int):
    return torch.empty_like(grad), torch.empty_like(grad), torch.empty_like(gate, dtype=torch.float32)


@_register_custom_op("tk_dit::gemm_linear_ln_adaln", mutates_args=())
def tk_gemm_linear_ln_adaln_op(
    flat: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    preact = torch.empty_like(out)
    mean = torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _C.layernorm_stats(flat, mean, rstd, eps)
    _C.gemm_linear_ln_adaln(flat, w, out, b, preact, shift, scale, mean, rstd, tokens)
    return out, preact, mean, rstd


@tk_gemm_linear_ln_adaln_op.register_fake
def _tk_gemm_linear_ln_adaln_fake(flat, w, b, shift, scale, tokens: int, eps: float):
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    return (
        out,
        torch.empty_like(out),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
    )


def _tk_gemm_linear_ln_adaln_setup(ctx, inputs, output) -> None:
    flat, w, b, shift, scale, tokens, eps = inputs
    _out, _preact, mean, rstd = output
    ctx.save_for_backward(flat, w, b, shift, scale, mean, rstd)
    ctx.tokens = tokens
    ctx.eps = eps


def _tk_gemm_linear_ln_adaln_backward(ctx, grad_out, _grad_preact, _grad_mean, _grad_rstd):
    flat, w, b, shift, scale, mean, rstd = ctx.saved_tensors
    grad = grad_out.contiguous()
    z, _, _ = tk_layernorm_adaln_op(flat.contiguous(), shift.contiguous(), scale.contiguous(), ctx.tokens, ctx.eps)
    dw = tk_dw_gemm_op(grad, z, w)
    dz = tk_dx_gemm_native_op(grad, w.contiguous(), flat)
    db = tk_bias_reduce_op(grad)
    dx, dshift, dscale = tk_layernorm_adaln_backward_op(dz, flat, scale, mean, rstd, ctx.tokens)
    return dx, dw, db.to(b.dtype), dshift.to(shift.dtype), dscale.to(scale.dtype), None, None


_register_autograd(
    tk_gemm_linear_ln_adaln_op,
    _tk_gemm_linear_ln_adaln_backward,
    _tk_gemm_linear_ln_adaln_setup,
)


@_register_custom_op("tk_dit::gemm_gelu_ln_adaln", mutates_args=())
def tk_gemm_gelu_ln_adaln_op(
    flat: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    preact = torch.empty_like(out)
    mean = torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _C.layernorm_stats(flat, mean, rstd, eps)
    _C.gemm_gelu_ln_adaln(flat, w, out, b, preact, shift, scale, mean, rstd, tokens)
    return out, preact, mean, rstd


@tk_gemm_gelu_ln_adaln_op.register_fake
def _tk_gemm_gelu_ln_adaln_fake(flat, w, b, shift, scale, tokens: int, eps: float):
    out = torch.empty((flat.shape[0], w.shape[0]), device=flat.device, dtype=flat.dtype)
    return (
        out,
        torch.empty_like(out),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
        torch.empty((flat.shape[0],), device=flat.device, dtype=torch.float32),
    )


def _tk_gemm_gelu_ln_adaln_setup(ctx, inputs, output) -> None:
    flat, w, b, shift, scale, tokens, eps = inputs
    _out, preact, mean, rstd = output
    ctx.save_for_backward(flat, w, b, shift, scale, preact, mean, rstd)
    ctx.tokens = tokens
    ctx.eps = eps


def _tk_gemm_gelu_ln_adaln_backward(ctx, grad_out, _grad_preact, _grad_mean, _grad_rstd):
    flat, w, b, shift, scale, preact, mean, rstd = ctx.saved_tensors
    grad = grad_out.contiguous()
    dz_gelu, db = tk_gelu_bwd_bias_op(grad, preact)
    z, _, _ = tk_layernorm_adaln_op(flat.contiguous(), shift.contiguous(), scale.contiguous(), ctx.tokens, ctx.eps)
    dw = tk_dw_gemm_op(dz_gelu, z, w)
    dz = tk_dx_gemm_native_op(dz_gelu, w.contiguous(), flat)
    dx, dshift, dscale = tk_layernorm_adaln_backward_op(dz, flat, scale, mean, rstd, ctx.tokens)
    return dx, dw, db.to(b.dtype), dshift.to(shift.dtype), dscale.to(scale.dtype), None, None


_register_autograd(
    tk_gemm_gelu_ln_adaln_op,
    _tk_gemm_gelu_ln_adaln_backward,
    _tk_gemm_gelu_ln_adaln_setup,
)


@_register_custom_op("tk_dit::gemm_linear_gated_residual", mutates_args=())
def tk_gemm_linear_gated_residual_op(
    flat_x: torch.Tensor,
    w: torch.Tensor,
    flat_residual: torch.Tensor,
    gate: torch.Tensor,
    b: torch.Tensor,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(flat_residual)
    projected = torch.empty_like(flat_residual)
    _C.gemm_linear_gated_residual(flat_x, w, flat_residual, gate, out, projected, b, tokens)
    return out, projected


@tk_gemm_linear_gated_residual_op.register_fake
def _tk_gemm_linear_gated_residual_fake(flat_x, w, flat_residual, gate, b, tokens: int):
    return torch.empty_like(flat_residual), torch.empty_like(flat_residual)


def _tk_gemm_linear_gated_residual_setup(ctx, inputs, output) -> None:
    flat_x, w, flat_residual, gate, b, tokens = inputs
    _out, projected = output
    ctx.save_for_backward(flat_x, w, gate, b, projected)
    ctx.tokens = tokens


def _tk_gemm_linear_gated_residual_backward(ctx, grad_out, _grad_projected):
    flat_x, w, gate, b, projected = ctx.saved_tensors
    grad = grad_out.contiguous()
    dresidual, dprojected, dgate = tk_gated_residual_backward_op(grad, projected, gate, ctx.tokens)
    dx = dprojected.matmul(w)
    dw = dprojected.transpose(0, 1).matmul(flat_x)
    db = dprojected.sum(dim=0)
    return dx, dw.to(w.dtype), dresidual, dgate.to(gate.dtype), db.to(b.dtype), None


_register_autograd(
    tk_gemm_linear_gated_residual_op,
    _tk_gemm_linear_gated_residual_backward,
    _tk_gemm_linear_gated_residual_setup,
)


_FLASH_ATTN3_FUNC = None


def flash_attn3_func():
    global _FLASH_ATTN3_FUNC
    if _FLASH_ATTN3_FUNC is not None:
        return _FLASH_ATTN3_FUNC
    try:
        import flash_attn_interface

        _FLASH_ATTN3_FUNC = flash_attn_interface.flash_attn_func
        return _FLASH_ATTN3_FUNC
    except ImportError:
        pass
    try:
        from kernels import get_kernel

        fa3_module = get_kernel("kernels-community/flash-attn3", version=1)
        _FLASH_ATTN3_FUNC = fa3_module.flash_attn_func
        return _FLASH_ATTN3_FUNC
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention-3 is unavailable. Install Dao-AILab flash-attention hopper package "
            "or the Hugging Face kernels package with kernels-community/flash-attn3."
        ) from exc


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FusedAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        scale_c = scale.contiguous()
        out, mean, rstd = tk_layernorm_adaln_op(flat, shift.contiguous(), scale_c, tokens, eps)
        ctx.save_for_backward(flat, scale_c, mean, rstd)
        ctx.tokens = tokens
        ctx.shape = x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, scale, mean, rstd = ctx.saved_tensors
        grad = grad_out.reshape_as(x).contiguous()
        dx, dshift, dscale = tk_layernorm_adaln_backward_op(grad, x, scale, mean, rstd, ctx.tokens)
        return dx.reshape(ctx.shape), dshift.to(scale.dtype), dscale.to(scale.dtype), None


def fused_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return FusedAdaLN.apply(x, shift, scale, eps)


def tk_linear_native(flat: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return tk_linear_native_op(flat.contiguous(), w.contiguous(), b.contiguous())


def tk_linear_gelu_native(flat: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out, _preact = tk_linear_gelu_native_op(flat.contiguous(), w.contiguous(), b.contiguous())
    return out


def can_use_ln_adaln_gemm(x: torch.Tensor, w: torch.Tensor) -> bool:
    rows = x.numel() // x.shape[-1]
    dim = x.shape[-1]
    out_features = w.shape[0]
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and rows % 128 == 0
        and dim % 64 == 0
        and out_features % 256 == 0
    )


def can_use_gated_linear(x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, w: torch.Tensor) -> bool:
    rows = x.numel() // x.shape[-1]
    in_features = x.shape[-1]
    out_features = w.shape[0]
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and residual.shape[:-1] == x.shape[:-1]
        and residual.shape[-1] == out_features
        and gate.shape == (x.shape[0], out_features)
        and rows % 128 == 0
        and in_features % 64 == 0
        and out_features % 256 == 0
    )


def recompute_adaln_flat(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float, tokens: int) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out, _, _ = tk_layernorm_adaln_op(flat, shift.contiguous(), scale.contiguous(), tokens, eps)
    return out


class FusedAdaLNLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float):
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        shift_c = shift.contiguous()
        scale_c = scale.contiguous()
        w_c = w.contiguous()
        b_c = b.contiguous()
        out, _preact, mean, rstd = tk_gemm_linear_ln_adaln_op(flat, w_c, b_c, shift_c, scale_c, tokens, eps)
        ctx.save_for_backward(flat, shift_c, scale_c, w_c, mean, rstd)
        ctx.tokens = tokens
        ctx.shape = x.shape
        ctx.eps = eps
        return out.reshape(batch, tokens, w_c.shape[0])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, shift, scale, w, mean, rstd = ctx.saved_tensors
        grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        z = recompute_adaln_flat(x.reshape(ctx.shape), shift, scale, ctx.eps, ctx.tokens)
        dw = torch.empty_like(w)
        dz = torch.empty_like(x)
        db = torch.empty((grad.shape[1],), device=grad.device, dtype=torch.float32)
        _linear_bwd_fused.dw_gemm(grad, z, dw)
        _linear_bwd_fused.dx_gemm_native(grad, w.contiguous(), dz)
        _linear_bwd_fused.bias_reduce(grad, db)
        dx = torch.empty_like(x)
        dshift = torch.empty_like(scale, dtype=torch.float32)
        dscale = torch.empty_like(scale, dtype=torch.float32)
        _C.layernorm_adaln_backward(dz, x, scale, mean, rstd, dx, dshift, dscale, ctx.tokens)
        return dx.reshape(ctx.shape), dshift.to(scale.dtype), dscale.to(scale.dtype), dw, db.to(grad.dtype), None


class FusedAdaLNLinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float):
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        shift_c = shift.contiguous()
        scale_c = scale.contiguous()
        w_c = w.contiguous()
        b_c = b.contiguous()
        out, preact, mean, rstd = tk_gemm_gelu_ln_adaln_op(flat, w_c, b_c, shift_c, scale_c, tokens, eps)
        ctx.save_for_backward(flat, shift_c, scale_c, w_c, preact, mean, rstd)
        ctx.tokens = tokens
        ctx.shape = x.shape
        ctx.eps = eps
        return out.reshape(batch, tokens, w_c.shape[0])

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, shift, scale, w, preact, mean, rstd = ctx.saved_tensors
        grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        dz_gelu = torch.empty_like(grad)
        db = torch.empty((grad.shape[1],), device=grad.device, dtype=torch.float32)
        _linear_bwd_fused.gelu_bwd_bias(grad, preact, dz_gelu, db)
        z = recompute_adaln_flat(x.reshape(ctx.shape), shift, scale, ctx.eps, ctx.tokens)
        dw = torch.empty_like(w)
        dz = torch.empty_like(x)
        _linear_bwd_fused.dw_gemm(dz_gelu, z, dw)
        _linear_bwd_fused.dx_gemm_native(dz_gelu, w.contiguous(), dz)
        dx = torch.empty_like(x)
        dshift = torch.empty_like(scale, dtype=torch.float32)
        dscale = torch.empty_like(scale, dtype=torch.float32)
        _C.layernorm_adaln_backward(dz, x, scale, mean, rstd, dx, dshift, dscale, ctx.tokens)
        return dx.reshape(ctx.shape), dshift.to(scale.dtype), dscale.to(scale.dtype), dw, db.to(grad.dtype), None


def fused_adaln_linear(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    linear: nn.Linear,
    eps: float,
) -> torch.Tensor:
    if can_use_ln_adaln_gemm(x, linear.weight):
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        out, _, _, _ = tk_gemm_linear_ln_adaln_op(
            flat,
            linear.weight.contiguous(),
            linear.bias.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            tokens,
            eps,
        )
        return out.reshape(batch, tokens, linear.weight.shape[0])
    return torch.nn.functional.linear(fused_adaln(x, shift, scale, eps), linear.weight, linear.bias)


def fused_adaln_linear_gelu(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    linear: nn.Linear,
    eps: float,
) -> torch.Tensor:
    out_features = linear.weight.shape[0]
    in_features = x.shape[-1]
    use_inline_gemm = can_use_ln_adaln_gemm(x, linear.weight) and out_features <= 2 * in_features
    if use_inline_gemm:
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        out, _, _, _ = tk_gemm_gelu_ln_adaln_op(
            flat,
            linear.weight.contiguous(),
            linear.bias.contiguous(),
            shift.contiguous(),
            scale.contiguous(),
            tokens,
            eps,
        )
        return out.reshape(batch, tokens, linear.weight.shape[0])
    # For wide projections such as MLP fc1 (D -> 4D), applying AdaLN inside
    # each GEMM output tile repeats the same input transform many times. We
    # materialize AdaLN once, then keep GEMM+GELU on the custom TK path.
    z = fused_adaln(x, shift, scale, eps).reshape(-1, in_features)
    out = tk_linear_gelu_native(z, linear.weight, linear.bias)
    return out.reshape(*x.shape[:-1], out_features)


class FusedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        flat_x = x.reshape(batch * tokens, dim).contiguous()
        flat_h = h.reshape(batch * tokens, dim).contiguous()
        gate_c = gate.contiguous()
        out = tk_gated_residual_op(flat_x, flat_h, gate_c, tokens)
        ctx.save_for_backward(flat_h, gate_c)
        ctx.tokens = tokens
        ctx.shape = x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h, gate = ctx.saved_tensors
        grad = grad_out.reshape_as(h).contiguous()
        dh, dgate = tk_gated_residual_backward_no_dx_op(grad, h, gate, ctx.tokens)
        return grad_out, dh.reshape(ctx.shape), dgate.to(gate.dtype)


def gated_residual(x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and h.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16:
        return FusedGatedResidual.apply(x, h, gate)
    return x + gate.unsqueeze(1).to(h.dtype) * h


class FusedLinearGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        batch, tokens, _ = x.shape
        flat_x = x.reshape(batch * tokens, x.shape[-1]).contiguous()
        flat_residual = residual.reshape(batch * tokens, residual.shape[-1]).contiguous()
        gate_c = gate.contiguous()
        w_c = w.contiguous()
        b_c = b.contiguous()
        out, projected = tk_gemm_linear_gated_residual_op(flat_x, w_c, flat_residual, gate_c, b_c, tokens)
        ctx.save_for_backward(flat_x, gate_c, w_c, projected)
        ctx.x_shape = x.shape
        ctx.residual_shape = residual.shape
        ctx.tokens = tokens
        return out.reshape_as(residual)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, gate, w, projected = ctx.saved_tensors
        grad = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
        dresidual = torch.empty_like(grad)
        dprojected = torch.empty_like(grad)
        dgate = torch.empty_like(gate, dtype=torch.float32)
        _C.gated_residual_backward(grad, projected, gate, dresidual, dprojected, dgate, ctx.tokens)
        dx = dprojected.matmul(w)
        dw = dprojected.transpose(0, 1).matmul(x)
        db = dprojected.sum(dim=0)
        return dx.reshape(ctx.x_shape), dresidual.reshape(ctx.residual_shape), dgate.to(gate.dtype), dw.to(w.dtype), db.to(grad.dtype)


def fused_linear_gated_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    linear: nn.Linear,
) -> torch.Tensor:
    if can_use_gated_linear(x, residual, gate, linear.weight):
        batch, tokens, _ = x.shape
        out, _ = tk_gemm_linear_gated_residual_op(
            x.reshape(batch * tokens, x.shape[-1]).contiguous(),
            linear.weight.contiguous(),
            residual.reshape(batch * tokens, residual.shape[-1]).contiguous(),
            gate.contiguous(),
            linear.bias.contiguous(),
            tokens,
        )
        return out.reshape_as(residual)
    return gated_residual(residual, linear(x), gate)


def linear_then_gated_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gate: torch.Tensor,
    linear: nn.Linear,
) -> torch.Tensor:
    return gated_residual(residual, torch.nn.functional.linear(x, linear.weight, linear.bias), gate)


class TkGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.nn.functional.gelu(x, approximate="tanh")

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        return tk_gelu_backward_op(grad_out.contiguous(), x.contiguous())


class TkLinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out, preact = tk_linear_gelu_native_op(x.contiguous(), w.contiguous(), b.contiguous())
        ctx.save_for_backward(x.contiguous(), w, preact)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w, preact = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dz, db = tk_gelu_bwd_bias_op(grad_out, preact)
        dw = tk_dw_gemm_op(dz, x, w)
        dx = tk_dx_gemm_native_op(dz, w.contiguous(), x)
        return dx, dw, db.to(grad_out.dtype)


class TkLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        ctx.save_for_backward(x_c, w)
        return tk_linear_native_op(x_c, w.contiguous(), b.contiguous())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dw = tk_dw_gemm_op(grad_out, x, w)
        dx = tk_dx_gemm_native_op(grad_out, w.contiguous(), x)
        db = tk_bias_reduce_op(grad_out)
        return dx, dw, db.to(grad_out.dtype)


class TkMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        h = tk_linear_gelu_native(flat, self.fc1.weight, self.fc1.bias)
        out = tk_linear_native(h, self.fc2.weight, self.fc2.bias)
        return out.reshape(shape)

    def forward_from_adaln(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        h = fused_adaln_linear_gelu(x, shift, scale, self.fc1, eps)
        shape = h.shape
        out = tk_linear_native(h.reshape(-1, shape[-1]).contiguous(), self.fc2.weight, self.fc2.bias)
        return out.reshape(x.shape)

    def forward_from_adaln_residual(
        self,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        # The native TK MLP backward path is only a win for the large-D
        # microbenchmarks it was tuned on. For DiT-S (D=384, hidden=1536),
        # the custom weight-gradient GEMM is slower than the compiled
        # cuBLAS/Inductor path, so keep GEMMs on torch and only use the
        # faster standalone GELU backward.
        if residual.shape[0] * residual.shape[1] >= 8192 and residual.shape[-1] < 1024:
            mlp_in = fused_adaln(residual, shift, scale, eps)
            h = TkGelu.apply(torch.nn.functional.linear(mlp_in, self.fc1.weight, self.fc1.bias))
            return gated_residual(
                residual,
                torch.nn.functional.linear(h, self.fc2.weight, self.fc2.bias),
                gate,
            )
        h = fused_adaln_linear_gelu(residual, shift, scale, self.fc1, eps)
        return fused_linear_gated_residual(h, residual, gate, self.fc2)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return fused_linear_gated_residual(h, residual, gate, self.fc2)

    def forward_residual_epilogue(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return linear_then_gated_residual(h, residual, gate, self.fc2)


class FusedInputMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x), approximate="tanh"))

    def forward_from_adaln(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        return self.fc2(fused_adaln_linear_gelu(x, shift, scale, self.fc1, eps))

    def forward_from_adaln_residual(
        self,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        h = fused_adaln_linear_gelu(residual, shift, scale, self.fc1, eps)
        return fused_linear_gated_residual(h, residual, gate, self.fc2)

    def forward_residual_epilogue(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
        return linear_then_gated_residual(h, residual, gate, self.fc2)


class SdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _attention_from_qkv(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, tokens, _, _, _ = qkv.shape
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return out.transpose(1, 2).reshape(batch, tokens, self.num_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        return self.proj(self._attention_from_qkv(qkv))

    def forward_from_adaln(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = fused_adaln_linear(x, shift, scale, self.qkv, eps).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        return self.proj(self._attention_from_qkv(qkv))

    def forward_from_adaln_residual(
        self,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        batch, tokens, _ = residual.shape
        qkv = fused_adaln_linear(residual, shift, scale, self.qkv, eps).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        attn = self._attention_from_qkv(qkv)
        return fused_linear_gated_residual(attn, residual, gate, self.proj)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        attn = self._attention_from_qkv(qkv)
        return fused_linear_gated_residual(attn, residual, gate, self.proj)

    def forward_residual_epilogue(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        attn = self._attention_from_qkv(qkv)
        return linear_then_gated_residual(attn, residual, gate, self.proj)


class FlashAttention3(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return self.proj(out.reshape(batch, tokens, dim))

    def forward_from_adaln(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = fused_adaln_linear(x, shift, scale, self.qkv, eps).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return self.proj(out.reshape(batch, tokens, dim))

    def forward_from_adaln_residual(
        self,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        batch, tokens, dim = residual.shape
        qkv = fused_adaln_linear(residual, shift, scale, self.qkv, eps).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return fused_linear_gated_residual(out.reshape(batch, tokens, dim), residual, gate, self.proj)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return fused_linear_gated_residual(out.reshape(batch, tokens, dim), residual, gate, self.proj)

    def forward_residual_epilogue(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return linear_then_gated_residual(out.reshape(batch, tokens, dim), residual, gate, self.proj)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (d, h, w)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(t_freq)))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        fused_adaln_enabled=False,
        fused_residual_enabled=False,
        tk_mlp_enabled=False,
        fused_input_projection_enabled=False,
        fused_output_projection_enabled=False,
        fused_epilogue_only_enabled=False,
        attention_backend="timm",
        **block_kwargs,
    ):
        super().__init__()
        self.fused_adaln_enabled = fused_adaln_enabled
        self.fused_residual_enabled = fused_residual_enabled
        self.fused_input_projection_enabled = fused_input_projection_enabled
        self.fused_output_projection_enabled = fused_output_projection_enabled
        self.fused_epilogue_only_enabled = fused_epilogue_only_enabled
        self.tk_mlp_enabled = tk_mlp_enabled
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attention_backend == "fa3":
            self.attn = FlashAttention3(hidden_size, num_heads=num_heads, qkv_bias=True)
        elif attention_backend == "timm":
            self.attn = (
                SdpaAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
                if fused_input_projection_enabled or fused_output_projection_enabled
                else Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
            )
        else:
            raise ValueError(f"unknown attention backend: {attention_backend}")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = (
            TkMlp(hidden_size, mlp_hidden_dim)
            if tk_mlp_enabled
            else FusedInputMlp(hidden_size, mlp_hidden_dim)
            if fused_input_projection_enabled or fused_output_projection_enabled
            else Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
                drop=0,
            )
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if self.fused_adaln_enabled:
            if (
                self.fused_input_projection_enabled
                and self.fused_output_projection_enabled
                and self.fused_residual_enabled
                and hasattr(self.attn, "forward_from_adaln_residual")
            ):
                x = self.attn.forward_from_adaln_residual(x, shift_msa, scale_msa, gate_msa, self.norm1.eps)
            elif self.fused_input_projection_enabled and hasattr(self.attn, "forward_from_adaln"):
                attn_out = self.attn.forward_from_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = gated_residual(x, attn_out, gate_msa) if self.fused_residual_enabled else x + gate_msa.unsqueeze(1) * attn_out
            elif (
                self.fused_epilogue_only_enabled
                and self.fused_residual_enabled
                and hasattr(self.attn, "forward_residual_epilogue")
            ):
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = self.attn.forward_residual_epilogue(attn_in, x, gate_msa)
            elif self.fused_output_projection_enabled and self.fused_residual_enabled and hasattr(self.attn, "forward_residual"):
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = self.attn.forward_residual(attn_in, x, gate_msa)
            else:
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                attn_out = self.attn(attn_in)
                x = gated_residual(x, attn_out, gate_msa) if self.fused_residual_enabled else x + gate_msa.unsqueeze(1) * attn_out
            if (
                self.fused_residual_enabled
                and self.tk_mlp_enabled
                and hasattr(self.mlp, "forward_from_adaln_residual")
            ):
                x = self.mlp.forward_from_adaln_residual(x, shift_mlp, scale_mlp, gate_mlp, self.norm2.eps)
            elif (
                self.fused_input_projection_enabled
                and self.fused_output_projection_enabled
                and self.fused_residual_enabled
                and hasattr(self.mlp, "forward_from_adaln_residual")
            ):
                x = self.mlp.forward_from_adaln_residual(x, shift_mlp, scale_mlp, gate_mlp, self.norm2.eps)
            elif self.fused_input_projection_enabled and hasattr(self.mlp, "forward_from_adaln"):
                mlp_out = self.mlp.forward_from_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = gated_residual(x, mlp_out, gate_mlp) if self.fused_residual_enabled else x + gate_mlp.unsqueeze(1) * mlp_out
            elif (
                self.fused_epilogue_only_enabled
                and self.fused_residual_enabled
                and hasattr(self.mlp, "forward_residual_epilogue")
            ):
                mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = self.mlp.forward_residual_epilogue(mlp_in, x, gate_mlp)
            elif self.fused_output_projection_enabled and self.fused_residual_enabled and hasattr(self.mlp, "forward_residual"):
                mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = self.mlp.forward_residual(mlp_in, x, gate_mlp)
            else:
                mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                mlp_out = self.mlp(mlp_in)
                x = gated_residual(x, mlp_out, gate_mlp) if self.fused_residual_enabled else x + gate_mlp.unsqueeze(1) * mlp_out
            return x
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, fused_adaln_enabled=False):
        super().__init__()
        self.fused_adaln_enabled = fused_adaln_enabled
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        if self.fused_adaln_enabled:
            x = fused_adaln(x, shift, scale, self.norm_final.eps)
        else:
            x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_class_condition=False,
        fused_adaln_enabled=False,
        fused_residual_enabled=False,
        tk_mlp_enabled=False,
        fused_input_projection_enabled=False,
        fused_output_projection_enabled=False,
        fused_epilogue_only_enabled=False,
        attention_backend="timm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.use_class_condition = use_class_condition
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) if use_class_condition else None
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                fused_adaln_enabled=fused_adaln_enabled,
                fused_residual_enabled=fused_residual_enabled,
                tk_mlp_enabled=tk_mlp_enabled,
                fused_input_projection_enabled=fused_input_projection_enabled,
                fused_output_projection_enabled=fused_output_projection_enabled,
                fused_epilogue_only_enabled=fused_epilogue_only_enabled,
                attention_backend=attention_backend,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, fused_adaln_enabled=fused_adaln_enabled)
        self.register_buffer("_pos_embed", torch.empty(0), persistent=False)
        self._pos_shape = None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.fc1.weight, std=0.02)
        nn.init.normal_(self.t_embedder.fc2.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def randomize_zero_init_layers(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
        nn.init.xavier_uniform_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].bias, std=0.02)
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.normal_(self.final_layer.linear.bias, std=0.02)

    def pos_embed(self, spatial_shape, dtype, device):
        if self._pos_shape != spatial_shape or self._pos_embed.numel() == 0:
            pos_embed = get_3d_sincos_pos_embed(self.x_embedder.proj.out_channels, spatial_shape)
            self._pos_embed = torch.from_numpy(pos_embed).to(device=device, dtype=dtype).unsqueeze(0)
            self._pos_shape = spatial_shape
        return self._pos_embed

    def unpatchify(self, x, spatial_shape):
        c = self.out_channels
        p = self.patch_size
        d, h, w = spatial_shape
        x = x.reshape(x.shape[0], d, h, w, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(x.shape[0], c, d * p, h * p, w * p)

    def forward(self, x, t, y=None):
        x, spatial_shape = self.x_embedder(x)
        x = x + self.pos_embed(spatial_shape, x.dtype, x.device)
        t = self.t_embedder(t)
        if self.use_class_condition:
            assert y is not None
            c = t + self.y_embedder(y, self.training)
        else:
            c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x, spatial_shape)


def get_3d_sincos_pos_embed(embed_dim, grid_size_dhw):
    d, h, w = grid_size_dhw
    dim_each = (embed_dim // 6) * 2
    dims = [dim_each, dim_each, dim_each]
    for i in range((embed_dim - sum(dims)) // 2):
        dims[i % 3] += 2
    assert sum(dims) == embed_dim and all(dim % 2 == 0 for dim in dims)
    emb_d = get_1d_sincos_pos_embed_from_grid(dims[0], np.arange(d, dtype=np.float32))
    emb_h = get_1d_sincos_pos_embed_from_grid(dims[1], np.arange(h, dtype=np.float32))
    emb_w = get_1d_sincos_pos_embed_from_grid(dims[2], np.arange(w, dtype=np.float32))
    emb_d = np.broadcast_to(emb_d[:, None, None, :], (d, h, w, dims[0])).copy()
    emb_h = np.broadcast_to(emb_h[None, :, None, :], (d, h, w, dims[1])).copy()
    emb_w = np.broadcast_to(emb_w[None, None, :, :], (d, h, w, dims[2])).copy()
    return np.concatenate([emb_d, emb_h, emb_w], axis=-1).reshape(d * h * w, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def dit_config(name: str):
    configs = {
        "S": dict(depth=12, hidden_size=384, patch_size=1, num_heads=6, in_channels=4),
        "L": dict(depth=24, hidden_size=1024, patch_size=1, num_heads=16, in_channels=32),
        "XL": dict(depth=28, hidden_size=1152, patch_size=1, num_heads=16, in_channels=4),
    }
    return configs[name]


def make_model(
    name: str,
    fused: bool,
    fused_residual: bool = False,
    tk_mlp: bool = False,
    fused_input_projection: bool = False,
    fused_output_projection: bool = False,
    attention_backend: str = "timm",
) -> DiT:
    torch.manual_seed(123)
    model = DiT(
        **dit_config(name),
        fused_adaln_enabled=fused,
        fused_residual_enabled=fused_residual,
        tk_mlp_enabled=tk_mlp,
        fused_input_projection_enabled=fused_input_projection,
        fused_output_projection_enabled=fused_output_projection,
        attention_backend=attention_backend,
    ).cuda().to(torch.bfloat16).train()
    model.randomize_zero_init_layers()
    return model


def variant_config(variant_name: str):
    variants = {
        "eager": (False, False, False, False, False, "timm", False),
        "compile": (False, False, False, False, False, "timm", True),
        "compile_fused_adaln": (True, False, False, False, False, "timm", True),
        "compile_tk_adaln_only": (True, False, False, False, False, "timm", True),
        "fused_adaln_residual": (True, True, False, False, False, "timm", False),
        "compile_fused_adaln_residual": (True, True, False, False, False, "timm", True),
        "compile_tk_adaln_residual_only": (True, True, False, False, False, "timm", True),
        "fused_adaln_residual_tk_mlp": (True, True, True, False, False, "timm", False),
        "compile_fused_adaln_residual_tk_mlp": (True, True, True, False, False, "timm", True),
        "fa3_attn": (False, False, False, False, False, "fa3", False),
        "compile_fa3_attn": (False, False, False, False, False, "fa3", True),
        "fused_adaln_residual_fa3": (True, True, False, False, False, "fa3", False),
        "compile_fused_adaln_residual_fa3": (True, True, False, False, False, "fa3", True),
        "fused_adaln_residual_fa3_tk_mlp": (True, True, True, False, False, "fa3", False),
        "compile_fused_adaln_residual_fa3_tk_mlp": (True, True, True, False, False, "fa3", True),
    }
    if variant_name not in variants:
        raise ValueError(f"unknown profile variant: {variant_name}")
    return variants[variant_name]


def clone_state(dst: nn.Module, src: nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=False)


def make_group(batch: int, channels: int, spatial: tuple[int, int, int], seed: int):
    x = uniform_bf16((batch, channels, *spatial), seed, -1.0, 1.0).requires_grad_(True)
    t = torch.randint(0, 1000, (batch,), device="cuda", dtype=torch.long)
    grad = uniform_bf16((batch, channels, *spatial), seed + 1, -1.0, 1.0)
    return x, t, grad


def train_step(model: nn.Module, group):
    x, t, grad = group
    out = model(x, t)
    out.backward(grad)
    model.zero_grad(set_to_none=True)
    x.grad = None


def memory_probe(model: nn.Module, group, label: str):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        train_step(model, group)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(f"{label}: PASS peak_allocated={peak / 2**30:.2f} GiB peak_reserved={reserved / 2**30:.2f} GiB")
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(f"{label}: OOM peak_allocated={peak / 2**30:.2f} GiB peak_reserved={reserved / 2**30:.2f} GiB error={exc}")
        torch.cuda.empty_cache()


def train_step_grad(model: nn.Module, x: torch.Tensor, t: torch.Tensor, grad: torch.Tensor):
    params = tuple(p for p in model.parameters() if p.requires_grad)
    out = model(x, t)
    grads = torch.autograd.grad(out, (x, *params), grad, allow_unused=True)
    return grads


def profile_variant_case(
    model_name: str,
    variant_name: str,
    batch: int,
    spatial: tuple[int, int, int],
    warmup: int,
    iters: int,
    rows: int,
) -> None:
    fused, fused_residual, tk_mlp, fused_input_projection, fused_output_projection, attention_backend, compiled = variant_config(variant_name)
    model = make_model(
        model_name,
        fused=fused,
        fused_residual=fused_residual,
        tk_mlp=tk_mlp,
        fused_input_projection=fused_input_projection,
        fused_output_projection=fused_output_projection,
        attention_backend=attention_backend,
    )
    model.pos_embed(spatial, torch.bfloat16, torch.device("cuda"))
    if compiled:
        model = torch.compile(model)
    group = make_group(batch, dit_config(model_name)["in_channels"], spatial, 77000)
    for _ in range(warmup):
        train_step(model, group)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            train_step(model, group)
    torch.cuda.synchronize()
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(f"\nTorch profiler DiT-{model_name} {variant_name} B{batch} T{tokens} warmup={warmup} iters={iters}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows))


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1.2e-1, rtol: float = 1.2e-1) -> bool:
    if actual is None or expected is None:
        ok = actual is None and expected is None
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        return ok
    a = actual.detach().float()
    e = expected.detach().float()
    max_abs = (a - e).abs().max().item()
    denom = e.abs().clamp_min(1e-6)
    max_rel = ((a - e).abs() / denom).max().item()
    ok = torch.allclose(a, e, atol=atol, rtol=rtol)
    print(f"  {name}: {'PASS' if ok else 'FAIL'} max_abs={max_abs:.4e} max_rel={max_rel:.4e}")
    return ok


def fused_input_projection_correctness(batch: int = 2, tokens: int = 64, dim: int = 1024, hidden_dim: int = 4096) -> bool:
    print(f"\nFused LN+AdaLN+projection correctness B{batch} T{tokens} D{dim}")
    torch.manual_seed(1234)
    x = uniform_bf16((batch, tokens, dim), 9000, -2.0, 2.0).requires_grad_(True)
    shift = uniform_bf16((batch, dim), 9001, -0.5, 0.5).requires_grad_(True)
    scale = uniform_bf16((batch, dim), 9002, -0.25, 0.25).requires_grad_(True)
    w_qkv = uniform_bf16((3 * dim, dim), 9003, -0.02, 0.02).requires_grad_(True)
    b_qkv = uniform_bf16((3 * dim,), 9004, -0.02, 0.02).requires_grad_(True)
    w_fc1 = uniform_bf16((hidden_dim, dim), 9005, -0.02, 0.02).requires_grad_(True)
    b_fc1 = uniform_bf16((hidden_dim,), 9006, -0.02, 0.02).requires_grad_(True)
    grad_qkv = uniform_bf16((batch, tokens, 3 * dim), 9007, -1.0, 1.0)
    grad_fc1 = uniform_bf16((batch, tokens, hidden_dim), 9008, -1.0, 1.0)

    def clone(t: torch.Tensor) -> torch.Tensor:
        out = t.detach().clone()
        out.requires_grad_(t.requires_grad)
        return out

    ref = [clone(t) for t in (x, shift, scale, w_qkv, b_qkv)]
    z = modulate(torch.nn.functional.layer_norm(ref[0].float(), (dim,), None, None, 1e-6), ref[1], ref[2]).to(torch.bfloat16)
    ref_out = torch.nn.functional.linear(z, ref[3], ref[4])
    ref_out.backward(grad_qkv)

    fused = [clone(t) for t in (x, shift, scale, w_qkv, b_qkv)]
    fused_out = FusedAdaLNLinear.apply(fused[0], fused[1], fused[2], fused[3], fused[4], 1e-6)
    fused_out.backward(grad_qkv)

    ok = check_close("linear output", fused_out, ref_out)
    for name, actual, expected in zip(("x", "shift", "scale", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"linear d{name}", actual, expected) and ok

    ref = [clone(t) for t in (x, shift, scale, w_fc1, b_fc1)]
    z = modulate(torch.nn.functional.layer_norm(ref[0].float(), (dim,), None, None, 1e-6), ref[1], ref[2]).to(torch.bfloat16)
    ref_out = torch.nn.functional.gelu(torch.nn.functional.linear(z, ref[3], ref[4]), approximate="tanh")
    ref_out.backward(grad_fc1)

    fused = [clone(t) for t in (x, shift, scale, w_fc1, b_fc1)]
    fused_out = FusedAdaLNLinearGelu.apply(fused[0], fused[1], fused[2], fused[3], fused[4], 1e-6)
    fused_out.backward(grad_fc1)

    ok = check_close("gelu output", fused_out, ref_out) and ok
    for name, actual, expected in zip(("x", "shift", "scale", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"gelu d{name}", actual, expected) and ok

    h_fc2 = uniform_bf16((batch, tokens, hidden_dim), 9010, -1.0, 1.0).requires_grad_(True)
    residual = uniform_bf16((batch, tokens, dim), 9011, -1.0, 1.0).requires_grad_(True)
    gate = uniform_bf16((batch, dim), 9012, -0.25, 0.25).requires_grad_(True)
    w_fc2 = uniform_bf16((dim, hidden_dim), 9013, -0.02, 0.02).requires_grad_(True)
    b_fc2 = uniform_bf16((dim,), 9014, -0.02, 0.02).requires_grad_(True)
    grad_res = uniform_bf16((batch, tokens, dim), 9015, -1.0, 1.0)

    ref = [clone(t) for t in (h_fc2, residual, gate, w_fc2, b_fc2)]
    ref_out = ref[1] + ref[2].unsqueeze(1) * torch.nn.functional.linear(ref[0], ref[3], ref[4])
    ref_out.backward(grad_res)

    fused = [clone(t) for t in (h_fc2, residual, gate, w_fc2, b_fc2)]
    fused_out = FusedLinearGatedResidual.apply(fused[0], fused[1], fused[2], fused[3], fused[4])
    fused_out.backward(grad_res)

    ok = check_close("linear+residual output", fused_out, ref_out) and ok
    for name, actual, expected in zip(("x", "residual", "gate", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"linear+residual d{name}", actual, expected) and ok

    print(f"Fused LN+AdaLN/projection epilogue correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def residual_group(batch: int, tokens: int, dim: int, seed: int):
    x = uniform_bf16((batch, tokens, dim), seed, -1.0, 1.0)
    h = uniform_bf16((batch, tokens, dim), seed + 1, -1.0, 1.0)
    gate = uniform_bf16((batch, dim), seed + 2, -0.5, 0.5)
    grad = uniform_bf16((batch, tokens, dim), seed + 3, -1.0, 1.0)
    return (
        x,
        h,
        gate,
        grad,
        torch.empty_like(x),
        torch.empty_like(h),
        torch.empty((batch, dim), device="cuda", dtype=torch.float32),
    )


def residual_torch_step(group):
    x, h, gate, grad, out, dh, dgate = group
    out.copy_(x + gate.unsqueeze(1) * h)
    dh.copy_(grad * gate.unsqueeze(1))
    dgate.copy_((grad.float() * h.float()).sum(dim=1))


def residual_fused_step(group):
    x, h, gate, grad, out, dh, dgate = group
    batch, tokens, dim = x.shape
    _C.gated_residual(
        x.reshape(batch * tokens, dim),
        h.reshape(batch * tokens, dim),
        gate,
        out.reshape(batch * tokens, dim),
        tokens,
    )
    _C.gated_residual_backward_no_dx(
        grad.reshape(batch * tokens, dim),
        h.reshape(batch * tokens, dim),
        gate,
        dh.reshape(batch * tokens, dim),
        dgate,
        tokens,
    )


def residual_correctness(batch: int = 2, tokens: int = 64, dim: int = 1024) -> bool:
    print(f"\nStandalone gated residual correctness B{batch} T{tokens} D{dim}")
    ref = residual_group(batch, tokens, dim, 9100)
    fused = tuple(t.clone() if i < 4 else torch.empty_like(t) for i, t in enumerate(ref))
    residual_torch_step(ref)
    residual_fused_step(fused)
    ok = check_close("residual output", fused[4], ref[4])
    ok = check_close("residual dh", fused[5], ref[5]) and ok
    ok = check_close("residual dgate", fused[6], ref[6]) and ok
    print(f"Standalone gated residual correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_residual(tokens_list: list[int], batches: list[int], dim: int, warmup: int, iters: int):
    if not residual_correctness(tokens=max(64, min(tokens_list))):
        raise SystemExit(1)
    for tokens in tokens_list:
        for batch in batches:
            input_bytes = batch * tokens * dim * 2 * 4 + batch * dim * 2
            groups_n = min(input_group_count(input_bytes), 8)
            torch_groups = [residual_group(batch, tokens, dim, 9200 + i * 10) for i in range(groups_n)]
            fused_groups = [residual_group(batch, tokens, dim, 9300 + i * 10) for i in range(groups_n)]
            # x,h,gate,grad reads + out,dh,dgate writes. dx is an alias of grad and does not need materialization.
            bytes_moved = batch * tokens * dim * 2 * 6 + batch * dim * (2 + 4)
            torch_result = profile_groups(
                f"residual torch B{batch} T{tokens} D{dim}",
                torch_groups,
                residual_torch_step,
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_moved,
            )
            fused_result = profile_groups(
                f"residual fused B{batch} T{tokens} D{dim}",
                fused_groups,
                residual_fused_step,
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_moved,
            )
            print_bench(torch_result)
            print_bench(fused_result)
            print(f"  speedup: {torch_result.us / fused_result.us:.2f}x")


def bench_case(
    model_name: str,
    batch: int,
    spatial: tuple[int, int, int],
    include_compile: bool,
    warmup: int,
    iters: int,
    include_fa3: bool,
    only_variants: set[str] | None = None,
):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    input_bytes = batch * cfg["in_channels"] * tokens * 2
    groups_n = min(input_group_count(input_bytes), 4)
    groups = [make_group(batch, cfg["in_channels"], spatial, 50000 + i * 10) for i in range(groups_n)]

    print(f"\n3D DiT-{model_name}/1 E2E train: batch={batch} tokens={tokens} spatial={spatial} groups={groups_n}")
    variants = [("eager", False, False, False, False, False, "timm", False)]
    if include_compile:
        variants.append(("compile", False, False, False, False, False, "timm", True))
    variants.extend([
        ("tk_mlp", False, False, True, False, False, "timm", False),
        ("fused_adaln", True, False, False, False, False, "timm", False),
        ("fused_adaln_residual", True, True, False, False, False, "timm", False),
        ("fused_adaln_residual_tk_mlp", True, True, True, False, False, "timm", False),
        ("fused_output_proj", True, True, False, False, True, "timm", False),
        ("fused_input_proj", True, True, False, True, False, "timm", False),
        ("fused_input_output_proj", True, True, False, True, True, "timm", False),
        ("fused_input_proj_tk_mlp", True, True, True, True, False, "timm", False),
        ("fused_input_output_proj_tk_mlp", True, True, True, True, True, "timm", False),
    ])
    if include_compile:
        variants.extend([
            ("compile_fused_adaln", True, False, False, False, False, "timm", True),
            ("compile_tk_adaln_only", True, False, False, False, False, "timm", True),
            ("compile_fused_adaln_residual", True, True, False, False, False, "timm", True),
            ("compile_tk_adaln_residual_only", True, True, False, False, False, "timm", True),
            ("compile_fused_adaln_tk_mlp", True, False, True, False, False, "timm", True),
            ("compile_fused_adaln_residual_tk_mlp", True, True, True, False, False, "timm", True),
        ])
    if include_fa3:
        variants.extend([
            ("fa3_attn", False, False, False, False, False, "fa3", False),
            ("fused_adaln_residual_fa3", True, True, False, False, False, "fa3", False),
            ("fused_adaln_residual_fa3_tk_mlp", True, True, True, False, False, "fa3", False),
            ("fused_output_proj_fa3", True, True, False, False, True, "fa3", False),
            ("fused_input_proj_fa3", True, True, False, True, False, "fa3", False),
            ("fused_input_output_proj_fa3", True, True, False, True, True, "fa3", False),
            ("fused_input_proj_fa3_tk_mlp", True, True, True, True, False, "fa3", False),
            ("fused_input_output_proj_fa3_tk_mlp", True, True, True, True, True, "fa3", False),
        ])
        if include_compile:
            variants.extend([
                ("compile_fa3_attn", False, False, False, False, False, "fa3", True),
                ("compile_fused_adaln_residual_fa3", True, True, False, False, False, "fa3", True),
                ("compile_fused_adaln_residual_fa3_tk_mlp", True, True, True, False, False, "fa3", True),
            ])
    if only_variants:
        missing = only_variants.difference(name for name, *_ in variants)
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")
        variants = [variant for variant in variants if variant[0] in only_variants]
    results = []
    for variant_name, fused, fused_residual, tk_mlp, fused_input_projection, fused_output_projection, attention_backend, compiled in variants:
        print(f"  running {variant_name}...", flush=True)
        model = make_model(
            model_name,
            fused=fused,
            fused_residual=fused_residual,
            tk_mlp=tk_mlp,
            fused_input_projection=fused_input_projection,
            fused_output_projection=fused_output_projection,
            attention_backend=attention_backend,
        )
        model.pos_embed(spatial, torch.bfloat16, torch.device("cuda"))
        if compiled:
            model = torch.compile(model)
        try:
            result = profile_groups(
                f"DiT-{model_name} B{batch} {variant_name} train",
                groups,
                lambda g, current_model=model: train_step(current_model, g),
                warmup=max(1, min(2, warmup)) if compiled else warmup,
                iters=iters,
            )
            results.append(result)
            print_bench(result)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"DiT-{model_name} B{batch} {variant_name} train: SKIP OOM ({exc})", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
    if results:
        base = results[0].us
        print("  speedup: " + ", ".join(f"{r.name} {base / r.us:.2f}x" for r in results[1:]))
    return results


def probe_case(model_name: str, batch: int, spatial: tuple[int, int, int], include_compile: bool = False, include_fa3: bool = False):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(f"\n3D DiT-{model_name}/1 memory probe: batch={batch} tokens={tokens} spatial={spatial}")
    group = make_group(batch, cfg["in_channels"], spatial, 90000)
    for label, fused, fused_residual in (
        ("eager", False, False),
        ("fused_adaln", True, False),
        ("fused_adaln_residual", True, True),
    ):
        model = make_model(model_name, fused=fused, fused_residual=fused_residual)
        memory_probe(model, group, f"DiT-{model_name} B{batch} {label} train")
        del model
        torch.cuda.empty_cache()
    model = make_model(model_name, fused=False, fused_residual=False, tk_mlp=True)
    memory_probe(model, group, f"DiT-{model_name} B{batch} tk_mlp train")
    del model
    torch.cuda.empty_cache()
    model = make_model(model_name, fused=True, fused_residual=True, tk_mlp=True)
    memory_probe(model, group, f"DiT-{model_name} B{batch} fused_adaln_residual_tk_mlp train")
    del model
    torch.cuda.empty_cache()
    if include_fa3:
        model = make_model(model_name, fused=False, attention_backend="fa3")
        memory_probe(model, group, f"DiT-{model_name} B{batch} fa3_attn train")
        del model
        torch.cuda.empty_cache()
        model = make_model(model_name, fused=True, fused_residual=True, attention_backend="fa3")
        memory_probe(model, group, f"DiT-{model_name} B{batch} fused_adaln_residual_fa3 train")
        del model
        torch.cuda.empty_cache()
    if include_compile:
        model = torch.compile(make_model(model_name, fused=False))
        memory_probe(model, group, f"DiT-{model_name} B{batch} compile train")
        del model
        torch.cuda.empty_cache()


def spatial_for_tokens(tokens: int) -> tuple[int, int, int]:
    shapes = {
        256: (4, 8, 8),
        512: (8, 8, 8),
        1024: (8, 8, 16),
        2048: (8, 16, 16),
        4096: (16, 16, 16),
        8192: (16, 16, 32),
        16384: (16, 32, 32),
        32768: (32, 32, 32),
        60000: (30, 40, 50),
        65536: (32, 32, 64),
    }
    if tokens in shapes:
        return shapes[tokens]
    best = (1, 1, tokens)
    best_score = (tokens, tokens)
    for d in range(1, int(round(tokens ** (1 / 3))) + 3):
        if tokens % d:
            continue
        plane = tokens // d
        for h in range(d, int(math.sqrt(plane)) + 2):
            if plane % h:
                continue
            w = plane // h
            dims = tuple(sorted((d, h, w)))
            score = (dims[-1] - dims[0], dims[-1])
            if score < best_score:
                best = dims
                best_score = score
    return best


def main():
    parser = argparse.ArgumentParser(description="Benchmark full 3D DiT training variants.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="S")
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--spatial", nargs=3, type=int, default=[2, 2, 4])
    parser.add_argument("--tokens", nargs="+", type=int, default=None, help="Token counts to benchmark. Arbitrary counts are mapped to an exact-product 3D shape.")
    parser.add_argument("--sweep", action="store_true", help="Run every token count in --tokens for every batch in --batches.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fa3", action="store_true", help="Include FlashAttention-3 attention variants.")
    parser.add_argument("--variants", nargs="+", default=None, help="Only run the named benchmark variants.")
    parser.add_argument("--profile-variant", default="", help="Run torch profiler for one named variant and exit.")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--check-fused-input", action="store_true", help="Run isolated LN+AdaLN+projection correctness checks and exit.")
    parser.add_argument("--bench-residual", action="store_true", help="Run isolated standalone gated residual forward+backward benchmarks and exit.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension for isolated residual benchmark.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--probe-memory", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    a = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.mm(a, b).sum().backward()
    torch.cuda.synchronize()
    if args.check_fused_input:
        if not fused_input_projection_correctness():
            raise SystemExit(1)
        return
    if args.bench_residual:
        bench_residual(args.tokens or [1024], args.batches, args.hidden_dim, args.warmup, args.iters)
        return
    if args.profile_variant:
        spatial = spatial_for_tokens(args.tokens[0]) if args.tokens else tuple(args.spatial)
        profile_variant_case(args.model, args.profile_variant, args.batches[0], spatial, args.warmup, args.iters, args.profile_rows)
        return
    all_results = []
    cases = []
    if args.sweep or args.tokens is not None:
        token_counts = args.tokens or [512, 1024, 2048, 4096, 8192, 16384, 32768, 60000]
        for tokens in token_counts:
            spatial = spatial_for_tokens(tokens)
            for batch in args.batches:
                cases.append((batch, spatial, tokens))
    else:
        spatial = tuple(args.spatial)
        cases = [(batch, spatial, spatial[0] * spatial[1] * spatial[2]) for batch in args.batches]
    for batch, spatial, tokens in cases:
        try:
            if args.probe_memory:
                probe_case(args.model, batch, spatial, args.compile, args.fa3)
            else:
                all_results.extend(bench_case(
                    args.model,
                    batch,
                    spatial,
                    args.compile,
                    args.warmup,
                    args.iters,
                    args.fa3,
                    set(args.variants) if args.variants else None,
                ))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"\nDiT-{args.model} B{batch} T{tokens}: SKIP OOM ({exc})")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\nDiT-{args.model} B{batch} T{tokens}: SKIP OOM ({exc})")
            else:
                raise
        torch.cuda.empty_cache()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
