from __future__ import annotations

import torch
import torch.nn as nn

import _C
import _gelu_bwd
import _linear_bwd_fused

__all__ = [
    "FusedAdaLNLinear",
    "FusedAdaLNLinearGelu",
    "FusedInputMlp",
    "FusedLinearGatedResidual",
    "TkMlp",
    "fused_adaln",
    "fused_adaln_linear",
    "fused_linear_gated_residual",
    "gated_residual",
    "linear_then_gated_residual",
    "modulate",
]


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
        dw = tk_dw_gemm_op(grad, z, w)
        dz = tk_dx_gemm_native_op(grad, w.contiguous(), x)
        db = tk_bias_reduce_op(grad)
        dx, dshift, dscale = tk_layernorm_adaln_backward_op(dz, x, scale, mean, rstd, ctx.tokens)
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
        dz_gelu, db = tk_gelu_bwd_bias_op(grad, preact)
        z = recompute_adaln_flat(x.reshape(ctx.shape), shift, scale, ctx.eps, ctx.tokens)
        dw = tk_dw_gemm_op(dz_gelu, z, w)
        dz = tk_dx_gemm_native_op(dz_gelu, w.contiguous(), x)
        dx, dshift, dscale = tk_layernorm_adaln_backward_op(dz, x, scale, mean, rstd, ctx.tokens)
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
        dresidual, dprojected, dgate = tk_gated_residual_backward_op(grad, projected, gate, ctx.tokens)
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
