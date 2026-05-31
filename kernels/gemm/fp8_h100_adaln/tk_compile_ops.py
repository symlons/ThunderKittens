from __future__ import annotations

import torch

import _C


@torch.library.custom_op(
    "tk_fp8_adaln::ln_adaln_quantize_stats_vec_delayed_k1024",
    mutates_args=(),
)
def ln_adaln_quantize_stats_vec_delayed_k1024(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    quant_scale: torch.Tensor,
    tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.ln_adaln_quantize_stats_vec_delayed_k1024(
        x, shift, scale, quant_scale, tokens, eps
    )


@ln_adaln_quantize_stats_vec_delayed_k1024.register_fake
def _(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    quant_scale: torch.Tensor,
    tokens: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shift, scale, quant_scale, tokens, eps
    q = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn)
    row_amax = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    return q, row_amax


@torch.library.custom_op(
    "tk_fp8_adaln::fp8_gemm_k1024_bf16_out_wide_scaled",
    mutates_args=(),
)
def fp8_gemm_k1024_bf16_out_wide_scaled(
    a: torch.Tensor,
    b: torch.Tensor,
    a_dequant_scale: float,
    b_dequant_scale: float,
) -> torch.Tensor:
    return _C.fp8_gemm_k1024_bf16_out_wide_scaled(
        a, b, a_dequant_scale, b_dequant_scale
    )


@fp8_gemm_k1024_bf16_out_wide_scaled.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    a_dequant_scale: float,
    b_dequant_scale: float,
) -> torch.Tensor:
    del a_dequant_scale, b_dequant_scale
    return torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)


@torch.library.custom_op(
    "tk_fp8_adaln::fp8_gemm_k4096_bf16_out_bias",
    mutates_args=(),
)
def fp8_gemm_k4096_bf16_out_bias(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _C.fp8_gemm_k4096_bf16_out_bias(a, b, bias)


@fp8_gemm_k4096_bf16_out_bias.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del bias
    return torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.bfloat16)


@torch.library.custom_op(
    "tk_fp8_adaln::bias_gelu_quantize_k4096",
    mutates_args=(),
)
def bias_gelu_quantize_k4096(
    x: torch.Tensor,
    bias: torch.Tensor,
    quant_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.bias_gelu_quantize_k4096(x, bias, quant_scale)


@bias_gelu_quantize_k4096.register_fake
def _(
    x: torch.Tensor,
    bias: torch.Tensor,
    quant_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del bias, quant_scale
    q = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn)
    row_amax = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    return q, row_amax
