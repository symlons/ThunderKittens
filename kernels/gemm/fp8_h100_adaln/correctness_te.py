from __future__ import annotations

import argparse

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled,
    ln_adaln_quantize_k1024,
    ln_adaln_quantize_stats_k1024,
)


def raw_u8(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:
        return t
    return t.view(torch.uint8)


def raw_e4m3(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t
    return t.view(torch.float8_e4m3fn)


def te_general_gemm_tn_bf16(x_fp8: torch.Tensor, w_fp8: torch.Tensor) -> torch.Tensor:
    try:
        out, _, _, _ = general_gemm(w_fp8, x_fp8, out_dtype=torch.bfloat16, layout="TN")
    except TypeError as exc:
        if "workspace" not in str(exc):
            raise
        workspace = torch.empty(4_194_304, dtype=torch.uint8, device=x_fp8.device)
        out, _, _, _ = general_gemm(w_fp8, x_fp8, workspace, out_dtype=torch.bfloat16, layout="TN")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1.75)
    args = parser.parse_args()

    torch.manual_seed(4321)
    device = "cuda"
    k = 1024
    rows = args.batch * args.tokens

    x = torch.randn((rows, k), device=device, dtype=torch.bfloat16) * 2.0
    shift = torch.randn((args.batch, k), device=device, dtype=torch.bfloat16) * 0.05
    scale = torch.randn((args.batch, k), device=device, dtype=torch.bfloat16) * 0.05
    mean = torch.randn((rows,), device=device, dtype=torch.float32) * 0.1
    rstd = torch.rand((rows,), device=device, dtype=torch.float32) + 0.5

    tk_q, tk_global_amax = ln_adaln_quantize_k1024(
        x, shift, scale, mean, rstd, args.tokens, args.scale
    )
    tk_q_stats, tk_global_amax_stats, mean_stats, rstd_stats = ln_adaln_quantize_stats_k1024(
        x, shift, scale, args.tokens, args.scale, 1e-6
    )
    torch.cuda.synchronize()

    batch_idx = torch.arange(rows, device=device) // args.tokens
    y = (x.float() - mean[:, None]) * rstd[:, None]
    y = (y * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()).contiguous()

    te_scale = torch.full((1,), args.scale, dtype=torch.float32, device=device)
    te_amax = torch.zeros((1,), dtype=torch.float32, device=device)
    quantizer = Float8Quantizer(
        scale=te_scale,
        amax=te_amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    te_q = quantizer(y)
    torch.cuda.synchronize()

    tk_raw = raw_u8(tk_q).cpu()
    te_raw = raw_u8(te_q._data).cpu()
    raw_match = torch.equal(tk_raw, te_raw)

    y_global_amax = y.abs().amax()
    te_global_amax = te_amax[0]

    mean_ref = x.float().mean(dim=1)
    rstd_ref = torch.rsqrt((x.float() - mean_ref[:, None]).square().mean(dim=1) + 1e-6)
    y_stats = (x.float() - mean_ref[:, None]) * rstd_ref[:, None]
    y_stats = (y_stats * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()).contiguous()
    te_amax_stats = torch.zeros((1,), dtype=torch.float32, device=device)
    quantizer_stats = Float8Quantizer(
        scale=te_scale,
        amax=te_amax_stats,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    te_q_stats = quantizer_stats(y_stats)
    torch.cuda.synchronize()
    stats_raw_match = torch.equal(raw_u8(tk_q_stats).cpu(), raw_u8(te_q_stats._data).cpu())

    print(f"raw_match={raw_match}")
    print(f"tk_global_amax={tk_global_amax.item():.8g}")
    print(f"te_global_amax={te_global_amax.item():.8g}")
    print(f"y_global_amax={y_global_amax.item():.8g}")
    print(f"te_scale_inv={te_q._scale_inv.item():.8g}")
    print(f"expected_scale_inv={1.0 / args.scale:.8g}")
    print(f"global_amax_abs_diff={(tk_global_amax - te_global_amax).abs().item():.8g}")
    print(f"scale_inv_abs_diff={(te_q._scale_inv - (1.0 / args.scale)).abs().item():.8g}")
    print(f"stats_raw_match={stats_raw_match}")
    print(f"stats_tk_global_amax={tk_global_amax_stats.item():.8g}")
    print(f"stats_te_global_amax={te_amax_stats.item():.8g}")
    print(f"stats_global_amax_abs_diff={(tk_global_amax_stats - te_amax_stats).abs().item():.8g}")
    print(f"stats_mean_max_abs_diff={(mean_stats - mean_ref).abs().max().item():.8g}")
    print(f"stats_rstd_max_abs_diff={(rstd_stats - rstd_ref).abs().max().item():.8g}")

    gemm_x = torch.randn((128, k), device=device, dtype=torch.bfloat16) * 0.02
    w = torch.randn((256, k), device=device, dtype=torch.bfloat16) * 0.02
    x_quantizer = Float8Quantizer(
        scale=torch.full((1,), args.scale, dtype=torch.float32, device=device),
        amax=torch.zeros((1,), dtype=torch.float32, device=device),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    w_quantizer = Float8Quantizer(
        scale=torch.full((1,), args.scale, dtype=torch.float32, device=device),
        amax=torch.zeros((1,), dtype=torch.float32, device=device),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    te_gemm_x = x_quantizer(gemm_x.float() / args.scale)
    te_w = w_quantizer(w.float() / args.scale)
    te_gemm_out = te_general_gemm_tn_bf16(te_gemm_x, te_w)
    tk_gemm_out = fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled(
        raw_e4m3(te_gemm_x._data),
        raw_e4m3(te_w._data),
        float(te_gemm_x._scale_inv.item()),
        float(te_w._scale_inv.item()),
    )
    gemm_ref = raw_e4m3(te_gemm_x._data).float() @ raw_e4m3(te_w._data).float().T
    gemm_ref = gemm_ref * te_gemm_x._scale_inv.float() * te_w._scale_inv.float()
    te_gemm_diff = (te_gemm_out.float() - gemm_ref).abs().max()
    tk_te_gemm_diff = (tk_gemm_out.float() - te_gemm_out.float()).abs().max()
    print(f"te_gemm_ref_max_abs_diff={te_gemm_diff.item():.8g}")
    print(f"tk_te_gemm_max_abs_diff={tk_te_gemm_diff.item():.8g}")

    if not raw_match:
        mismatches = (tk_raw != te_raw).sum().item()
        raise AssertionError(f"TK FP8 bytes differ from TE: {mismatches} mismatches")
    if (tk_global_amax - te_global_amax).abs().item() > 1e-5:
        raise AssertionError("TK reduced amax differs from TE amax")
    if (tk_global_amax - y_global_amax).abs().item() > 1e-5:
        raise AssertionError("TK reduced amax differs from PyTorch y.abs().amax()")
    if (te_q._scale_inv - (1.0 / args.scale)).abs().item() > 1e-7:
        raise AssertionError("TE scale_inv does not match reciprocal scale")
    if not stats_raw_match:
        raise AssertionError("stats TK FP8 bytes differ from TE")
    if (tk_global_amax_stats - te_amax_stats).abs().item() > 1e-5:
        raise AssertionError("stats TK reduced amax differs from TE amax")
    if (mean_stats - mean_ref).abs().max().item() > 1e-5 or (rstd_stats - rstd_ref).abs().max().item() > 1e-5:
        raise AssertionError("stats mean/rstd differs from PyTorch")
    if te_gemm_diff.item() > 0.03125:
        raise AssertionError("TE GEMM differs from dequantized torch reference")
    if tk_te_gemm_diff.item() > 0.03125:
        raise AssertionError("TK scaled GEMM differs from TE GEMM")


if __name__ == "__main__":
    main()
