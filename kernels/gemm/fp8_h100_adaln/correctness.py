from __future__ import annotations

import argparse

import torch

from _C import fp8_gemm_k1024_fp32_out, ln_adaln_quantize_k1024, ln_adaln_quantize_stats_k1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=7)
    parser.add_argument("--inv-quant-scale", type=float, default=1.75)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = "cuda"
    k = 1024
    rows = args.batch * args.tokens

    x = torch.randn((rows, k), device=device, dtype=torch.bfloat16) * 2.0
    shift = torch.randn((args.batch, k), device=device, dtype=torch.bfloat16) * 0.05
    scale = torch.randn((args.batch, k), device=device, dtype=torch.bfloat16) * 0.05
    mean = torch.randn((rows,), device=device, dtype=torch.float32) * 0.1
    rstd = torch.rand((rows,), device=device, dtype=torch.float32) + 0.5

    q, global_amax = ln_adaln_quantize_k1024(
        x, shift, scale, mean, rstd, args.tokens, args.inv_quant_scale
    )
    q_stats, global_amax_stats, mean_stats, rstd_stats = ln_adaln_quantize_stats_k1024(
        x, shift, scale, args.tokens, args.inv_quant_scale, 1e-6
    )
    torch.cuda.synchronize()

    batch_idx = torch.arange(rows, device=device) // args.tokens
    y = (x.float() - mean[:, None]) * rstd[:, None]
    y = y * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()
    q_ref = (y * args.inv_quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    global_amax_ref = y.abs().amax()

    mean_ref = x.float().mean(dim=1)
    rstd_ref = torch.rsqrt((x.float() - mean_ref[:, None]).square().mean(dim=1) + 1e-6)
    y_stats = (x.float() - mean_ref[:, None]) * rstd_ref[:, None]
    y_stats = y_stats * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()
    q_stats_ref = (y_stats * args.inv_quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    global_amax_stats_ref = y_stats.abs().amax()

    q_match = torch.equal(q.cpu(), q_ref.cpu())
    global_abs = (global_amax[0] - global_amax_ref).abs().item()

    print(f"q_match={q_match}")
    print(f"global_amax={global_amax[0].item():.8g}")
    print(f"global_amax_ref={global_amax_ref.item():.8g}")
    print(f"global_amax_abs_diff={global_abs:.8g}")
    q_stats_match = torch.equal(q_stats.cpu(), q_stats_ref.cpu())
    mean_abs = (mean_stats - mean_ref).abs().max().item()
    rstd_abs = (rstd_stats - rstd_ref).abs().max().item()
    global_stats_abs = (global_amax_stats[0] - global_amax_stats_ref).abs().item()
    print(f"stats_q_match={q_stats_match}")
    print(f"stats_global_amax={global_amax_stats[0].item():.8g}")
    print(f"stats_global_amax_ref={global_amax_stats_ref.item():.8g}")
    print(f"stats_global_amax_abs_diff={global_stats_abs:.8g}")
    print(f"mean_max_abs_diff={mean_abs:.8g}")
    print(f"rstd_max_abs_diff={rstd_abs:.8g}")

    gemm_m = 128
    gemm_n = 256
    q_gemm = torch.randn((gemm_m, k), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w = (torch.randn((gemm_n, k), device=device, dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
    gemm = fp8_gemm_k1024_fp32_out(q_gemm, w)
    gemm_ref = q_gemm.float() @ w.float().T
    gemm_diff = (gemm - gemm_ref).abs()
    gemm_abs = gemm_diff.max().item()
    gemm_mean = gemm_diff.mean().item()
    print(f"fp32_out_gemm_max_abs_diff={gemm_abs:.8g}")
    print(f"fp32_out_gemm_mean_abs_diff={gemm_mean:.8g}")

    if not q_match:
        mismatches = (q.cpu() != q_ref.cpu()).sum().item()
        raise AssertionError(f"FP8 quantized output mismatch: {mismatches} elements")
    if global_abs > 1e-5:
        raise AssertionError(f"global amax mismatch: max abs {global_abs}")
    if not q_stats_match:
        mismatches = (q_stats.cpu() != q_stats_ref.cpu()).sum().item()
        raise AssertionError(f"stats FP8 quantized output mismatch: {mismatches} elements")
    if global_stats_abs > 1e-5:
        raise AssertionError(f"stats global amax mismatch: max abs {global_stats_abs}")
    if mean_abs > 1e-5 or rstd_abs > 1e-5:
        raise AssertionError(f"stats mismatch: mean {mean_abs}, rstd {rstd_abs}")
    if gemm_abs > 1e-3 or gemm_mean > 1e-4:
        raise AssertionError(f"FP32 output GEMM mismatch: max abs {gemm_abs}, mean abs {gemm_mean}")


if __name__ == "__main__":
    main()
