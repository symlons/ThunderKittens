from __future__ import annotations

import argparse

import torch

from _C import (
    fp8_gemm_k1024_bf16_out,
    fp8_gemm_k1024_bf16_out_scaled,
    fp8_gemm_k1024_bf16_out_wide_scaled,
    fp8_gemm_k1024_bf16_out_deepaccum,
    fp8_gemm_k1024_bf16_out_deepaccum_n64,
    fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled,
    fp8_gemm_k1024_bf16_out_deepaccum_scaled,
    fp8_gemm_k1024_bf16_out_pipe,
    fp8_gemm_k1024_bf16_out_pipe64,
    fp8_gemm_k1024_fp32_out,
    ln_adaln_quantize_k1024,
    ln_adaln_quantize_stats_k1024,
)


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

    gemm_bf16 = fp8_gemm_k1024_bf16_out(q_gemm, w)
    gemm_bf16_ref = gemm_ref.to(torch.bfloat16)
    gemm_bf16_diff = (gemm_bf16.float() - gemm_bf16_ref.float()).abs()
    gemm_bf16_abs = gemm_bf16_diff.max().item()
    gemm_bf16_mean = gemm_bf16_diff.mean().item()
    print(f"bf16_out_gemm_max_abs_diff={gemm_bf16_abs:.8g}")
    print(f"bf16_out_gemm_mean_abs_diff={gemm_bf16_mean:.8g}")

    a_dequant_scale = 0.75
    b_dequant_scale = 1.25
    gemm_bf16_scaled = fp8_gemm_k1024_bf16_out_scaled(
        q_gemm, w, a_dequant_scale, b_dequant_scale
    )
    gemm_bf16_scaled_ref = (gemm_ref * (a_dequant_scale * b_dequant_scale)).to(torch.bfloat16)
    gemm_bf16_scaled_diff = (gemm_bf16_scaled.float() - gemm_bf16_scaled_ref.float()).abs()
    gemm_bf16_scaled_abs = gemm_bf16_scaled_diff.max().item()
    gemm_bf16_scaled_mean = gemm_bf16_scaled_diff.mean().item()
    print(f"bf16_out_scaled_gemm_max_abs_diff={gemm_bf16_scaled_abs:.8g}")
    print(f"bf16_out_scaled_gemm_mean_abs_diff={gemm_bf16_scaled_mean:.8g}")

    gemm_bf16_wide_scaled = fp8_gemm_k1024_bf16_out_wide_scaled(
        q_gemm, w, a_dequant_scale, b_dequant_scale
    )
    gemm_bf16_wide_scaled_diff = (gemm_bf16_wide_scaled.float() - gemm_bf16_scaled_ref.float()).abs()
    gemm_bf16_wide_scaled_abs = gemm_bf16_wide_scaled_diff.max().item()
    gemm_bf16_wide_scaled_mean = gemm_bf16_wide_scaled_diff.mean().item()
    print(f"bf16_out_wide_scaled_gemm_max_abs_diff={gemm_bf16_wide_scaled_abs:.8g}")
    print(f"bf16_out_wide_scaled_gemm_mean_abs_diff={gemm_bf16_wide_scaled_mean:.8g}")

    gemm_bf16_deepaccum = fp8_gemm_k1024_bf16_out_deepaccum(q_gemm, w)
    gemm_bf16_deepaccum_diff = (gemm_bf16_deepaccum.float() - gemm_bf16_ref.float()).abs()
    gemm_bf16_deepaccum_abs = gemm_bf16_deepaccum_diff.max().item()
    gemm_bf16_deepaccum_mean = gemm_bf16_deepaccum_diff.mean().item()
    print(f"bf16_out_deepaccum_gemm_max_abs_diff={gemm_bf16_deepaccum_abs:.8g}")
    print(f"bf16_out_deepaccum_gemm_mean_abs_diff={gemm_bf16_deepaccum_mean:.8g}")

    gemm_bf16_deepaccum_scaled = fp8_gemm_k1024_bf16_out_deepaccum_scaled(
        q_gemm, w, a_dequant_scale, b_dequant_scale
    )
    gemm_bf16_deepaccum_scaled_diff = (
        gemm_bf16_deepaccum_scaled.float() - gemm_bf16_scaled_ref.float()
    ).abs()
    gemm_bf16_deepaccum_scaled_abs = gemm_bf16_deepaccum_scaled_diff.max().item()
    gemm_bf16_deepaccum_scaled_mean = gemm_bf16_deepaccum_scaled_diff.mean().item()
    print(f"bf16_out_deepaccum_scaled_gemm_max_abs_diff={gemm_bf16_deepaccum_scaled_abs:.8g}")
    print(f"bf16_out_deepaccum_scaled_gemm_mean_abs_diff={gemm_bf16_deepaccum_scaled_mean:.8g}")

    gemm_bf16_deepaccum_n64 = fp8_gemm_k1024_bf16_out_deepaccum_n64(q_gemm, w)
    gemm_bf16_deepaccum_n64_diff = (gemm_bf16_deepaccum_n64.float() - gemm_bf16_ref.float()).abs()
    gemm_bf16_deepaccum_n64_abs = gemm_bf16_deepaccum_n64_diff.max().item()
    gemm_bf16_deepaccum_n64_mean = gemm_bf16_deepaccum_n64_diff.mean().item()
    print(f"bf16_out_deepaccum_n64_gemm_max_abs_diff={gemm_bf16_deepaccum_n64_abs:.8g}")
    print(f"bf16_out_deepaccum_n64_gemm_mean_abs_diff={gemm_bf16_deepaccum_n64_mean:.8g}")

    gemm_bf16_deepaccum_n64_scaled = fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled(
        q_gemm, w, a_dequant_scale, b_dequant_scale
    )
    gemm_bf16_deepaccum_n64_scaled_diff = (
        gemm_bf16_deepaccum_n64_scaled.float() - gemm_bf16_scaled_ref.float()
    ).abs()
    gemm_bf16_deepaccum_n64_scaled_abs = gemm_bf16_deepaccum_n64_scaled_diff.max().item()
    gemm_bf16_deepaccum_n64_scaled_mean = gemm_bf16_deepaccum_n64_scaled_diff.mean().item()
    print(f"bf16_out_deepaccum_n64_scaled_gemm_max_abs_diff={gemm_bf16_deepaccum_n64_scaled_abs:.8g}")
    print(f"bf16_out_deepaccum_n64_scaled_gemm_mean_abs_diff={gemm_bf16_deepaccum_n64_scaled_mean:.8g}")

    gemm_bf16_pipe = fp8_gemm_k1024_bf16_out_pipe(q_gemm, w)
    gemm_bf16_pipe_diff = (gemm_bf16_pipe.float() - gemm_bf16_ref.float()).abs()
    gemm_bf16_pipe_abs = gemm_bf16_pipe_diff.max().item()
    gemm_bf16_pipe_mean = gemm_bf16_pipe_diff.mean().item()
    print(f"bf16_out_pipe_gemm_max_abs_diff={gemm_bf16_pipe_abs:.8g}")
    print(f"bf16_out_pipe_gemm_mean_abs_diff={gemm_bf16_pipe_mean:.8g}")

    gemm_bf16_pipe64 = fp8_gemm_k1024_bf16_out_pipe64(q_gemm, w)
    gemm_bf16_pipe64_diff = (gemm_bf16_pipe64.float() - gemm_bf16_ref.float()).abs()
    gemm_bf16_pipe64_abs = gemm_bf16_pipe64_diff.max().item()
    gemm_bf16_pipe64_mean = gemm_bf16_pipe64_diff.mean().item()
    print(f"bf16_out_pipe64_gemm_max_abs_diff={gemm_bf16_pipe64_abs:.8g}")
    print(f"bf16_out_pipe64_gemm_mean_abs_diff={gemm_bf16_pipe64_mean:.8g}")

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
    if gemm_bf16_abs > 2e-2 or gemm_bf16_mean > 2e-3:
        raise AssertionError(f"BF16 output GEMM mismatch: max abs {gemm_bf16_abs}, mean abs {gemm_bf16_mean}")
    if gemm_bf16_scaled_abs > 2e-2 or gemm_bf16_scaled_mean > 2e-3:
        raise AssertionError(
            f"Scaled BF16 output GEMM mismatch: max abs {gemm_bf16_scaled_abs}, mean abs {gemm_bf16_scaled_mean}"
        )
    if gemm_bf16_wide_scaled_abs > 2e-2 or gemm_bf16_wide_scaled_mean > 2e-3:
        raise AssertionError(
            "Wide scaled BF16 output GEMM mismatch: "
            f"max abs {gemm_bf16_wide_scaled_abs}, mean abs {gemm_bf16_wide_scaled_mean}"
        )
    if gemm_bf16_deepaccum_abs > 2e-2 or gemm_bf16_deepaccum_mean > 2e-3:
        raise AssertionError(
            f"DeepAccum BF16 output GEMM mismatch: max abs {gemm_bf16_deepaccum_abs}, mean abs {gemm_bf16_deepaccum_mean}"
        )
    if gemm_bf16_deepaccum_scaled_abs > 2e-2 or gemm_bf16_deepaccum_scaled_mean > 2e-3:
        raise AssertionError(
            "DeepAccum scaled BF16 output GEMM mismatch: "
            f"max abs {gemm_bf16_deepaccum_scaled_abs}, mean abs {gemm_bf16_deepaccum_scaled_mean}"
        )
    if gemm_bf16_deepaccum_n64_abs > 2e-2 or gemm_bf16_deepaccum_n64_mean > 2e-3:
        raise AssertionError(
            f"DeepAccum N64 BF16 output GEMM mismatch: max abs {gemm_bf16_deepaccum_n64_abs}, "
            f"mean abs {gemm_bf16_deepaccum_n64_mean}"
        )
    if gemm_bf16_deepaccum_n64_scaled_abs > 2e-2 or gemm_bf16_deepaccum_n64_scaled_mean > 2e-3:
        raise AssertionError(
            "DeepAccum N64 scaled BF16 output GEMM mismatch: "
            f"max abs {gemm_bf16_deepaccum_n64_scaled_abs}, mean abs {gemm_bf16_deepaccum_n64_scaled_mean}"
        )
    if gemm_bf16_pipe_abs > 2e-2 or gemm_bf16_pipe_mean > 2e-3:
        raise AssertionError(
            f"pipelined BF16 output GEMM mismatch: max abs {gemm_bf16_pipe_abs}, mean abs {gemm_bf16_pipe_mean}"
        )
    if gemm_bf16_pipe64_abs > 2e-2 or gemm_bf16_pipe64_mean > 2e-3:
        raise AssertionError(
            f"pipelined BF16 output GEMM pipe64 mismatch: max abs {gemm_bf16_pipe64_abs}, mean abs {gemm_bf16_pipe64_mean}"
        )


if __name__ == "__main__":
    main()
