from __future__ import annotations

import argparse

import torch

from _C import (
    bias_gelu_quantize_k4096,
    fp8_gemm_k1024_bf16_out,
    fp8_gemm_k1024_bf16_out_scaled,
    fp8_gemm_k1024_bf16_out_wide_scaled,
    fp8_gemm_k1024_bf16_out_deepaccum,
    fp8_gemm_k1024_bf16_out_deepaccum_n64,
    fp8_gemm_k1024_bf16_out_deepaccum_n64_scaled,
    fp8_gemm_k1024_bf16_out_deepaccum_scaled,
    fp8_gemm_k1024_bf16_out_pipe,
    fp8_gemm_k1024_bf16_out_pipe64,
    fp8_gemm_k4096_bf16_out_bias,
    fp8_gemm_k1024_fp32_out,
    delayed_scaling_update,
    ln_adaln_quantize_k1024,
    ln_adaln_quantize_precomputed_vec_k1024,
    ln_adaln_quantize_stats_k1024,
    ln_adaln_quantize_stats_delayed_k1024,
    ln_adaln_quantize_stats_vec_delayed_k1024,
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
    quant_scale = torch.full((1,), args.inv_quant_scale, device=device, dtype=torch.float32)
    q_delayed, row_amax_delayed, mean_delayed, rstd_delayed = ln_adaln_quantize_stats_delayed_k1024(
        x, shift, scale, quant_scale, args.tokens, 1e-6
    )
    delayed_scale = torch.ones((1,), device=device, dtype=torch.float32)
    delayed_scale_inv = torch.ones((1,), device=device, dtype=torch.float32)
    amax_history = torch.zeros((16,), device=device, dtype=torch.float32)
    hist_idx = torch.zeros((1,), device=device, dtype=torch.int32)
    delayed_global_amax, _, _, _, _ = delayed_scaling_update(
        row_amax_delayed, delayed_scale, delayed_scale_inv, amax_history, hist_idx, 1e-6
    )
    q_vec_delayed, row_amax_vec = ln_adaln_quantize_stats_vec_delayed_k1024(
        x, shift, scale, quant_scale, args.tokens, 1e-6
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
    q_pre_vec, row_amax_pre_vec = ln_adaln_quantize_precomputed_vec_k1024(
        x, shift, scale, mean_ref, rstd_ref, quant_scale, args.tokens
    )
    torch.cuda.synchronize()

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
    q_delayed_match = torch.equal(q_delayed.cpu(), q_stats_ref.cpu())
    row_amax_abs = (row_amax_delayed - y_stats.abs().amax(dim=1)).abs().max().item()
    delayed_global_abs = (delayed_global_amax[0] - global_amax_stats_ref).abs().item()
    delayed_mean_abs = (mean_delayed - mean_ref).abs().max().item()
    delayed_rstd_abs = (rstd_delayed - rstd_ref).abs().max().item()
    expected_next_scale = 448.0 / max(global_amax_stats_ref.item(), 1e-6)
    expected_next_scale_inv = max(global_amax_stats_ref.item(), 1e-6) / 448.0
    q_vec_delayed_match = torch.equal(q_vec_delayed.cpu(), q_stats_ref.cpu())
    row_amax_vec_abs = (row_amax_vec - y_stats.abs().amax(dim=1)).abs().max().item()
    q_pre_vec_match = torch.equal(q_pre_vec.cpu(), q_stats_ref.cpu())
    row_amax_pre_vec_abs = (row_amax_pre_vec - y_stats.abs().amax(dim=1)).abs().max().item()
    print(f"stats_q_match={q_stats_match}")
    print(f"stats_global_amax={global_amax_stats[0].item():.8g}")
    print(f"stats_global_amax_ref={global_amax_stats_ref.item():.8g}")
    print(f"stats_global_amax_abs_diff={global_stats_abs:.8g}")
    print(f"mean_max_abs_diff={mean_abs:.8g}")
    print(f"rstd_max_abs_diff={rstd_abs:.8g}")
    print(f"delayed_q_match={q_delayed_match}")
    print(f"delayed_row_amax_max_abs_diff={row_amax_abs:.8g}")
    print(f"delayed_global_amax={delayed_global_amax[0].item():.8g}")
    print(f"delayed_global_amax_abs_diff={delayed_global_abs:.8g}")
    print(f"delayed_mean_max_abs_diff={delayed_mean_abs:.8g}")
    print(f"delayed_rstd_max_abs_diff={delayed_rstd_abs:.8g}")
    print(f"delayed_next_scale={delayed_scale[0].item():.8g}")
    print(f"delayed_expected_next_scale={expected_next_scale:.8g}")
    print(f"delayed_next_scale_inv={delayed_scale_inv[0].item():.8g}")
    print(f"delayed_expected_next_scale_inv={expected_next_scale_inv:.8g}")
    print(f"vec_delayed_q_match={q_vec_delayed_match}")
    print(f"vec_delayed_row_amax_max_abs_diff={row_amax_vec_abs:.8g}")
    print(f"precomputed_vec_q_match={q_pre_vec_match}")
    print(f"precomputed_vec_row_amax_max_abs_diff={row_amax_pre_vec_abs:.8g}")

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

    k_down = 4096
    down_m = 128
    down_n = 1024
    up = torch.randn((down_m, k_down), device=device, dtype=torch.bfloat16) * 1.5
    up_bias = torch.randn((k_down,), device=device, dtype=torch.bfloat16) * 0.02
    up_q, up_row_amax = bias_gelu_quantize_k4096(up, up_bias, quant_scale)
    up_ref = torch.nn.functional.gelu(up.float() + up_bias.float(), approximate="tanh")
    up_q_ref = (up_ref * args.inv_quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    up_q_match = torch.equal(up_q.cpu(), up_q_ref.cpu())
    up_q_dequant_diff = (up_q.float() - up_q_ref.float()).abs()
    up_q_dequant_abs = up_q_dequant_diff.max().item()
    up_q_dequant_mean = up_q_dequant_diff.mean().item()
    up_row_amax_abs = (up_row_amax - up_ref.abs().amax(dim=1)).abs().max().item()
    print(f"bias_gelu_quantize_q_match={up_q_match}")
    print(f"bias_gelu_quantize_q_dequant_max_abs_diff={up_q_dequant_abs:.8g}")
    print(f"bias_gelu_quantize_q_dequant_mean_abs_diff={up_q_dequant_mean:.8g}")
    print(f"bias_gelu_quantize_row_amax_max_abs_diff={up_row_amax_abs:.8g}")

    q_down = torch.randn((down_m, k_down), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w_down = (torch.randn((down_n, k_down), device=device, dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
    b_down = torch.randn((down_n,), device=device, dtype=torch.bfloat16) * 0.02
    down = fp8_gemm_k4096_bf16_out_bias(q_down, w_down, b_down)
    down_ref = (q_down.float() @ w_down.float().T + b_down.float()).to(torch.bfloat16)
    down_diff = (down.float() - down_ref.float()).abs()
    down_abs = down_diff.max().item()
    down_mean = down_diff.mean().item()
    print(f"k4096_bf16_out_bias_gemm_max_abs_diff={down_abs:.8g}")
    print(f"k4096_bf16_out_bias_gemm_mean_abs_diff={down_mean:.8g}")

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
    if not q_delayed_match:
        mismatches = (q_delayed.cpu() != q_stats_ref.cpu()).sum().item()
        raise AssertionError(f"delayed stats FP8 quantized output mismatch: {mismatches} elements")
    if row_amax_abs > 1e-5 or delayed_global_abs > 1e-5:
        raise AssertionError(f"delayed amax mismatch: row {row_amax_abs}, global {delayed_global_abs}")
    if delayed_mean_abs > 1e-5 or delayed_rstd_abs > 1e-5:
        raise AssertionError(f"delayed stats mismatch: mean {delayed_mean_abs}, rstd {delayed_rstd_abs}")
    if abs(delayed_scale[0].item() - expected_next_scale) > 1e-5:
        raise AssertionError("delayed scale update mismatch")
    if abs(delayed_scale_inv[0].item() - expected_next_scale_inv) > 1e-7:
        raise AssertionError("delayed scale_inv update mismatch")
    if not q_vec_delayed_match:
        mismatches = (q_vec_delayed.cpu() != q_stats_ref.cpu()).sum().item()
        raise AssertionError(f"vectorized delayed FP8 output mismatch: {mismatches} elements")
    if row_amax_vec_abs > 1e-5:
        raise AssertionError(f"vectorized delayed row amax mismatch: {row_amax_vec_abs}")
    if not q_pre_vec_match:
        mismatches = (q_pre_vec.cpu() != q_stats_ref.cpu()).sum().item()
        raise AssertionError(f"precomputed vectorized FP8 output mismatch: {mismatches} elements")
    if row_amax_pre_vec_abs > 1e-5:
        raise AssertionError(f"precomputed vectorized row amax mismatch: {row_amax_pre_vec_abs}")
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
    if not up_q_match and (up_q_dequant_abs > 0.03125 or up_q_dequant_mean > 1e-5):
        mismatches = (up_q.cpu() != up_q_ref.cpu()).sum().item()
        raise AssertionError(
            "bias+GELU FP8 output mismatch: "
            f"{mismatches} elements, max abs {up_q_dequant_abs}, mean abs {up_q_dequant_mean}"
        )
    if up_row_amax_abs > 1e-5:
        raise AssertionError(f"bias+GELU row amax mismatch: {up_row_amax_abs}")


if __name__ == "__main__":
    main()
