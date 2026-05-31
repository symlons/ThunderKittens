from __future__ import annotations

import argparse
import statistics

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    fp8_gemm_k1024,
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


def event_bench(fn, *, warmup: int, iters: int) -> dict[str, float]:
    # Same timing pattern as kernels/layernorm/benchmark.py.
    for stage in ["warmup", "timed"]:
        times = []
        reps = warmup if stage == "warmup" else iters
        for _ in range(reps):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            if stage == "timed":
                times.append(start.elapsed_time(end))
            del out
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def te_quantize_e4m3(x: torch.Tensor, scale: float) -> torch.Tensor:
    quantizer = Float8Quantizer(
        scale=torch.full((1,), scale, device=x.device, dtype=torch.float32),
        amax=torch.zeros((1,), device=x.device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    return quantizer(x)


def te_raw_e4m3(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t
    return t.view(torch.float8_e4m3fn)


def te_general_gemm_tn_bf16(x_fp8: torch.Tensor, w_fp8: torch.Tensor) -> torch.Tensor:
    # TE's TN path computes B @ A.T.  With A=[N,K] weights and B=[M,K] activations,
    # this is the same logical GEMM as x @ w.T.
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
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 3072, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--te-scale-a", type=float, default=1.0)
    parser.add_argument("--te-scale-b", type=float, default=1.0)
    args = parser.parse_args()

    m = args.batch * args.tokens
    k = 1024
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    shift = torch.randn((args.batch, k), device="cuda", dtype=torch.bfloat16) * 0.02
    scale = torch.randn((args.batch, k), device="cuda", dtype=torch.bfloat16) * 0.02
    mean = x.float().mean(dim=1)
    rstd = torch.rsqrt(x.float().var(dim=1, unbiased=False) + 1e-6)

    quant_stats = event_bench(
        lambda: ln_adaln_quantize_k1024(x, shift, scale, mean, rstd, args.tokens, 1.0),
        warmup=args.warmup,
        iters=args.iters,
    )
    fused_stats_quant_stats = event_bench(
        lambda: ln_adaln_quantize_stats_k1024(x, shift, scale, args.tokens, 1.0, 1e-6),
        warmup=args.warmup,
        iters=args.iters,
    )
    q, global_amax = ln_adaln_quantize_k1024(x, shift, scale, mean, rstd, args.tokens, 1.0)
    q_stats, global_amax_stats, _, _ = ln_adaln_quantize_stats_k1024(x, shift, scale, args.tokens, 1.0, 1e-6)

    print(f"ln_adaln_quantize_k1024_per_tensor: {quant_stats}")
    print(f"ln_adaln_quantize_stats_k1024_per_tensor: {fused_stats_quant_stats}")
    print(f"global_amax={global_amax[0].item():.8g}")
    print(f"global_amax_stats={global_amax_stats[0].item():.8g}")

    for n in args.n:
        w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
        w = w_bf16.to(torch.float8_e4m3fn)
        te_x = te_quantize_e4m3(q.float() / args.te_scale_a, args.te_scale_a)
        te_w = te_quantize_e4m3(w_bf16.float() / args.te_scale_b, args.te_scale_b)
        te_a_inv = float(te_x._scale_inv.item())
        te_b_inv = float(te_w._scale_inv.item())
        stats = event_bench(lambda: fp8_gemm_k1024(q, w), warmup=args.warmup, iters=args.iters)
        stats_fused_q = event_bench(lambda: fp8_gemm_k1024(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_fp32 = event_bench(lambda: fp8_gemm_k1024_fp32_out(q, w), warmup=args.warmup, iters=args.iters)
        stats_fp32_fused_q = event_bench(lambda: fp8_gemm_k1024_fp32_out(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_bf16 = event_bench(lambda: fp8_gemm_k1024_bf16_out(q, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_scaled = event_bench(lambda: fp8_gemm_k1024_bf16_out_scaled(q, w, 1.0, 1.0), warmup=args.warmup, iters=args.iters)
        stats_bf16_scaled_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out_scaled(q_stats, w, 1.0, 1.0), warmup=args.warmup, iters=args.iters)
        stats_bf16_wide_scaled = event_bench(lambda: fp8_gemm_k1024_bf16_out_wide_scaled(q, w, 1.0, 1.0), warmup=args.warmup, iters=args.iters)
        stats_bf16_wide_scaled_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out_wide_scaled(q_stats, w, 1.0, 1.0), warmup=args.warmup, iters=args.iters)
        stats_bf16_deepaccum = event_bench(lambda: fp8_gemm_k1024_bf16_out_deepaccum(q, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_deepaccum_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out_deepaccum(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_deepaccum_n64 = event_bench(lambda: fp8_gemm_k1024_bf16_out_deepaccum_n64(q, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_deepaccum_n64_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out_deepaccum_n64(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_pipe = event_bench(lambda: fp8_gemm_k1024_bf16_out_pipe(q, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_pipe_fused_q = event_bench(lambda: fp8_gemm_k1024_bf16_out_pipe(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_bf16_pipe64 = event_bench(lambda: fp8_gemm_k1024_bf16_out_pipe64(q, w), warmup=args.warmup, iters=args.iters)
        stats_te = event_bench(lambda: te_general_gemm_tn_bf16(te_x, te_w), warmup=args.warmup, iters=args.iters)
        stats_tk_te_raw = event_bench(
            lambda: fp8_gemm_k1024_bf16_out_wide_scaled(te_raw_e4m3(te_x._data), te_raw_e4m3(te_w._data), te_a_inv, te_b_inv),
            warmup=args.warmup,
            iters=args.iters,
        )
        flops = 2.0 * m * n * k
        stats["tflops"] = flops / (stats["mean_ms"] * 1e-3) / 1e12
        stats_fused_q["tflops"] = flops / (stats_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_fp32["tflops"] = flops / (stats_fp32["mean_ms"] * 1e-3) / 1e12
        stats_fp32_fused_q["tflops"] = flops / (stats_fp32_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16["tflops"] = flops / (stats_bf16["mean_ms"] * 1e-3) / 1e12
        stats_bf16_fused_q["tflops"] = flops / (stats_bf16_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_scaled["tflops"] = flops / (stats_bf16_scaled["mean_ms"] * 1e-3) / 1e12
        stats_bf16_scaled_fused_q["tflops"] = flops / (stats_bf16_scaled_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_wide_scaled["tflops"] = flops / (stats_bf16_wide_scaled["mean_ms"] * 1e-3) / 1e12
        stats_bf16_wide_scaled_fused_q["tflops"] = flops / (stats_bf16_wide_scaled_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_deepaccum["tflops"] = flops / (stats_bf16_deepaccum["mean_ms"] * 1e-3) / 1e12
        stats_bf16_deepaccum_fused_q["tflops"] = flops / (stats_bf16_deepaccum_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_deepaccum_n64["tflops"] = flops / (stats_bf16_deepaccum_n64["mean_ms"] * 1e-3) / 1e12
        stats_bf16_deepaccum_n64_fused_q["tflops"] = flops / (stats_bf16_deepaccum_n64_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_pipe["tflops"] = flops / (stats_bf16_pipe["mean_ms"] * 1e-3) / 1e12
        stats_bf16_pipe_fused_q["tflops"] = flops / (stats_bf16_pipe_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_bf16_pipe64["tflops"] = flops / (stats_bf16_pipe64["mean_ms"] * 1e-3) / 1e12
        stats_te["tflops"] = flops / (stats_te["mean_ms"] * 1e-3) / 1e12
        stats_tk_te_raw["tflops"] = flops / (stats_tk_te_raw["mean_ms"] * 1e-3) / 1e12
        print(f"fp8_gemm_k1024 M={m} N={n} K=1024: {stats}")
        print(f"fp8_gemm_k1024_fused_stats_q M={m} N={n} K=1024: {stats_fused_q}")
        print(f"fp8_gemm_k1024_fp32_out M={m} N={n} K=1024: {stats_fp32}")
        print(f"fp8_gemm_k1024_fp32_out_fused_stats_q M={m} N={n} K=1024: {stats_fp32_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out M={m} N={n} K=1024: {stats_bf16}")
        print(f"fp8_gemm_k1024_bf16_out_fused_stats_q M={m} N={n} K=1024: {stats_bf16_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_scaled M={m} N={n} K=1024: {stats_bf16_scaled}")
        print(f"fp8_gemm_k1024_bf16_out_scaled_fused_stats_q M={m} N={n} K=1024: {stats_bf16_scaled_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_wide_scaled M={m} N={n} K=1024: {stats_bf16_wide_scaled}")
        print(f"fp8_gemm_k1024_bf16_out_wide_scaled_fused_stats_q M={m} N={n} K=1024: {stats_bf16_wide_scaled_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_deepaccum M={m} N={n} K=1024: {stats_bf16_deepaccum}")
        print(f"fp8_gemm_k1024_bf16_out_deepaccum_fused_stats_q M={m} N={n} K=1024: {stats_bf16_deepaccum_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_deepaccum_n64 M={m} N={n} K=1024: {stats_bf16_deepaccum_n64}")
        print(f"fp8_gemm_k1024_bf16_out_deepaccum_n64_fused_stats_q M={m} N={n} K=1024: {stats_bf16_deepaccum_n64_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_pipe M={m} N={n} K=1024: {stats_bf16_pipe}")
        print(f"fp8_gemm_k1024_bf16_out_pipe_fused_stats_q M={m} N={n} K=1024: {stats_bf16_pipe_fused_q}")
        print(f"fp8_gemm_k1024_bf16_out_pipe64 M={m} N={n} K=1024: {stats_bf16_pipe64}")
        print(f"te_general_gemm_fp8_tn_bf16_out_per_tensor M={m} N={n} K=1024: {stats_te}")
        print(f"tk_bf16_out_wide_scaled_te_raw M={m} N={n} K=1024: {stats_tk_te_raw}")


if __name__ == "__main__":
    main()
