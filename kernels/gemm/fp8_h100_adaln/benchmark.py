from __future__ import annotations

import argparse
import statistics

import torch

from _C import fp8_gemm_k1024, fp8_gemm_k1024_fp32_out, ln_adaln_quantize_k1024, ln_adaln_quantize_stats_k1024


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 3072, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
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
        w = (torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
        stats = event_bench(lambda: fp8_gemm_k1024(q, w), warmup=args.warmup, iters=args.iters)
        stats_fused_q = event_bench(lambda: fp8_gemm_k1024(q_stats, w), warmup=args.warmup, iters=args.iters)
        stats_fp32 = event_bench(lambda: fp8_gemm_k1024_fp32_out(q, w), warmup=args.warmup, iters=args.iters)
        stats_fp32_fused_q = event_bench(lambda: fp8_gemm_k1024_fp32_out(q_stats, w), warmup=args.warmup, iters=args.iters)
        flops = 2.0 * m * n * k
        stats["tflops"] = flops / (stats["mean_ms"] * 1e-3) / 1e12
        stats_fused_q["tflops"] = flops / (stats_fused_q["mean_ms"] * 1e-3) / 1e12
        stats_fp32["tflops"] = flops / (stats_fp32["mean_ms"] * 1e-3) / 1e12
        stats_fp32_fused_q["tflops"] = flops / (stats_fp32_fused_q["mean_ms"] * 1e-3) / 1e12
        print(f"fp8_gemm_k1024 M={m} N={n} K=1024: {stats}")
        print(f"fp8_gemm_k1024_fused_stats_q M={m} N={n} K=1024: {stats_fused_q}")
        print(f"fp8_gemm_k1024_fp32_out M={m} N={n} K=1024: {stats_fp32}")
        print(f"fp8_gemm_k1024_fp32_out_fused_stats_q M={m} N={n} K=1024: {stats_fp32_fused_q}")


if __name__ == "__main__":
    main()
