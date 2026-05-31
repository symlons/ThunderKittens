from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from _C import fp8_gemm_k4096_bf16_out_bias as raw_fp8_gemm_k4096_bf16_out_bias
from tk_compile_ops import fp8_gemm_k4096_bf16_out_bias


def event_bench(fn, *, warmup: int, iters: int) -> dict[str, float]:
    times: list[float] = []
    for idx in range(warmup + iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if idx >= warmup:
            times.append(start.elapsed_time(end))
        del out
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def print_stats(label: str, stats: dict[str, float], baseline_ms: float | None = None) -> None:
    ratio = "" if baseline_ms is None else f" vs_bf16={baseline_ms / stats['mean_ms']:.3f}x"
    print(
        f"{label}: mean={stats['mean_ms']:.6f} ms "
        f"median={stats['median_ms']:.6f} min={stats['min_ms']:.6f} max={stats['max_ms']:.6f}{ratio}",
        flush=True,
    )


def bf16_linear(x, w, b):
    return F.linear(x, w, b)


def tk_mlp_down_raw(x, w_fp8, b):
    q = x.contiguous().to(torch.float8_e4m3fn)
    return raw_fp8_gemm_k4096_bf16_out_bias(q, w_fp8, b)


def tk_mlp_down_compile_op(x, w_fp8, b):
    q = x.contiguous().to(torch.float8_e4m3fn)
    return fp8_gemm_k4096_bf16_out_bias(q, w_fp8, b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    hidden = args.hidden
    mlp_hidden = hidden * args.mlp_ratio
    assert hidden == 1024
    assert mlp_hidden == 4096

    max_m = args.batch * max(args.tokens)
    x_all = torch.randn((max_m, mlp_hidden), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((hidden, mlp_hidden), device="cuda", dtype=torch.bfloat16) * (1.0 / math.sqrt(mlp_hidden))
    b = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16) * 0.02
    w_fp8 = w.to(torch.float8_e4m3fn)

    bf16_compiled = torch.compile(bf16_linear, fullgraph=True, mode="max-autotune-no-cudagraphs")
    tk_compiled = torch.compile(tk_mlp_down_compile_op, fullgraph=True, mode="max-autotune-no-cudagraphs")

    print("scope: MLP-down only, shape M x 4096 -> M x 1024, fused bias in TK via init_bias", flush=True)
    for tokens in args.tokens:
        m = args.batch * tokens
        x = x_all[:m].contiguous()
        q = x.to(torch.float8_e4m3fn)
        print(f"\nshape: M={m} K={mlp_hidden} N={hidden}", flush=True)

        bf16_compiled(x, w, b)
        tk_mlp_down_raw(x, w_fp8, b)
        tk_compiled(x, w_fp8, b)
        raw_fp8_gemm_k4096_bf16_out_bias(q, w_fp8, b)
        torch.cuda.synchronize()

        bf16_stats = event_bench(lambda: bf16_compiled(x, w, b), warmup=args.warmup, iters=args.iters)
        cast_stats = event_bench(lambda: x.to(torch.float8_e4m3fn), warmup=args.warmup, iters=args.iters)
        raw_stats = event_bench(lambda: tk_mlp_down_raw(x, w_fp8, b), warmup=args.warmup, iters=args.iters)
        compile_stats = event_bench(lambda: tk_compiled(x, w_fp8, b), warmup=args.warmup, iters=args.iters)
        gemm_only_stats = event_bench(lambda: raw_fp8_gemm_k4096_bf16_out_bias(q, w_fp8, b), warmup=args.warmup, iters=args.iters)

        print_stats("bf16_compile_linear", bf16_stats)
        print_stats("bf16_to_fp8_cast_only", cast_stats, bf16_stats["mean_ms"])
        print_stats("tk_cast_plus_k4096_bias_gemm_raw", raw_stats, bf16_stats["mean_ms"])
        print_stats("tk_cast_plus_k4096_bias_gemm_compile_op", compile_stats, bf16_stats["mean_ms"])
        print_stats("tk_k4096_bias_gemm_only_prequantized", gemm_only_stats, bf16_stats["mean_ms"])


if __name__ == "__main__":
    main()
