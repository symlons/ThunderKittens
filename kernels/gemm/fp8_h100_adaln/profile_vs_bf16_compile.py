from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from _C import fp8_gemm_k1024, fp8_gemm_k1024_fp32_out, ln_adaln_quantize_k1024, ln_adaln_quantize_stats_k1024


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


def compile_bench(fn):
    return torch.compile(fn, fullgraph=True, mode="max-autotune-no-cudagraphs")


def bf16_ln_adaln_linear(x3, shift, scale, weight, eps: float):
    y = F.layer_norm(x3, (x3.shape[-1],), eps=eps)
    y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return F.linear(y, weight)


def bf16_adaln_linear_from_stats(x2, shift, scale, mean, rstd, weight, tokens: int):
    batch = shift.shape[0]
    y = (x2.float() - mean[:, None]) * rstd[:, None]
    y = y.to(torch.bfloat16).reshape(batch, tokens, x2.shape[-1])
    y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return F.linear(y, weight)


def row_stats(x2, eps: float):
    xf = x2.float()
    mean = xf.mean(dim=1)
    var = (xf - mean[:, None]).square().mean(dim=1)
    return mean, torch.rsqrt(var + eps)


def print_stats(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label}: mean={stats['mean_ms']:.6f} ms "
        f"median={stats['median_ms']:.6f} min={stats['min_ms']:.6f} max={stats['max_ms']:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 3072, 4096])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    batch = args.batch
    tokens = args.tokens
    m = batch * tokens
    k = 1024
    assert tokens == 1024, "This profile is for the requested T=1024 shape."

    x3 = torch.randn((batch, tokens, k), device="cuda", dtype=torch.bfloat16)
    x2 = x3.reshape(m, k).contiguous()
    shift = torch.randn((batch, k), device="cuda", dtype=torch.bfloat16) * 0.02
    scale = torch.randn((batch, k), device="cuda", dtype=torch.bfloat16) * 0.02
    mean, rstd = row_stats(x2, args.eps)

    quant_scale = 1.0
    q, global_amax = ln_adaln_quantize_k1024(x2, shift, scale, mean, rstd, tokens, quant_scale)
    q_fused_stats, global_amax_fused_stats, _, _ = ln_adaln_quantize_stats_k1024(
        x2, shift, scale, tokens, quant_scale, args.eps
    )
    torch.cuda.synchronize()
    print(
        f"shape: B={batch} T={tokens} M={m} K={k}; "
        f"precomputed_stats_amax={global_amax[0].item():.6g} "
        f"fused_stats_amax={global_amax_fused_stats[0].item():.6g}",
        flush=True,
    )

    stats_only = event_bench(lambda: row_stats(x2, args.eps), warmup=args.warmup, iters=args.iters)
    quant_only = event_bench(
        lambda: ln_adaln_quantize_k1024(x2, shift, scale, mean, rstd, tokens, quant_scale),
        warmup=args.warmup,
        iters=args.iters,
    )
    stats_quant = event_bench(
        lambda: ln_adaln_quantize_k1024(x2, shift, scale, *row_stats(x2, args.eps), tokens, quant_scale),
        warmup=args.warmup,
        iters=args.iters,
    )
    fused_stats_quant = event_bench(
        lambda: ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, quant_scale, args.eps),
        warmup=args.warmup,
        iters=args.iters,
    )
    print_stats("torch_row_stats_only", stats_only)
    print_stats("tk_ln_adaln_quant_only_precomputed_stats", quant_only)
    print_stats("torch_row_stats_plus_tk_quant", stats_quant)
    print_stats("tk_fused_ln_stats_adaln_quant", fused_stats_quant)

    compiled_full = compile_bench(bf16_ln_adaln_linear)
    compiled_from_stats = compile_bench(bf16_adaln_linear_from_stats)

    for n in args.n:
        print(f"\nN={n}", flush=True)
        w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
        w_fp8 = w_bf16.to(torch.float8_e4m3fn)

        # Trigger Inductor compilation before timing.
        compiled_full(x3, shift, scale, w_bf16, args.eps)
        compiled_from_stats(x2, shift, scale, mean, rstd, w_bf16, tokens)
        torch.cuda.synchronize()

        bf16_full = event_bench(
            lambda: compiled_full(x3, shift, scale, w_bf16, args.eps),
            warmup=args.warmup,
            iters=args.iters,
        )
        bf16_from_stats = event_bench(
            lambda: compiled_from_stats(x2, shift, scale, mean, rstd, w_bf16, tokens),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_gemm = event_bench(lambda: fp8_gemm_k1024(q, w_fp8), warmup=args.warmup, iters=args.iters)
        fp8_gemm_fused_stats_q = event_bench(
            lambda: fp8_gemm_k1024(q_fused_stats, w_fp8),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_quant_gemm = event_bench(
            lambda: fp8_gemm_k1024(
                ln_adaln_quantize_k1024(x2, shift, scale, mean, rstd, tokens, quant_scale)[0],
                w_fp8,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_fused_stats_quant_gemm = event_bench(
            lambda: fp8_gemm_k1024(
                ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, quant_scale, args.eps)[0],
                w_fp8,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_fused_stats_quant_gemm_fp32 = event_bench(
            lambda: fp8_gemm_k1024_fp32_out(
                ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, quant_scale, args.eps)[0],
                w_fp8,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_stats_quant_gemm = event_bench(
            lambda: fp8_gemm_k1024(
                ln_adaln_quantize_k1024(x2, shift, scale, *row_stats(x2, args.eps), tokens, quant_scale)[0],
                w_fp8,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        print_stats("bf16_compile_full_ln_adaln_linear", bf16_full)
        print_stats("bf16_compile_adaln_linear_precomputed_stats", bf16_from_stats)
        print_stats("tk_fp8_gemm_only_prequantized_input", fp8_gemm)
        print_stats("tk_fp8_gemm_only_fused_stats_quantized_input", fp8_gemm_fused_stats_q)
        print_stats("tk_fp8_quant_plus_gemm_precomputed_stats", fp8_quant_gemm)
        print_stats("torch_stats_plus_tk_fp8_quant_plus_gemm", fp8_stats_quant_gemm)
        print_stats("tk_fused_stats_quant_plus_fp8_gemm", fp8_fused_stats_quant_gemm)
        print_stats("tk_fused_stats_quant_plus_fp8_gemm_fp32_out", fp8_fused_stats_quant_gemm_fp32)
        print(
            "speed ratios vs bf16_compile_full: "
            f"gemm_only={bf16_full['mean_ms'] / fp8_gemm['mean_ms']:.3f}x "
            f"quant_gemm={bf16_full['mean_ms'] / fp8_quant_gemm['mean_ms']:.3f}x "
            f"torch_stats_quant_gemm={bf16_full['mean_ms'] / fp8_stats_quant_gemm['mean_ms']:.3f}x "
            f"tk_fused_stats_quant_gemm={bf16_full['mean_ms'] / fp8_fused_stats_quant_gemm['mean_ms']:.3f}x "
            f"tk_fused_stats_quant_gemm_fp32={bf16_full['mean_ms'] / fp8_fused_stats_quant_gemm_fp32['mean_ms']:.3f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
