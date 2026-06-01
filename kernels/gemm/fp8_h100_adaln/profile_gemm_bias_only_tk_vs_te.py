from __future__ import annotations

import argparse
import statistics

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    fp8_gemm_k1024_bf16_out_bias,
    fp8_gemm_k1024_bf16_out_bias_sm4,
    fp8_gemm_k1024_bf16_out_bias_sm8,
    fp8_gemm_k1024_bf16_out_bias_sm12,
    fp8_gemm_k1024_bf16_out_bias_sm16,
    fp8_gemm_k1024_bf16_out_bias_b_evict_last,
    fp8_gemm_k1024_bf16_out_bias_a_first_b_last,
    fp8_gemm_k1024_bf16_out_bias_store_evict_first,
    fp8_gemm_k1024_bf16_out_bias_all_cache_hints,
    fp8_gemm_k1024_bf16_out_bias_cluster2,
    fp8_gemm_k4096_bf16_out_bias,
)


K1024_TK_VARIANTS = {
    "auto": fp8_gemm_k1024_bf16_out_bias,
    "sm4": fp8_gemm_k1024_bf16_out_bias_sm4,
    "sm8": fp8_gemm_k1024_bf16_out_bias_sm8,
    "sm12": fp8_gemm_k1024_bf16_out_bias_sm12,
    "sm16": fp8_gemm_k1024_bf16_out_bias_sm16,
    "b_last": fp8_gemm_k1024_bf16_out_bias_b_evict_last,
    "a_first_b_last": fp8_gemm_k1024_bf16_out_bias_a_first_b_last,
    "store_first": fp8_gemm_k1024_bf16_out_bias_store_evict_first,
    "all_hints": fp8_gemm_k1024_bf16_out_bias_all_cache_hints,
}


def event_bench(fn, *, warmup: int, iters: int) -> dict[str, float]:
    times: list[float] = []
    for i in range(warmup + iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
        del out
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }


def tflops(ms: float, m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k / (ms * 1e-3) / 1e12


def make_fp8(x: torch.Tensor):
    quantizer = Float8Quantizer(
        scale=torch.ones((1,), device=x.device, dtype=torch.float32),
        amax=torch.zeros((1,), device=x.device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    return quantizer(x)


def te_gemm_bias(w_fp8, x_fp8, bias):
    out, *_ = general_gemm(w_fp8, x_fp8, out_dtype=torch.bfloat16, layout="TN", bias=bias)
    return out


def run_case(name: str, m: int, k: int, n: int, warmup: int, iters: int, tk_variant: str) -> None:
    x_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16) * 0.02
    x_te = make_fp8(x_bf16)
    w_te = make_fp8(w_bf16)
    x_tk = x_te._data.view(torch.float8_e4m3fn)
    w_tk = w_te._data.view(torch.float8_e4m3fn)

    if k == 1024:
        tk_kernel = K1024_TK_VARIANTS[tk_variant]
        tk_fn = lambda: tk_kernel(x_tk, w_tk, bias)
    elif k == 4096:
        tk_variant = "k4096"
        tk_fn = lambda: fp8_gemm_k4096_bf16_out_bias(x_tk, w_tk, bias)
    else:
        raise ValueError(k)
    te_fn = lambda: te_gemm_bias(w_te, x_te, bias)

    tk_out = tk_fn()
    te_out = te_fn()
    torch.cuda.synchronize()
    max_diff = (tk_out.float() - te_out.float()).abs().max().item()

    tk = event_bench(tk_fn, warmup=warmup, iters=iters)
    te = event_bench(te_fn, warmup=warmup, iters=iters)
    print(
        f"{name}: M={m} K={k} N={n} tk_variant={tk_variant} "
        f"tk_mean={tk['mean']:.6f}ms tk_min={tk['min']:.6f}ms tk_tflops={tflops(tk['mean'], m, n, k):.1f} "
        f"te_mean={te['mean']:.6f}ms te_min={te['min']:.6f}ms te_tflops={tflops(te['mean'], m, n, k):.1f} "
        f"tk_vs_te={te['mean'] / tk['mean']:.3f}x max_diff={max_diff:.6g}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TK FP8 GEMM+bias with Transformer Engine/cuBLASLt.")
    parser.add_argument("--batches", type=int, nargs="+", default=[24, 32, 64, 96, 128, 256])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--cases", nargs="+", default=["qkv", "attn_out", "fc1", "fc2"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--tk-variant", choices=sorted(K1024_TK_VARIANTS), default="auto")
    parser.add_argument("--sweep-tk-variants", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    all_cases = {
        "qkv": ("qkv_gemm_bias", 1024, 3072),
        "attn_out": ("attn_out_gemm_bias", 1024, 1024),
        "fc1": ("fc1_gemm_bias", 1024, 4096),
        "fc2": ("fc2_gemm_bias", 4096, 1024),
    }
    cases = [all_cases[name] for name in args.cases]
    for batch in args.batches:
        print(f"\nB={batch} T={args.tokens}", flush=True)
        m = batch * args.tokens
        for name, k, n in cases:
            variants = [v for v in sorted(K1024_TK_VARIANTS) if v != "auto"] if args.sweep_tk_variants and k == 1024 else [args.tk_variant]
            for tk_variant in variants:
                run_case(name, m, k, n, args.warmup, args.iters, tk_variant)


if __name__ == "__main__":
    main()
