from __future__ import annotations

import statistics

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    fp8_gemm_k1024_bf16_out_wide_scaled,
    fp8_gemm_k4096_bf16_out_bias,
)


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
    q = Float8Quantizer(
        scale=torch.ones((1,), device=x.device, dtype=torch.float32),
        amax=torch.zeros((1,), device=x.device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    return q(x)


def te_gemm(w_fp8, x_fp8, workspace):
    try:
        out, *_ = general_gemm(w_fp8, x_fp8, out_dtype=torch.bfloat16, layout="TN")
    except TypeError as exc:
        if "workspace" not in str(exc):
            raise
        out, *_ = general_gemm(w_fp8, x_fp8, workspace, out_dtype=torch.bfloat16, layout="TN")
    return out


def run_case(name: str, m: int, k: int, n: int, warmup: int, iters: int) -> None:
    x_bf16 = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.02
    w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
    x_te = make_fp8(x_bf16)
    w_te = make_fp8(w_bf16)
    x_tk = x_te._data.view(torch.float8_e4m3fn)
    w_tk = w_te._data.view(torch.float8_e4m3fn)
    workspace = torch.empty(4_194_304, dtype=torch.uint8, device="cuda")

    if k == 1024:
        tk_fn = lambda: fp8_gemm_k1024_bf16_out_wide_scaled(x_tk, w_tk, 1.0, 1.0)
    elif k == 4096:
        # The current custom K=4096 forward entry point includes a fused bias epilogue.
        bias = torch.zeros((n,), device="cuda", dtype=torch.bfloat16)
        tk_fn = lambda: fp8_gemm_k4096_bf16_out_bias(x_tk, w_tk, bias)
    else:
        raise ValueError(k)

    te_fn = lambda: te_gemm(w_te, x_te, workspace)

    tk_fn()
    te_fn()
    torch.cuda.synchronize()

    tk = event_bench(tk_fn, warmup=warmup, iters=iters)
    te = event_bench(te_fn, warmup=warmup, iters=iters)
    print(
        f"{name}: M={m} K={k} N={n} "
        f"tk_mean={tk['mean']:.6f}ms tk_min={tk['min']:.6f}ms tk_tflops={tflops(tk['mean'], m, n, k):.1f} "
        f"te_mean={te['mean']:.6f}ms te_min={te['min']:.6f}ms te_tflops={tflops(te['mean'], m, n, k):.1f} "
        f"tk_vs_te={te['mean'] / tk['mean']:.3f}x",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    warmup = 10
    iters = 30
    tokens = 1024
    cases = [
        ("qkv_fwd_gemm", 1024, 3072),
        ("attn_out_fwd_gemm", 1024, 1024),
        ("fc1_fwd_gemm", 1024, 4096),
        ("fc2_fwd_gemm", 4096, 1024),
    ]
    for batch in [4, 8, 16, 32, 64]:
        print(f"\nB={batch} T={tokens}", flush=True)
        m = batch * tokens
        for name, k, n in cases:
            run_case(name, m, k, n, warmup, iters)


if __name__ == "__main__":
    main()
