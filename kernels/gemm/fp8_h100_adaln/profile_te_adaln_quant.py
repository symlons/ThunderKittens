from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    fp8_gemm_k1024_bf16_out_wide_scaled,
    ln_adaln_quantize_stats_k1024,
)


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


def print_stats(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label}: mean={stats['mean_ms']:.6f} ms "
        f"median={stats['median_ms']:.6f} min={stats['min_ms']:.6f} max={stats['max_ms']:.6f}",
        flush=True,
    )


def tflops(mean_ms: float, m: int, n: int, k: int) -> float:
    return (2.0 * m * n * k) / (mean_ms * 1e-3) / 1e12


def compile_bench(fn):
    return torch.compile(fn, fullgraph=True, mode="max-autotune-no-cudagraphs")


def bf16_ln_adaln_linear(x3, shift, scale, weight, eps: float):
    y = F.layer_norm(x3, (x3.shape[-1],), eps=eps)
    y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return F.linear(y, weight)


def bf16_ln_adaln_2d(x3, shift, scale, eps: float):
    y = F.layer_norm(x3, (x3.shape[-1],), eps=eps)
    y = y * (1.0 + scale[:, None, :]) + shift[:, None, :]
    return y.reshape(-1, x3.shape[-1]).contiguous()


def make_te_quantizer(device: torch.device | str, scale: float = 1.0):
    return Float8Quantizer(
        scale=torch.full((1,), scale, device=device, dtype=torch.float32),
        amax=torch.zeros((1,), device=device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )


def raw_e4m3(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t
    return t.view(torch.float8_e4m3fn)


def te_general_gemm_tn_bf16(x_fp8, w_fp8, workspace: torch.Tensor | None) -> torch.Tensor:
    try:
        out, _, _, _ = general_gemm(w_fp8, x_fp8, out_dtype=torch.bfloat16, layout="TN")
    except TypeError as exc:
        if "workspace" not in str(exc):
            raise
        if workspace is None:
            workspace = torch.empty(4_194_304, dtype=torch.uint8, device=x_fp8.device)
        out, _, _, _ = general_gemm(w_fp8, x_fp8, workspace, out_dtype=torch.bfloat16, layout="TN")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 3072, 4096])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--quant-scale", type=float, default=1.0)
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

    compiled_full = compile_bench(bf16_ln_adaln_linear)
    compiled_producer = compile_bench(bf16_ln_adaln_2d)

    q_tk, tk_amax, _, _ = ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, args.quant_scale, args.eps)
    produced_adaln = compiled_producer(x3, shift, scale, args.eps)
    torch.cuda.synchronize()

    producer_only = event_bench(
        lambda: compiled_producer(x3, shift, scale, args.eps),
        warmup=args.warmup,
        iters=args.iters,
    )
    tk_quant_only = event_bench(
        lambda: ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, args.quant_scale, args.eps),
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"shape: B={batch} T={tokens} M={m} K={k}; "
        f"tk_fused_stats_amax={tk_amax[0].item():.6g}",
        flush=True,
    )
    print_stats("bf16_compile_ln_adaln_producer_only", producer_only)
    print_stats("tk_fused_ln_stats_adaln_quant_only", tk_quant_only)

    for n in args.n:
        print(f"\nN={n}", flush=True)
        w_bf16 = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.02
        w_raw_fp8 = w_bf16.to(torch.float8_e4m3fn)
        act_quantizer = make_te_quantizer(x2.device, args.quant_scale)
        weight_quantizer = make_te_quantizer(x2.device, args.quant_scale)
        te_workspace = torch.empty(4_194_304, dtype=torch.uint8, device=x2.device)

        te_x_fp8 = act_quantizer(produced_adaln.float() / args.quant_scale)
        te_w_fp8 = weight_quantizer(w_bf16.float() / args.quant_scale)
        te_x_scale_inv = float(te_x_fp8._scale_inv.item())
        te_w_scale_inv = float(te_w_fp8._scale_inv.item())
        te_general_gemm_tn_bf16(te_x_fp8, te_w_fp8, te_workspace)
        compiled_full(x3, shift, scale, w_bf16, args.eps)
        fp8_gemm_k1024_bf16_out_wide_scaled(q_tk, w_raw_fp8, 1.0, 1.0)
        torch.cuda.synchronize()

        bf16_full = event_bench(
            lambda: compiled_full(x3, shift, scale, w_bf16, args.eps),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_gemm_only = event_bench(
            lambda: te_general_gemm_tn_bf16(te_x_fp8, te_w_fp8, te_workspace),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_act_quant_gemm = event_bench(
            lambda: te_general_gemm_tn_bf16(
                act_quantizer(produced_adaln.float() / args.quant_scale),
                te_w_fp8,
                te_workspace,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_act_weight_quant_gemm = event_bench(
            lambda: te_general_gemm_tn_bf16(
                act_quantizer(produced_adaln.float() / args.quant_scale),
                weight_quantizer(w_bf16.float() / args.quant_scale),
                te_workspace,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_full_prequant_weight = event_bench(
            lambda: te_general_gemm_tn_bf16(
                act_quantizer(compiled_producer(x3, shift, scale, args.eps).float() / args.quant_scale),
                te_w_fp8,
                te_workspace,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_full_quant_weight = event_bench(
            lambda: te_general_gemm_tn_bf16(
                act_quantizer(compiled_producer(x3, shift, scale, args.eps).float() / args.quant_scale),
                weight_quantizer(w_bf16.float() / args.quant_scale),
                te_workspace,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_gemm_only = event_bench(
            lambda: fp8_gemm_k1024_bf16_out_wide_scaled(q_tk, w_raw_fp8, 1.0, 1.0),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_full = event_bench(
            lambda: fp8_gemm_k1024_bf16_out_wide_scaled(
                ln_adaln_quantize_stats_k1024(x2, shift, scale, tokens, args.quant_scale, args.eps)[0],
                w_raw_fp8,
                1.0,
                1.0,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_on_te_raw = event_bench(
            lambda: fp8_gemm_k1024_bf16_out_wide_scaled(
                raw_e4m3(te_x_fp8._data),
                raw_e4m3(te_w_fp8._data),
                te_x_scale_inv,
                te_w_scale_inv,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )

        rows = [
            ("bf16_compile_full_ln_adaln_linear", bf16_full),
            ("te_gemm_only_preproduced_prequantized", te_gemm_only),
            ("te_act_quant_plus_gemm_preproduced", te_act_quant_gemm),
            ("te_act_weight_quant_plus_gemm_preproduced", te_act_weight_quant_gemm),
            ("te_compiled_adaln_act_quant_plus_gemm", te_full_prequant_weight),
            ("te_compiled_adaln_act_weight_quant_plus_gemm", te_full_quant_weight),
            ("tk_wide_scaled_gemm_only_prequantized", tk_gemm_only),
            ("tk_fused_stats_adaln_quant_plus_wide_scaled_gemm", tk_full),
            ("tk_wide_scaled_gemm_on_te_raw_fp8", tk_on_te_raw),
        ]
        print("| case | mean ms | TFLOP/s | vs bf16 compile |")
        print("|---|---:|---:|---:|")
        for label, stats in rows:
            print(
                f"| {label} | {stats['mean_ms']:.6f} | "
                f"{tflops(stats['mean_ms'], m, n, k):.2f} | "
                f"{bf16_full['mean_ms'] / stats['mean_ms']:.3f}x |",
                flush=True,
            )


if __name__ == "__main__":
    main()
