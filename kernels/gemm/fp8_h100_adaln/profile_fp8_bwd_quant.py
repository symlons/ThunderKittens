from __future__ import annotations

import argparse
import statistics

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.gemm import general_gemm
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    bf16_quantize_delayed,
    bf16_quantize_rowwise_transpose_delayed,
    bf16_quantize_transpose_delayed,
    delayed_scaling_update,
    fp8_dgrad_2xacc_scaled,
    fp8_wgrad_2xacc_scaled,
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


def make_te_quantizer(device: torch.device, scale: float, *, rowwise: bool, columnwise: bool) -> Float8Quantizer:
    return Float8Quantizer(
        scale=torch.full((1,), scale, device=device, dtype=torch.float32),
        amax=torch.zeros((1,), device=device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=rowwise,
        columnwise=columnwise,
    )


def raw_e4m3(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        return t
    return t.view(torch.float8_e4m3fn)


def te_dgrad(w_fp8, dy_fp8, workspace: torch.Tensor | None = None) -> torch.Tensor:
    kwargs = dict(layout="NN", grad=True, out_dtype=torch.bfloat16, use_split_accumulator=True)
    try:
        out, *_ = general_gemm(w_fp8, dy_fp8, **kwargs)
    except TypeError as exc:
        if "workspace" not in str(exc):
            raise
        if workspace is None:
            workspace = torch.empty(4_194_304, dtype=torch.uint8, device=dy_fp8.device)
        out, *_ = general_gemm(w_fp8, dy_fp8, workspace, **kwargs)
    return out


def te_wgrad(x_fp8, dy_fp8, workspace: torch.Tensor | None = None) -> torch.Tensor:
    kwargs = dict(layout="NT", grad=True, out_dtype=torch.bfloat16, use_split_accumulator=True)
    try:
        out, *_ = general_gemm(x_fp8, dy_fp8, **kwargs)
    except TypeError as exc:
        if "workspace" not in str(exc):
            raise
        if workspace is None:
            workspace = torch.empty(4_194_304, dtype=torch.uint8, device=dy_fp8.device)
        out, *_ = general_gemm(x_fp8, dy_fp8, workspace, **kwargs)
    return out


def tflops(mean_ms: float, m: int, n: int, k: int) -> float:
    return (2.0 * m * n * k) / (mean_ms * 1e-3) / 1e12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--quant-scale", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    m = args.m
    k = args.k
    scale = torch.full((1,), args.quant_scale, device=device, dtype=torch.float32)
    scale_inv = torch.full((1,), 1.0 / args.quant_scale, device=device, dtype=torch.float32)
    amax_history = torch.zeros((16,), device=device, dtype=torch.float32)
    hist_idx = torch.zeros((1,), device=device, dtype=torch.int32)

    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)

    q_row, row_amax = bf16_quantize_delayed(x, scale)
    q_t, row_amax_t = bf16_quantize_transpose_delayed(x, scale)
    q_both, q_both_t, row_amax_both = bf16_quantize_rowwise_transpose_delayed(x, scale)
    ref = (x.float() * args.quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(q_row, ref, rtol=0, atol=0)
    torch.testing.assert_close(q_t, ref.T.contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(q_both, ref, rtol=0, atol=0)
    torch.testing.assert_close(q_both_t, ref.T.contiguous(), rtol=0, atol=0)
    torch.testing.assert_close(row_amax, x.float().abs().amax(dim=1), rtol=0, atol=0)
    torch.testing.assert_close(row_amax_t, row_amax, rtol=0, atol=0)
    torch.testing.assert_close(row_amax_both, row_amax, rtol=0, atol=0)

    te_row_q = make_te_quantizer(device, args.quant_scale, rowwise=True, columnwise=False)
    te_col_q = make_te_quantizer(device, args.quant_scale, rowwise=True, columnwise=True)
    te_both_q = te_col_q
    te_x_row = te_row_q(x)
    te_x_col = te_col_q(x)
    te_x_both = te_both_q(x)
    torch.cuda.synchronize()

    print(f"shape: M={m} K={k}; delayed quant_scale={args.quant_scale}", flush=True)
    for label, fn in [
        ("te_bf16_quant_rowwise", lambda: te_row_q(x)),
        ("te_bf16_quant_rowwise_and_columnwise", lambda: te_col_q(x)),
        ("tk_bf16_quant_rowwise", lambda: bf16_quantize_delayed(x, scale)),
        ("tk_bf16_quant_columnwise_transpose", lambda: bf16_quantize_transpose_delayed(x, scale)),
        ("tk_bf16_quant_rowwise_and_columnwise", lambda: bf16_quantize_rowwise_transpose_delayed(x, scale)),
        ("tk_delayed_amax_history_scale_update", lambda: delayed_scaling_update(row_amax, scale, scale_inv, amax_history, hist_idx, args.eps)),
    ]:
        print_stats(label, event_bench(fn, warmup=args.warmup, iters=args.iters))

    for n in args.n:
        print(f"\nbackward shape: M={m} N={n} K={k}", flush=True)
        dy = torch.randn((m, n), device=device, dtype=torch.bfloat16)
        w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
        dy_scale = torch.full((1,), args.quant_scale, device=device, dtype=torch.float32)
        w_scale = torch.full((1,), args.quant_scale, device=device, dtype=torch.float32)
        x_scale = torch.full((1,), args.quant_scale, device=device, dtype=torch.float32)

        dy_row, dy_t, dy_amax = bf16_quantize_rowwise_transpose_delayed(dy, dy_scale)
        x_t, x_amax = bf16_quantize_transpose_delayed(x, x_scale)
        w_t, w_amax = bf16_quantize_transpose_delayed(w, w_scale)
        tk_dx = fp8_dgrad_2xacc_scaled(dy_row, w_t, 1.0 / args.quant_scale, 1.0 / args.quant_scale)
        tk_dw = fp8_wgrad_2xacc_scaled(x_t, dy_t, 1.0 / args.quant_scale, 1.0 / args.quant_scale)

        dy_q = make_te_quantizer(device, args.quant_scale, rowwise=True, columnwise=True)
        x_q = make_te_quantizer(device, args.quant_scale, rowwise=True, columnwise=True)
        w_q = make_te_quantizer(device, args.quant_scale, rowwise=True, columnwise=True)
        te_dy = dy_q(dy)
        te_x = x_q(x)
        te_w = w_q(w)
        workspace = torch.empty(4_194_304, dtype=torch.uint8, device=device)
        te_dx = te_dgrad(te_w, te_dy, workspace)
        te_dw = te_wgrad(te_x, te_dy, workspace)
        torch.cuda.synchronize()

        dx_ref = dy.float() @ w.float()
        dw_ref = x.float().T @ dy.float()
        te_dw_ref = dy.float().T @ x.float()
        print(
            "correctness: "
            f"tk_dx_max={((tk_dx.float() - dx_ref).abs().max().item()):.6g} "
            f"tk_dw_KN_max={((tk_dw.float() - dw_ref).abs().max().item()):.6g} "
            f"te_dx_max={((te_dx.float() - dx_ref).abs().max().item()):.6g} "
            f"te_dw_NK_max={((te_dw.float() - te_dw_ref).abs().max().item()):.6g}",
            flush=True,
        )

        tk_prep = event_bench(
            lambda: (
                bf16_quantize_rowwise_transpose_delayed(dy, dy_scale),
                bf16_quantize_transpose_delayed(x, x_scale),
                bf16_quantize_transpose_delayed(w, w_scale),
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_prep = event_bench(
            lambda: (dy_q(dy), x_q(x), w_q(w)),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_dgrad_full = event_bench(
            lambda: (
                lambda qdy, qw: fp8_dgrad_2xacc_scaled(qdy[0], qw[0], 1.0 / args.quant_scale, 1.0 / args.quant_scale)
            )(
                bf16_quantize_delayed(dy, dy_scale),
                bf16_quantize_transpose_delayed(w, w_scale),
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_wgrad_full = event_bench(
            lambda: (
                lambda qx, qdy: fp8_wgrad_2xacc_scaled(qx[0], qdy[0], 1.0 / args.quant_scale, 1.0 / args.quant_scale)
            )(
                bf16_quantize_transpose_delayed(x, x_scale),
                bf16_quantize_transpose_delayed(dy, dy_scale),
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        tk_both_full = event_bench(
            lambda: (
                lambda qdy, qx, qw: (
                    fp8_dgrad_2xacc_scaled(qdy[0], qw[0], 1.0 / args.quant_scale, 1.0 / args.quant_scale),
                    fp8_wgrad_2xacc_scaled(qx[0], qdy[1], 1.0 / args.quant_scale, 1.0 / args.quant_scale),
                )
            )(
                bf16_quantize_rowwise_transpose_delayed(dy, dy_scale),
                bf16_quantize_transpose_delayed(x, x_scale),
                bf16_quantize_transpose_delayed(w, w_scale),
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_dgrad_full = event_bench(
            lambda: te_dgrad(w_q(w), dy_q(dy), workspace),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_wgrad_full = event_bench(
            lambda: te_wgrad(x_q(x), dy_q(dy), workspace),
            warmup=args.warmup,
            iters=args.iters,
        )
        te_both_full = event_bench(
            lambda: (
                lambda qdy, qx, qw: (te_dgrad(qw, qdy, workspace), te_wgrad(qx, qdy, workspace))
            )(dy_q(dy), x_q(x), w_q(w)),
            warmup=args.warmup,
            iters=args.iters,
        )

        rows = [
            ("te_bwd_prep_only_dy_both_xT_wT", te_prep, None, "prep"),
            ("tk_bwd_prep_only_dy_both_xT_wT", tk_prep, None, "prep"),
            ("te_dgrad_quant_transpose_plus_gemm", te_dgrad_full, (m, k, n), "dgrad"),
            ("tk_dgrad_quant_transpose_plus_2xacc_gemm", tk_dgrad_full, (m, k, n), "dgrad"),
            ("te_wgrad_quant_transpose_plus_gemm", te_wgrad_full, (k, n, m), "wgrad"),
            ("tk_wgrad_quant_transpose_plus_2xacc_gemm", tk_wgrad_full, (k, n, m), "wgrad"),
            ("te_dgrad_wgrad_shared_prep_plus_gemms", te_both_full, None, "both"),
            ("tk_dgrad_wgrad_shared_prep_plus_2xacc_gemms", tk_both_full, None, "both"),
        ]
        print("| case | mean ms | min ms | TFLOP/s | tk/te vs pair |")
        print("|---|---:|---:|---:|---:|")
        pair_baseline: dict[str, float] = {}
        for label, stats, dims, pair in rows:
            if label.startswith("te_"):
                pair_baseline[pair] = stats["mean_ms"]
            flop_text = ""
            if dims is not None:
                flop_text = f"{tflops(stats['mean_ms'], *dims):.2f}"
            elif "dgrad_wgrad" in label:
                flop_text = f"{(2.0 * 2.0 * m * n * k) / (stats['mean_ms'] * 1e-3) / 1e12:.2f}"
            ratio = ""
            if label.startswith("tk_") and pair in pair_baseline:
                ratio = f"{pair_baseline[pair] / stats['mean_ms']:.3f}x"
            print(f"| {label} | {stats['mean_ms']:.6f} | {stats['min_ms']:.6f} | {flop_text} | {ratio} |", flush=True)


if __name__ == "__main__":
    main()
