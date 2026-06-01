from __future__ import annotations

import argparse
import statistics

import torch
import transformer_engine  # noqa: F401
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

from _C import (
    bf16_quantize_rowwise_transpose_delayed,
    gate_bwd_quantize_rowwise_transpose_delayed,
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


def print_stats(label: str, stats: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    ratio = "" if baseline is None else f" speedup={baseline['mean_ms'] / stats['mean_ms']:.3f}x"
    print(
        f"{label}: mean={stats['mean_ms']:.6f} ms median={stats['median_ms']:.6f} "
        f"min={stats['min_ms']:.6f} max={stats['max_ms']:.6f}{ratio}",
        flush=True,
    )


def make_te_quantizer(device: torch.device, scale: float) -> Float8Quantizer:
    return Float8Quantizer(
        scale=torch.full((1,), scale, device=device, dtype=torch.float32),
        amax=torch.zeros((1,), device=device, dtype=torch.float32),
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )


def separate_torch_tk(grad_out, branch_out, gate, quant_scale, tokens: int):
    bsz, hidden = gate.shape
    dbranch = (grad_out.reshape(bsz, tokens, hidden) * gate[:, None, :]).reshape(bsz * tokens, hidden).contiguous()
    dgate = (grad_out.reshape(bsz, tokens, hidden).float() * branch_out.reshape(bsz, tokens, hidden).float()).sum(dim=1)
    q, q_t, row_amax = bf16_quantize_rowwise_transpose_delayed(dbranch, quant_scale)
    return q, q_t, row_amax, dgate


def separate_torch_te(grad_out, branch_out, gate, quantizer, tokens: int):
    bsz, hidden = gate.shape
    dbranch = (grad_out.reshape(bsz, tokens, hidden) * gate[:, None, :]).reshape(bsz * tokens, hidden).contiguous()
    dgate = (grad_out.reshape(bsz, tokens, hidden).float() * branch_out.reshape(bsz, tokens, hidden).float()).sum(dim=1)
    q = quantizer(dbranch)
    return q, dgate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--hidden", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--quant-scale", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    quant_scale = torch.full((1,), args.quant_scale, device=device, dtype=torch.float32)

    for tokens in args.tokens:
        for hidden in args.hidden:
            rows = args.batch * tokens
            print(f"\nshape: B={args.batch} T={tokens} rows={rows} H={hidden}", flush=True)
            grad_out = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)
            branch_out = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)
            gate = torch.randn((args.batch, hidden), device=device, dtype=torch.bfloat16) * 0.02
            quantizer = make_te_quantizer(device, args.quant_scale)

            fused = gate_bwd_quantize_rowwise_transpose_delayed(grad_out, branch_out, gate, quant_scale, tokens)
            sep_tk = separate_torch_tk(grad_out, branch_out, gate, quant_scale, tokens)
            sep_te = separate_torch_te(grad_out, branch_out, gate, quantizer, tokens)
            torch.cuda.synchronize()

            ref_dbranch_f32 = grad_out.reshape(args.batch, tokens, hidden).float() * gate[:, None, :].float()
            ref_dbranch = ref_dbranch_f32.reshape(rows, hidden).contiguous()
            ref_q = (ref_dbranch * args.quant_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            ref_dgate = (grad_out.reshape(args.batch, tokens, hidden).float() * branch_out.reshape(args.batch, tokens, hidden).float()).sum(dim=1)
            torch.testing.assert_close(fused[0], ref_q, rtol=0, atol=0)
            torch.testing.assert_close(fused[1], ref_q.T.contiguous(), rtol=0, atol=0)
            torch.testing.assert_close(fused[2], ref_dbranch.abs().amax(dim=1), rtol=0, atol=0)
            torch.testing.assert_close(fused[3], ref_dgate, rtol=1e-4, atol=1e-3)
            torch.testing.assert_close(sep_tk[3], fused[3], rtol=1e-4, atol=1e-3)

            torch_tk_stats = event_bench(
                lambda: separate_torch_tk(grad_out, branch_out, gate, quant_scale, tokens),
                warmup=args.warmup,
                iters=args.iters,
            )
            torch_te_stats = event_bench(
                lambda: separate_torch_te(grad_out, branch_out, gate, quantizer, tokens),
                warmup=args.warmup,
                iters=args.iters,
            )
            fused_stats = event_bench(
                lambda: gate_bwd_quantize_rowwise_transpose_delayed(grad_out, branch_out, gate, quant_scale, tokens),
                warmup=args.warmup,
                iters=args.iters,
            )
            print_stats("separate_torch_gate_dgate_plus_tk_quant_transpose", torch_tk_stats)
            print_stats("separate_torch_gate_dgate_plus_te_quant_transpose", torch_te_stats)
            print_stats("tk_fused_gate_bwd_quant_transpose", fused_stats, torch_tk_stats)
            print(f"te_baseline_to_fused_speedup={torch_te_stats['mean_ms'] / fused_stats['mean_ms']:.3f}x", flush=True)


if __name__ == "__main__":
    main()
