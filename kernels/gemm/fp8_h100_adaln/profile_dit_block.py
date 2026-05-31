from __future__ import annotations

import argparse
import math
import statistics

import torch
import torch.nn.functional as F

from _C import (
    bias_gelu_quantize_k4096 as raw_bias_gelu_quantize_k4096,
    fp8_gemm_k4096_bf16_out_bias as raw_fp8_gemm_k4096_bf16_out_bias,
    fp8_gemm_k1024_bf16_out_wide_scaled as raw_fp8_gemm_k1024_bf16_out_wide_scaled,
    ln_adaln_quantize_stats_vec_delayed_k1024 as raw_ln_adaln_quantize_stats_vec_delayed_k1024,
)
from tk_compile_ops import (
    bias_gelu_quantize_k4096,
    fp8_gemm_k4096_bf16_out_bias,
    fp8_gemm_k1024_bf16_out_wide_scaled,
    ln_adaln_quantize_stats_vec_delayed_k1024,
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


def print_stats(label: str, stats: dict[str, float], baseline_ms: float | None = None) -> None:
    ratio = "" if baseline_ms is None else f" vs_bf16={baseline_ms / stats['mean_ms']:.3f}x"
    print(
        f"{label}: mean={stats['mean_ms']:.6f} ms "
        f"median={stats['median_ms']:.6f} min={stats['min_ms']:.6f} max={stats['max_ms']:.6f}{ratio}",
        flush=True,
    )


def compile_bench(fn):
    return torch.compile(fn, fullgraph=True, mode="max-autotune-no-cudagraphs")


def modulate(x, shift, scale):
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def ada_chunks(c, ada_w, ada_b):
    chunks = F.linear(F.silu(c), ada_w, ada_b).chunk(6, dim=1)
    return tuple(t.contiguous() for t in chunks)


def bf16_dit_block(
    x,
    c,
    ada_w,
    ada_b,
    qkv_w,
    qkv_b,
    proj_w,
    proj_b,
    fc1_w,
    fc1_b,
    fc2_w,
    fc2_b,
    heads: int,
    eps: float,
):
    bsz, tokens, hidden = x.shape
    head_dim = hidden // heads
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_chunks(c, ada_w, ada_b)

    y = modulate(F.layer_norm(x, (hidden,), eps=eps), shift_msa, scale_msa)
    qkv = F.linear(y, qkv_w, qkv_b)
    qkv = qkv.reshape(bsz, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    attn = attn.transpose(1, 2).reshape(bsz, tokens, hidden)
    attn = F.linear(attn, proj_w, proj_b)
    x = x + gate_msa[:, None, :] * attn

    y = modulate(F.layer_norm(x, (hidden,), eps=eps), shift_mlp, scale_mlp)
    y = F.linear(y, fc1_w, fc1_b)
    y = F.gelu(y, approximate="tanh")
    y = F.linear(y, fc2_w, fc2_b)
    x = x + gate_mlp[:, None, :] * y
    return x


def fp8_hybrid_dit_block(
    x,
    c,
    ada_w,
    ada_b,
    qkv_w_fp8,
    qkv_b,
    proj_w,
    proj_b,
    fc1_w_fp8,
    fc1_b,
    fc2_w,
    fc2_b,
    quant_scale,
    heads: int,
    eps: float,
):
    bsz, tokens, hidden = x.shape
    head_dim = hidden // heads
    x2 = x.reshape(bsz * tokens, hidden).contiguous()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_chunks(c, ada_w, ada_b)

    q_msa, _ = ln_adaln_quantize_stats_vec_delayed_k1024(x2, shift_msa, scale_msa, quant_scale, tokens, eps)
    qkv = fp8_gemm_k1024_bf16_out_wide_scaled(q_msa, qkv_w_fp8, 1.0, 1.0)
    qkv = qkv + qkv_b
    qkv = qkv.reshape(bsz, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    attn = attn.transpose(1, 2).reshape(bsz, tokens, hidden)
    attn = F.linear(attn, proj_w, proj_b)
    x = x + gate_msa[:, None, :] * attn

    x2 = x.reshape(bsz * tokens, hidden).contiguous()
    q_mlp, _ = ln_adaln_quantize_stats_vec_delayed_k1024(x2, shift_mlp, scale_mlp, quant_scale, tokens, eps)
    y = fp8_gemm_k1024_bf16_out_wide_scaled(q_mlp, fc1_w_fp8, 1.0, 1.0)
    y_fp8, _ = bias_gelu_quantize_k4096(y, fc1_b, quant_scale)
    y = fp8_gemm_k4096_bf16_out_bias(y_fp8, fc2_w, fc2_b).reshape(bsz, tokens, hidden)
    x = x + gate_mlp[:, None, :] * y
    return x


def fp8_hybrid_dit_block_raw(
    x,
    c,
    ada_w,
    ada_b,
    qkv_w_fp8,
    qkv_b,
    proj_w,
    proj_b,
    fc1_w_fp8,
    fc1_b,
    fc2_w,
    fc2_b,
    quant_scale,
    heads: int,
    eps: float,
):
    bsz, tokens, hidden = x.shape
    head_dim = hidden // heads
    x2 = x.reshape(bsz * tokens, hidden).contiguous()
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_chunks(c, ada_w, ada_b)

    q_msa, _ = raw_ln_adaln_quantize_stats_vec_delayed_k1024(
        x2, shift_msa, scale_msa, quant_scale, tokens, eps
    )
    qkv = raw_fp8_gemm_k1024_bf16_out_wide_scaled(q_msa, qkv_w_fp8, 1.0, 1.0)
    qkv = qkv + qkv_b
    qkv = qkv.reshape(bsz, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    attn = attn.transpose(1, 2).reshape(bsz, tokens, hidden)
    attn = F.linear(attn, proj_w, proj_b)
    x = x + gate_msa[:, None, :] * attn

    x2 = x.reshape(bsz * tokens, hidden).contiguous()
    q_mlp, _ = raw_ln_adaln_quantize_stats_vec_delayed_k1024(
        x2, shift_mlp, scale_mlp, quant_scale, tokens, eps
    )
    y = raw_fp8_gemm_k1024_bf16_out_wide_scaled(q_mlp, fc1_w_fp8, 1.0, 1.0)
    y_fp8, _ = raw_bias_gelu_quantize_k4096(y, fc1_b, quant_scale)
    y = raw_fp8_gemm_k4096_bf16_out_bias(y_fp8, fc2_w, fc2_b).reshape(bsz, tokens, hidden)
    x = x + gate_mlp[:, None, :] * y
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    bsz = args.batch
    hidden = args.hidden
    heads = args.heads
    mlp_hidden = hidden * args.mlp_ratio
    assert hidden == 1024, "Current TK FP8 kernels are specialized for K=1024."
    assert hidden % heads == 0
    assert mlp_hidden % 256 == 0

    max_tokens = max(args.tokens)
    x_all = torch.randn((bsz, max_tokens, hidden), device="cuda", dtype=torch.bfloat16)
    c = torch.randn((bsz, hidden), device="cuda", dtype=torch.bfloat16)
    std = 1.0 / math.sqrt(hidden)
    ada_w = (torch.randn((6 * hidden, hidden), device="cuda", dtype=torch.bfloat16) * std)
    ada_b = torch.randn((6 * hidden,), device="cuda", dtype=torch.bfloat16) * 0.02
    qkv_w = torch.randn((3 * hidden, hidden), device="cuda", dtype=torch.bfloat16) * std
    qkv_b = torch.randn((3 * hidden,), device="cuda", dtype=torch.bfloat16) * 0.02
    proj_w = torch.randn((hidden, hidden), device="cuda", dtype=torch.bfloat16) * std
    proj_b = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16) * 0.02
    fc1_w = torch.randn((mlp_hidden, hidden), device="cuda", dtype=torch.bfloat16) * std
    fc1_b = torch.randn((mlp_hidden,), device="cuda", dtype=torch.bfloat16) * 0.02
    fc2_w = torch.randn((hidden, mlp_hidden), device="cuda", dtype=torch.bfloat16) * (1.0 / math.sqrt(mlp_hidden))
    fc2_b = torch.randn((hidden,), device="cuda", dtype=torch.bfloat16) * 0.02

    qkv_w_fp8 = qkv_w.to(torch.float8_e4m3fn)
    fc1_w_fp8 = fc1_w.to(torch.float8_e4m3fn)
    fc2_w_fp8 = fc2_w.to(torch.float8_e4m3fn)
    quant_scale = torch.ones((1,), device="cuda", dtype=torch.float32)

    bf16_compiled = compile_bench(bf16_dit_block)
    fp8_compiled = compile_bench(fp8_hybrid_dit_block)
    print(
        "scope: one DiTBlock forward; FP8 hybrid uses TK for AdaLN->QKV, AdaLN->MLP-up, fused bias+GELU(tanh.approx)+quant, and MLP-down+bias",
        flush=True,
    )
    for tokens in args.tokens:
        x = x_all[:, :tokens, :].contiguous()
        print(f"\nshape: B={bsz} T={tokens} H={hidden} heads={heads} mlp_hidden={mlp_hidden}", flush=True)
        bf16_compiled(
            x, c, ada_w, ada_b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, heads, args.eps
        )
        fp8_hybrid_dit_block_raw(
            x,
            c,
            ada_w,
            ada_b,
            qkv_w_fp8,
            qkv_b,
            proj_w,
            proj_b,
            fc1_w_fp8,
            fc1_b,
            fc2_w_fp8,
            fc2_b,
            quant_scale,
            heads,
            args.eps,
        )
        fp8_compiled(
            x,
            c,
            ada_w,
            ada_b,
            qkv_w_fp8,
            qkv_b,
            proj_w,
            proj_b,
            fc1_w_fp8,
            fc1_b,
            fc2_w_fp8,
            fc2_b,
            quant_scale,
            heads,
            args.eps,
        )
        torch.cuda.synchronize()

        bf16_stats = event_bench(
            lambda: bf16_compiled(
                x, c, ada_w, ada_b, qkv_w, qkv_b, proj_w, proj_b, fc1_w, fc1_b, fc2_w, fc2_b, heads, args.eps
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_stats = event_bench(
            lambda: fp8_hybrid_dit_block_raw(
                x,
                c,
                ada_w,
                ada_b,
                qkv_w_fp8,
                qkv_b,
                proj_w,
                proj_b,
                fc1_w_fp8,
                fc1_b,
                fc2_w_fp8,
                fc2_b,
                quant_scale,
                heads,
                args.eps,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        fp8_compile_stats = event_bench(
            lambda: fp8_compiled(
                x,
                c,
                ada_w,
                ada_b,
                qkv_w_fp8,
                qkv_b,
                proj_w,
                proj_b,
                fc1_w_fp8,
                fc1_b,
                fc2_w_fp8,
                fc2_b,
                quant_scale,
                heads,
                args.eps,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
        print_stats("bf16_compile_dit_block", bf16_stats)
        print_stats("tk_fp8_hybrid_raw_dit_block", fp8_stats, bf16_stats["mean_ms"])
        print_stats("tk_fp8_hybrid_compile_op_dit_block", fp8_compile_stats, bf16_stats["mean_ms"])


if __name__ == "__main__":
    main()
