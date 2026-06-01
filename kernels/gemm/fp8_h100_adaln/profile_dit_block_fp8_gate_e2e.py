from __future__ import annotations

import argparse
import statistics
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

from _C import gate_bwd_quantize_rowwise_transpose_delayed


class TkFp8GateResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gate, quant_scale, tokens: int):
        ctx.tokens = tokens
        ctx.save_for_backward(branch.contiguous(), gate.contiguous(), quant_scale)
        return x + gate[:, None, :] * branch

    @staticmethod
    def backward(ctx, grad_out):
        branch, gate, quant_scale = ctx.saved_tensors
        bsz, tokens, hidden = grad_out.shape
        q, _q_t, _row_amax, dgate = gate_bwd_quantize_rowwise_transpose_delayed(
            grad_out.reshape(bsz * tokens, hidden).contiguous(),
            branch.reshape(bsz * tokens, hidden).contiguous(),
            gate,
            quant_scale,
            ctx.tokens,
        )
        # This makes the rest of PyTorch autograd see the FP8-rounded branch gradient.
        dbranch = q.to(torch.bfloat16).reshape(bsz, tokens, hidden)
        return grad_out, dbranch, dgate.to(gate.dtype), None, None


def tk_fp8_gate_residual(x, branch, gate, quant_scale, tokens: int):
    return TkFp8GateResidual.apply(x, branch, gate, quant_scale, tokens)


class SimpleDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: int, *, fp8_gate_bwd: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.fp8_gate_bwd = fp8_gate_bwd
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * mlp_ratio,
            act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.register_buffer("quant_scale", torch.ones((1,), dtype=torch.float32), persistent=False)

    def _residual(self, x, branch, gate):
        if self.fp8_gate_bwd:
            return tk_fp8_gate_residual(x, branch, gate, self.quant_scale.float(), x.shape[1])
        return x + gate[:, None, :] * branch

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        y = self.norm1(x)
        y = y * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = self._residual(x, self.attn(y), gate_msa)
        y = self.norm2(x)
        y = y * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = self._residual(x, self.mlp(y), gate_mlp)
        return x


def event_bench(fn, *, warmup: int, iters: int):
    times = []
    for idx in range(warmup + iters):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        if idx >= warmup:
            times.append(s.elapsed_time(e))
    return {"mean": statistics.mean(times), "median": statistics.median(times), "min": min(times), "max": max(times)}


def zero_grads(block, x, c):
    x.grad = None
    c.grad = None
    for p in block.parameters():
        p.grad = None


def step(block, x, c, grad):
    out = block(x, c)
    out.backward(grad)
    zero_grads(block, x, c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    base = SimpleDiTBlock(args.hidden, args.heads, args.mlp_ratio, fp8_gate_bwd=False).to(device).to(torch.bfloat16).train()
    fp8 = SimpleDiTBlock(args.hidden, args.heads, args.mlp_ratio, fp8_gate_bwd=True).to(device).to(torch.bfloat16).train()
    fp8.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn((args.batch, args.tokens, args.hidden), device=device, dtype=torch.bfloat16)
    c0 = torch.randn((args.batch, args.hidden), device=device, dtype=torch.bfloat16)
    grad = torch.randn_like(x0)
    x = x0.detach().clone().requires_grad_(True)
    c = c0.detach().clone().requires_grad_(True)
    x_fp8 = x0.detach().clone().requires_grad_(True)
    c_fp8 = c0.detach().clone().requires_grad_(True)

    with torch.no_grad():
        fwd_diff = (base(x, c).float() - fp8(x_fp8, c_fp8).float()).abs()
    print(f"shape: B={args.batch} T={args.tokens} H={args.hidden} heads={args.heads}")
    print(f"forward_diff max={fwd_diff.max().item():.6g} mean={fwd_diff.mean().item():.6g}")

    step(base, x, c, grad)
    step(fp8, x_fp8, c_fp8, grad)
    torch.cuda.synchronize()

    base_stats = event_bench(lambda: step(base, x, c, grad), warmup=args.warmup, iters=args.iters)
    fp8_stats = event_bench(lambda: step(fp8, x_fp8, c_fp8, grad), warmup=args.warmup, iters=args.iters)
    print(
        f"bf16_gate_bwd_train: mean={base_stats['mean']:.6f} ms median={base_stats['median']:.6f} "
        f"min={base_stats['min']:.6f} max={base_stats['max']:.6f}"
    )
    print(
        f"tk_fp8_gate_bwd_train: mean={fp8_stats['mean']:.6f} ms median={fp8_stats['median']:.6f} "
        f"min={fp8_stats['min']:.6f} max={fp8_stats['max']:.6f} speedup={base_stats['mean']/fp8_stats['mean']:.3f}x"
    )


if __name__ == "__main__":
    main()
