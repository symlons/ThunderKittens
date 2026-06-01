from __future__ import annotations

import argparse
import statistics
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

from _C import (
    bf16_quantize_delayed,
    bf16_quantize_rowwise_transpose_delayed,
    bf16_quantize_rowwise_transpose_db_delayed,
    bf16_quantize_transpose_delayed,
    fp8_dgrad_2xacc_scaled,
    fp8_gemm_k1024_bf16_out_wide_scaled,
    fp8_gemm_k4096_bf16_out_bias,
    fp8_wgrad_2xacc_scaled,
)


class Fp8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, quant_scale):
        orig_shape = x.shape
        k = weight.shape[1]
        x_flat = x.reshape(-1, k).contiguous()
        weight_c = weight.contiguous()

        # TE-style delayed-scaling usage for trainable Linear: prepare rowwise data for
        # fprop and columnwise data for the bwd GEMMs while the BF16 tensors are live.
        qx, qx_t, _ = bf16_quantize_rowwise_transpose_delayed(x_flat, quant_scale)
        qw, qw_t, _ = bf16_quantize_rowwise_transpose_delayed(weight_c, quant_scale)

        if k == 1024:
            out = fp8_gemm_k1024_bf16_out_wide_scaled(qx, qw, 1.0, 1.0)
            out = out + bias
        elif k == 4096:
            out = fp8_gemm_k4096_bf16_out_bias(qx, qw, bias.contiguous())
        else:
            raise RuntimeError(f"unsupported K={k}")
        ctx.orig_shape = orig_shape
        ctx.save_for_backward(qx_t, qw_t, bias, quant_scale)
        return out.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        qx_t, qw_t, bias, quant_scale = ctx.saved_tensors
        grad = grad_out.reshape(-1, bias.shape[0]).contiguous()
        qdy, qdy_t, _, db = bf16_quantize_rowwise_transpose_db_delayed(grad, quant_scale)
        dequant = 1.0 / float(quant_scale[0].item())
        dx = fp8_dgrad_2xacc_scaled(qdy, qw_t, dequant, dequant)
        dw_kn = fp8_wgrad_2xacc_scaled(qx_t, qdy_t, dequant, dequant)
        return dx.reshape(ctx.orig_shape), dw_kn.transpose(0, 1).contiguous(), db.to(bias.dtype), None


def fp8_linear(x, weight, bias, quant_scale):
    return Fp8LinearFn.apply(x, weight, bias, quant_scale)


class Fp8Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, quant_scale):
        bsz, tokens, hidden = x.shape
        qkv = fp8_linear(x, self.qkv.weight, self.qkv.bias, quant_scale)
        qkv = qkv.reshape(bsz, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).reshape(bsz, tokens, hidden)
        return fp8_linear(y, self.proj.weight, self.proj.bias, quant_scale)


class Fp8Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * mlp_ratio, bias=True)
        self.fc2 = nn.Linear(hidden_size * mlp_ratio, hidden_size, bias=True)

    def forward(self, x, quant_scale):
        x = fp8_linear(x, self.fc1.weight, self.fc1.bias, quant_scale)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        return fp8_linear(x, self.fc2.weight, self.fc2.bias, quant_scale)


class Fp8LinearDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Fp8Attention(hidden_size, heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Fp8Mlp(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.register_buffer("quant_scale", torch.ones((1,), dtype=torch.float32), persistent=False)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        y = self.norm1(x)
        y = y * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.attn(y, self.quant_scale.float())
        y = self.norm2(x)
        y = y * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(y, self.quant_scale.float())
        return x


class Bf16DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, mlp_ratio: int):
        super().__init__()
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

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        y = self.norm1(x)
        y = y * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.attn(y)
        y = self.norm2(x)
        y = y * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(y)
        return x


def copy_weights(src: Bf16DiTBlock, dst: Fp8LinearDiTBlock) -> None:
    with torch.no_grad():
        dst.adaLN_modulation.load_state_dict(src.adaLN_modulation.state_dict())
        dst.attn.qkv.weight.copy_(src.attn.qkv.weight)
        dst.attn.qkv.bias.copy_(src.attn.qkv.bias)
        dst.attn.proj.weight.copy_(src.attn.proj.weight)
        dst.attn.proj.bias.copy_(src.attn.proj.bias)
        dst.mlp.fc1.weight.copy_(src.mlp.fc1.weight)
        dst.mlp.fc1.bias.copy_(src.mlp.fc1.bias)
        dst.mlp.fc2.weight.copy_(src.mlp.fc2.weight)
        dst.mlp.fc2.bias.copy_(src.mlp.fc2.bias)


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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    bf16 = Bf16DiTBlock(args.hidden, args.heads, args.mlp_ratio).to(device).to(torch.bfloat16).train()
    fp8 = Fp8LinearDiTBlock(args.hidden, args.heads, args.mlp_ratio).to(device).to(torch.bfloat16).train()
    copy_weights(bf16, fp8)

    x0 = torch.randn((args.batch, args.tokens, args.hidden), device=device, dtype=torch.bfloat16)
    c0 = torch.randn((args.batch, args.hidden), device=device, dtype=torch.bfloat16)
    grad = torch.randn_like(x0)
    x = x0.detach().clone().requires_grad_(True)
    c = c0.detach().clone().requires_grad_(True)
    x_fp8 = x0.detach().clone().requires_grad_(True)
    c_fp8 = c0.detach().clone().requires_grad_(True)

    with torch.no_grad():
        diff = (bf16(x, c).float() - fp8(x_fp8, c_fp8).float()).abs()
    print(f"shape: B={args.batch} T={args.tokens} H={args.hidden} heads={args.heads}")
    print(f"forward_fp8_vs_bf16 max={diff.max().item():.6g} mean={diff.mean().item():.6g}")

    step(bf16, x, c, grad)
    step(fp8, x_fp8, c_fp8, grad)
    torch.cuda.synchronize()
    bf16_stats = event_bench(lambda: step(bf16, x, c, grad), warmup=args.warmup, iters=args.iters)
    fp8_stats = event_bench(lambda: step(fp8, x_fp8, c_fp8, grad), warmup=args.warmup, iters=args.iters)
    print(
        f"bf16_linear_train: mean={bf16_stats['mean']:.6f} ms median={bf16_stats['median']:.6f} "
        f"min={bf16_stats['min']:.6f} max={bf16_stats['max']:.6f}"
    )
    print(
        f"tk_fp8_linear_train: mean={fp8_stats['mean']:.6f} ms median={fp8_stats['median']:.6f} "
        f"min={fp8_stats['min']:.6f} max={fp8_stats['max']:.6f} speedup={bf16_stats['mean']/fp8_stats['mean']:.3f}x"
    )


if __name__ == "__main__":
    main()
