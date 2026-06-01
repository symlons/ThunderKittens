from __future__ import annotations

import argparse
import statistics
from functools import partial
from typing import cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

from profile_dit_block_fp8_linear_e2e import Fp8LinearDiTBlock, Bf16DiTBlock, copy_weights


class TeAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = te.ops.Linear(hidden, 3 * hidden, bias=True, dtype=torch.bfloat16)
        self.proj = te.ops.Linear(hidden, hidden, bias=True, dtype=torch.bfloat16)

    def forward(self, x):
        b, t, h = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        y = y.transpose(1, 2).reshape(b, t, h)
        return self.proj(y)


class TeMlp(nn.Module):
    def __init__(self, hidden: int, mlp_ratio: int):
        super().__init__()
        self.fc1 = te.ops.Linear(hidden, hidden * mlp_ratio, bias=True, dtype=torch.bfloat16)
        self.act = te.ops.GELU()
        self.fc2 = te.ops.Linear(hidden * mlp_ratio, hidden, bias=True, dtype=torch.bfloat16)
        self.ops = te.ops.Sequential(self.fc1, self.act, self.fc2)

    def forward(self, x):
        return self.ops(x)


class TeDiTBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = TeAttention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = TeMlp(hidden, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        y = self.norm1(x)
        y = y * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.attn(y)
        y = self.norm2(x)
        y = y * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(y)
        return x


def copy_to_te(src: Bf16DiTBlock, dst: TeDiTBlock) -> None:
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
    times=[]
    for i in range(warmup+iters):
        torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        if i>=warmup: times.append(s.elapsed_time(e))
    return {"mean": statistics.mean(times), "median": statistics.median(times), "min": min(times), "max": max(times)}


def zero_grads(block, x, c):
    x.grad=None; c.grad=None
    for p in block.parameters(): p.grad=None


def step(block, x, c, grad, recipe=None):
    if recipe is None:
        out=block(x,c)
    else:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            out=block(x,c)
    out.backward(grad)
    zero_grads(block,x,c)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch',type=int,default=4); ap.add_argument('--tokens',type=int,default=1024)
    ap.add_argument('--hidden',type=int,default=1024); ap.add_argument('--heads',type=int,default=16)
    ap.add_argument('--mlp-ratio',type=int,default=4); ap.add_argument('--warmup',type=int,default=2); ap.add_argument('--iters',type=int,default=3)
    args=ap.parse_args()
    torch.manual_seed(1234); torch.backends.cuda.matmul.allow_tf32=True; torch.set_float32_matmul_precision('high')
    dev='cuda'
    bf16=Bf16DiTBlock(args.hidden,args.heads,args.mlp_ratio).to(dev).bfloat16().train()
    tk=Fp8LinearDiTBlock(args.hidden,args.heads,args.mlp_ratio).to(dev).bfloat16().train(); copy_weights(bf16,tk)
    teb=TeDiTBlock(args.hidden,args.heads,args.mlp_ratio).to(dev).bfloat16().train(); copy_to_te(bf16,teb)
    recipe=DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16)
    x0=torch.randn((args.batch,args.tokens,args.hidden),device=dev,dtype=torch.bfloat16); c0=torch.randn((args.batch,args.hidden),device=dev,dtype=torch.bfloat16); grad=torch.randn_like(x0)
    xb=x0.detach().clone().requires_grad_(True); cb=c0.detach().clone().requires_grad_(True)
    xt=x0.detach().clone().requires_grad_(True); ct=c0.detach().clone().requires_grad_(True)
    xe=x0.detach().clone().requires_grad_(True); ce=c0.detach().clone().requires_grad_(True)
    step(bf16,xb,cb,grad); step(tk,xt,ct,grad); step(teb,xe,ce,grad,recipe); torch.cuda.synchronize()
    bs=event_bench(lambda: step(bf16,xb,cb,grad),warmup=args.warmup,iters=args.iters)
    ts=event_bench(lambda: step(tk,xt,ct,grad),warmup=args.warmup,iters=args.iters)
    es=event_bench(lambda: step(teb,xe,ce,grad,recipe),warmup=args.warmup,iters=args.iters)
    print(f"shape: B={args.batch} T={args.tokens} H={args.hidden}")
    for name,st in [('bf16',bs),('tk_fp8',ts),('te_fp8_ops_delayed',es)]:
        print(f"{name}: mean={st['mean']:.6f} ms median={st['median']:.6f} min={st['min']:.6f} max={st['max']:.6f}")
    print(f"tk_vs_te_speedup={es['mean']/ts['mean']:.3f}x tk_vs_bf16={bs['mean']/ts['mean']:.3f}x te_vs_bf16={bs['mean']/es['mean']:.3f}x")

if __name__=='__main__': main()
