from __future__ import annotations

import argparse
import math
import time
from functools import partial
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

import _C
import _linear_bwd_fused
from tk_bench import input_group_count, profile_groups, print_bench, uniform_bf16


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FusedAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        batch, tokens, dim = x.shape
        flat = x.reshape(batch * tokens, dim).contiguous()
        out = torch.empty_like(flat)
        mean = torch.empty((flat.shape[0],), device=x.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        _C.layernorm_adaln(flat, shift.contiguous(), scale.contiguous(), out, mean, rstd, tokens, eps)
        ctx.save_for_backward(flat, scale.contiguous(), mean, rstd)
        ctx.tokens = tokens
        ctx.shape = x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, scale, mean, rstd = ctx.saved_tensors
        grad = grad_out.reshape_as(x).contiguous()
        dx = torch.empty_like(x)
        dshift = torch.empty_like(scale, dtype=torch.float32)
        dscale = torch.empty_like(scale, dtype=torch.float32)
        _C.layernorm_adaln_backward(grad, x, scale, mean, rstd, dx, dshift, dscale, ctx.tokens)
        return dx.reshape(ctx.shape), dshift.to(scale.dtype), dscale.to(scale.dtype), None


def fused_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return FusedAdaLN.apply(x, shift, scale, eps)


class FusedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        flat_x = x.reshape(batch * tokens, dim).contiguous()
        flat_h = h.reshape(batch * tokens, dim).contiguous()
        gate_c = gate.contiguous()
        out = torch.empty_like(flat_x)
        _C.gated_residual(flat_x, flat_h, gate_c, out, tokens)
        ctx.save_for_backward(flat_h, gate_c)
        ctx.tokens = tokens
        ctx.shape = x.shape
        return out.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h, gate = ctx.saved_tensors
        grad = grad_out.reshape_as(h).contiguous()
        dx = torch.empty_like(grad)
        dh = torch.empty_like(grad)
        dgate = torch.empty_like(gate, dtype=torch.float32)
        _C.gated_residual_backward(grad, h, gate, dx, dh, dgate, ctx.tokens)
        return dx.reshape(ctx.shape), dh.reshape(ctx.shape), dgate.to(gate.dtype)


def gated_residual(x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16 and h.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16:
        return FusedGatedResidual.apply(x, h, gate)
    return x + gate.unsqueeze(1).to(h.dtype) * h


class TkLinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        out = torch.empty((x_c.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        preact = torch.empty_like(out)
        _C.gemm_custom_native(x_c, w.contiguous(), out, b.contiguous(), preact)
        ctx.save_for_backward(x_c, w, preact)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w, preact = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dz = torch.empty_like(grad_out)
        db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
        _linear_bwd_fused.gelu_bwd_bias(grad_out, preact, dz, db)
        dw = torch.empty_like(w)
        dx = torch.empty_like(x)
        _linear_bwd_fused.dw_gemm(dz, x, dw)
        _linear_bwd_fused.dx_gemm_native(dz, w.contiguous(), dx)
        return dx, dw, db.to(grad_out.dtype)


class TkLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        ctx.save_for_backward(x_c, w)
        out = torch.empty((x_c.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        _C.gemm_linear_native(x_c, w.contiguous(), out, b.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dw = torch.empty_like(w)
        dx = torch.empty_like(x)
        _linear_bwd_fused.dw_gemm(grad_out, x, dw)
        _linear_bwd_fused.dx_gemm_native(grad_out, w.contiguous(), dx)
        db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
        _linear_bwd_fused.bias_reduce(grad_out, db)
        return dx, dw, db.to(grad_out.dtype)


class TkMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        h = TkLinearGelu.apply(flat, self.fc1.weight, self.fc1.bias)
        out = TkLinear.apply(h, self.fc2.weight, self.fc2.bias)
        return out.reshape(shape)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        b, c, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (d, h, w)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.fc1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(t_freq)))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, fused_adaln_enabled=False, fused_residual_enabled=False, tk_mlp_enabled=False, **block_kwargs):
        super().__init__()
        self.fused_adaln_enabled = fused_adaln_enabled
        self.fused_residual_enabled = fused_residual_enabled
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = (
            TkMlp(hidden_size, mlp_hidden_dim)
            if tk_mlp_enabled
            else Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
                drop=0,
            )
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if self.fused_adaln_enabled:
            attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
            attn_out = self.attn(attn_in)
            x = gated_residual(x, attn_out, gate_msa) if self.fused_residual_enabled else x + gate_msa.unsqueeze(1) * attn_out
            mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
            mlp_out = self.mlp(mlp_in)
            x = gated_residual(x, mlp_out, gate_mlp) if self.fused_residual_enabled else x + gate_mlp.unsqueeze(1) * mlp_out
            return x
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, fused_adaln_enabled=False):
        super().__init__()
        self.fused_adaln_enabled = fused_adaln_enabled
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        if self.fused_adaln_enabled:
            x = fused_adaln(x, shift, scale, self.norm_final.eps)
        else:
            x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_class_condition=False,
        fused_adaln_enabled=False,
        fused_residual_enabled=False,
        tk_mlp_enabled=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.use_class_condition = use_class_condition
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) if use_class_condition else None
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                mlp_ratio=mlp_ratio,
                fused_adaln_enabled=fused_adaln_enabled,
                fused_residual_enabled=fused_residual_enabled,
                tk_mlp_enabled=tk_mlp_enabled,
            )
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, fused_adaln_enabled=fused_adaln_enabled)
        self.register_buffer("_pos_embed", torch.empty(0), persistent=False)
        self._pos_shape = None
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.fc1.weight, std=0.02)
        nn.init.normal_(self.t_embedder.fc2.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def randomize_zero_init_layers(self):
        for block in self.blocks:
            nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
        nn.init.xavier_uniform_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.normal_(self.final_layer.adaLN_modulation[-1].bias, std=0.02)
        nn.init.xavier_uniform_(self.final_layer.linear.weight)
        nn.init.normal_(self.final_layer.linear.bias, std=0.02)

    def pos_embed(self, spatial_shape, dtype, device):
        if self._pos_shape != spatial_shape or self._pos_embed.numel() == 0:
            pos_embed = get_3d_sincos_pos_embed(self.x_embedder.proj.out_channels, spatial_shape)
            self._pos_embed = torch.from_numpy(pos_embed).to(device=device, dtype=dtype).unsqueeze(0)
            self._pos_shape = spatial_shape
        return self._pos_embed

    def unpatchify(self, x, spatial_shape):
        c = self.out_channels
        p = self.patch_size
        d, h, w = spatial_shape
        x = x.reshape(x.shape[0], d, h, w, p, p, p, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(x.shape[0], c, d * p, h * p, w * p)

    def forward(self, x, t, y=None):
        x, spatial_shape = self.x_embedder(x)
        x = x + self.pos_embed(spatial_shape, x.dtype, x.device)
        t = self.t_embedder(t)
        if self.use_class_condition:
            assert y is not None
            c = t + self.y_embedder(y, self.training)
        else:
            c = t
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x, spatial_shape)


def get_3d_sincos_pos_embed(embed_dim, grid_size_dhw):
    d, h, w = grid_size_dhw
    dim_each = (embed_dim // 6) * 2
    dims = [dim_each, dim_each, dim_each]
    for i in range((embed_dim - sum(dims)) // 2):
        dims[i % 3] += 2
    assert sum(dims) == embed_dim and all(dim % 2 == 0 for dim in dims)
    emb_d = get_1d_sincos_pos_embed_from_grid(dims[0], np.arange(d, dtype=np.float32))
    emb_h = get_1d_sincos_pos_embed_from_grid(dims[1], np.arange(h, dtype=np.float32))
    emb_w = get_1d_sincos_pos_embed_from_grid(dims[2], np.arange(w, dtype=np.float32))
    emb_d = np.broadcast_to(emb_d[:, None, None, :], (d, h, w, dims[0])).copy()
    emb_h = np.broadcast_to(emb_h[None, :, None, :], (d, h, w, dims[1])).copy()
    emb_w = np.broadcast_to(emb_w[None, None, :, :], (d, h, w, dims[2])).copy()
    return np.concatenate([emb_d, emb_h, emb_w], axis=-1).reshape(d * h * w, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def dit_config(name: str):
    configs = {
        "S": dict(depth=12, hidden_size=384, patch_size=1, num_heads=6, in_channels=4),
        "L": dict(depth=24, hidden_size=1024, patch_size=1, num_heads=16, in_channels=32),
        "XL": dict(depth=28, hidden_size=1152, patch_size=1, num_heads=16, in_channels=4),
    }
    return configs[name]


def make_model(name: str, fused: bool, fused_residual: bool = False, tk_mlp: bool = False) -> DiT:
    torch.manual_seed(123)
    model = DiT(
        **dit_config(name),
        fused_adaln_enabled=fused,
        fused_residual_enabled=fused_residual,
        tk_mlp_enabled=tk_mlp,
    ).cuda().to(torch.bfloat16).train()
    model.randomize_zero_init_layers()
    return model


def clone_state(dst: nn.Module, src: nn.Module) -> None:
    dst.load_state_dict(src.state_dict(), strict=False)


def make_group(batch: int, channels: int, spatial: tuple[int, int, int], seed: int):
    x = uniform_bf16((batch, channels, *spatial), seed, -1.0, 1.0).requires_grad_(True)
    t = torch.randint(0, 1000, (batch,), device="cuda", dtype=torch.long)
    grad = uniform_bf16((batch, channels, *spatial), seed + 1, -1.0, 1.0)
    return x, t, grad


def train_step(model: nn.Module, group):
    x, t, grad = group
    out = model(x, t)
    out.backward(grad)
    model.zero_grad(set_to_none=True)
    x.grad = None


def memory_probe(model: nn.Module, group, label: str):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        train_step(model, group)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(f"{label}: PASS peak_allocated={peak / 2**30:.2f} GiB peak_reserved={reserved / 2**30:.2f} GiB")
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        print(f"{label}: OOM peak_allocated={peak / 2**30:.2f} GiB peak_reserved={reserved / 2**30:.2f} GiB error={exc}")
        torch.cuda.empty_cache()


def train_step_grad(model: nn.Module, x: torch.Tensor, t: torch.Tensor, grad: torch.Tensor):
    params = tuple(p for p in model.parameters() if p.requires_grad)
    out = model(x, t)
    grads = torch.autograd.grad(out, (x, *params), grad, allow_unused=True)
    return grads


def bench_case(model_name: str, batch: int, spatial: tuple[int, int, int], include_compile: bool, warmup: int, iters: int):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    input_bytes = batch * cfg["in_channels"] * tokens * 2
    groups_n = min(input_group_count(input_bytes), 4)
    groups = [make_group(batch, cfg["in_channels"], spatial, 50000 + i * 10) for i in range(groups_n)]

    print(f"\n3D DiT-{model_name}/1 E2E train: batch={batch} tokens={tokens} spatial={spatial} groups={groups_n}")
    variants = [("eager", False, False, False, False)]
    if include_compile:
        variants.append(("compile", False, False, False, True))
    variants.extend([
        ("tk_mlp", False, False, True, False),
        ("fused_adaln", True, False, False, False),
        ("fused_adaln_residual", True, True, False, False),
        ("fused_adaln_residual_tk_mlp", True, True, True, False),
    ])
    results = []
    for variant_name, fused, fused_residual, tk_mlp, compiled in variants:
        print(f"  running {variant_name}...", flush=True)
        model = make_model(model_name, fused=fused, fused_residual=fused_residual, tk_mlp=tk_mlp)
        if compiled:
            model = torch.compile(model)
        try:
            result = profile_groups(
                f"DiT-{model_name} B{batch} {variant_name} train",
                groups,
                lambda g, current_model=model: train_step(current_model, g),
                warmup=max(1, min(2, warmup)) if compiled else warmup,
                iters=iters,
            )
            results.append(result)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"DiT-{model_name} B{batch} {variant_name} train: SKIP OOM ({exc})", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
    for result in results:
        print_bench(result)
    if results:
        base = results[0].us
        print("  speedup: " + ", ".join(f"{r.name} {base / r.us:.2f}x" for r in results[1:]))
    return results


def probe_case(model_name: str, batch: int, spatial: tuple[int, int, int], include_compile: bool = False):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(f"\n3D DiT-{model_name}/1 memory probe: batch={batch} tokens={tokens} spatial={spatial}")
    group = make_group(batch, cfg["in_channels"], spatial, 90000)
    for label, fused, fused_residual in (
        ("eager", False, False),
        ("fused_adaln", True, False),
        ("fused_adaln_residual", True, True),
    ):
        model = make_model(model_name, fused=fused, fused_residual=fused_residual)
        memory_probe(model, group, f"DiT-{model_name} B{batch} {label} train")
        del model
        torch.cuda.empty_cache()
    model = make_model(model_name, fused=False, fused_residual=False, tk_mlp=True)
    memory_probe(model, group, f"DiT-{model_name} B{batch} tk_mlp train")
    del model
    torch.cuda.empty_cache()
    model = make_model(model_name, fused=True, fused_residual=True, tk_mlp=True)
    memory_probe(model, group, f"DiT-{model_name} B{batch} fused_adaln_residual_tk_mlp train")
    del model
    torch.cuda.empty_cache()
    if include_compile:
        model = torch.compile(make_model(model_name, fused=False))
        memory_probe(model, group, f"DiT-{model_name} B{batch} compile train")
        del model
        torch.cuda.empty_cache()


def spatial_for_tokens(tokens: int) -> tuple[int, int, int]:
    shapes = {
        256: (4, 8, 8),
        512: (8, 8, 8),
        1024: (8, 8, 16),
        2048: (8, 16, 16),
        4096: (16, 16, 16),
        8192: (16, 16, 32),
        16384: (16, 32, 32),
        32768: (32, 32, 32),
        60000: (30, 40, 50),
        65536: (32, 32, 64),
    }
    if tokens in shapes:
        return shapes[tokens]
    best = (1, 1, tokens)
    best_score = (tokens, tokens)
    for d in range(1, int(round(tokens ** (1 / 3))) + 3):
        if tokens % d:
            continue
        plane = tokens // d
        for h in range(d, int(math.sqrt(plane)) + 2):
            if plane % h:
                continue
            w = plane // h
            dims = tuple(sorted((d, h, w)))
            score = (dims[-1] - dims[0], dims[-1])
            if score < best_score:
                best = dims
                best_score = score
    return best


def main():
    parser = argparse.ArgumentParser(description="Benchmark full 3D DiT training variants.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="S")
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--spatial", nargs=3, type=int, default=[2, 2, 4])
    parser.add_argument("--tokens", nargs="+", type=int, default=None, help="Token counts to benchmark. Arbitrary counts are mapped to an exact-product 3D shape.")
    parser.add_argument("--sweep", action="store_true", help="Run every token count in --tokens for every batch in --batches.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--probe-memory", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    a = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.mm(a, b).sum().backward()
    torch.cuda.synchronize()
    all_results = []
    cases = []
    if args.sweep or args.tokens is not None:
        token_counts = args.tokens or [512, 1024, 2048, 4096, 8192, 16384, 32768, 60000]
        for tokens in token_counts:
            spatial = spatial_for_tokens(tokens)
            for batch in args.batches:
                cases.append((batch, spatial, tokens))
    else:
        spatial = tuple(args.spatial)
        cases = [(batch, spatial, spatial[0] * spatial[1] * spatial[2]) for batch in args.batches]
    for batch, spatial, tokens in cases:
        try:
            if args.probe_memory:
                probe_case(args.model, batch, spatial, args.compile)
            else:
                all_results.extend(bench_case(args.model, batch, spatial, args.compile, args.warmup, args.iters))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"\nDiT-{args.model} B{batch} T{tokens}: SKIP OOM ({exc})")
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\nDiT-{args.model} B{batch} T{tokens}: SKIP OOM ({exc})")
            else:
                raise
        torch.cuda.empty_cache()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
