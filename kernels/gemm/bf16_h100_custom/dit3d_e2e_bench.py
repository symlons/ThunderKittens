from __future__ import annotations

import argparse
import math
import time
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp

import _C
from tk_bench import input_group_count, profile_groups, print_bench, uniform_bf16
from tk_dit_ops import (
    FusedAdaLNLinear,
    FusedAdaLNLinearGelu,
    FusedInputMlp,
    FusedLinearGatedResidual,
    TkMlp,
    fused_adaln,
    fused_adaln_linear,
    fused_linear_gated_residual,
    gated_residual,
    linear_then_gated_residual,
    modulate,
)
from dit_profile_utils import REGULAR_TIMM_BLOCK_KWARGS


class ProjectedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _qkv(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)

    def _fused_adaln_qkv(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return fused_adaln_linear(x, shift, scale, self.qkv, eps).reshape(batch, tokens, 3, self.num_heads, self.head_dim)

    def _attention_from_qkv(self, qkv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._attention_from_qkv(self._qkv(x)))

    def forward_from_adaln(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
        return self.proj(self._attention_from_qkv(self._fused_adaln_qkv(x, shift, scale, eps)))

    def forward_from_adaln_residual(
        self,
        residual: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        attn = self._attention_from_qkv(self._fused_adaln_qkv(residual, shift, scale, eps))
        return fused_linear_gated_residual(attn, residual, gate, self.proj)

    def forward_residual(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        attn = self._attention_from_qkv(self._qkv(x))
        return fused_linear_gated_residual(attn, residual, gate, self.proj)

    def forward_residual_epilogue(self, x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        attn = self._attention_from_qkv(self._qkv(x))
        return linear_then_gated_residual(attn, residual, gate, self.proj)


class SdpaAttention(ProjectedAttention):
    def _attention_from_qkv(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, tokens, _, _, _ = qkv.shape
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return out.transpose(1, 2).reshape(batch, tokens, self.num_heads * self.head_dim)


class FlashAttention3(ProjectedAttention):
    def _attention_from_qkv(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, tokens, _, _, _ = qkv.shape
        q, k, v = qkv.unbind(dim=2)
        out = flash_attn3_func()(q.contiguous(), k.contiguous(), v.contiguous(), causal=False)
        return out.reshape(batch, tokens, self.num_heads * self.head_dim)


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
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        fused_adaln_enabled=False,
        fused_residual_enabled=False,
        tk_mlp_enabled=False,
        fused_input_projection_enabled=False,
        fused_output_projection_enabled=False,
        fused_epilogue_only_enabled=False,
        attention_backend="timm",
        **block_kwargs,
    ):
        super().__init__()
        self.fused_adaln_enabled = fused_adaln_enabled
        self.fused_residual_enabled = fused_residual_enabled
        self.fused_input_projection_enabled = fused_input_projection_enabled
        self.fused_output_projection_enabled = fused_output_projection_enabled
        self.fused_epilogue_only_enabled = fused_epilogue_only_enabled
        self.tk_mlp_enabled = tk_mlp_enabled
        use_projection_fusion = fused_input_projection_enabled or fused_output_projection_enabled

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if attention_backend == "fa3":
            self.attn = FlashAttention3(hidden_size, num_heads=num_heads, qkv_bias=True)
        elif attention_backend == "timm" and use_projection_fusion:
            self.attn = SdpaAttention(hidden_size, num_heads=num_heads, qkv_bias=True)
        elif attention_backend == "timm":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            raise ValueError(f"unknown attention backend: {attention_backend}")

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if tk_mlp_enabled:
            self.mlp = TkMlp(hidden_size, mlp_hidden_dim)
        elif use_projection_fusion:
            self.mlp = FusedInputMlp(hidden_size, mlp_hidden_dim)
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
                drop=0,
            )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if self.fused_adaln_enabled:
            if (
                self.fused_input_projection_enabled
                and self.fused_output_projection_enabled
                and self.fused_residual_enabled
                and hasattr(self.attn, "forward_from_adaln_residual")
            ):
                x = self.attn.forward_from_adaln_residual(x, shift_msa, scale_msa, gate_msa, self.norm1.eps)
            elif self.fused_input_projection_enabled and hasattr(self.attn, "forward_from_adaln"):
                attn_out = self.attn.forward_from_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = gated_residual(x, attn_out, gate_msa) if self.fused_residual_enabled else x + gate_msa.unsqueeze(1) * attn_out
            elif (
                self.fused_epilogue_only_enabled
                and self.fused_residual_enabled
                and hasattr(self.attn, "forward_residual_epilogue")
            ):
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = self.attn.forward_residual_epilogue(attn_in, x, gate_msa)
            elif self.fused_output_projection_enabled and self.fused_residual_enabled and hasattr(self.attn, "forward_residual"):
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                x = self.attn.forward_residual(attn_in, x, gate_msa)
            else:
                attn_in = fused_adaln(x, shift_msa, scale_msa, self.norm1.eps)
                attn_out = self.attn(attn_in)
                x = gated_residual(x, attn_out, gate_msa) if self.fused_residual_enabled else x + gate_msa.unsqueeze(1) * attn_out
            if (
                self.fused_residual_enabled
                and self.tk_mlp_enabled
                and hasattr(self.mlp, "forward_from_adaln_residual")
            ):
                x = self.mlp.forward_from_adaln_residual(x, shift_mlp, scale_mlp, gate_mlp, self.norm2.eps)
            elif (
                self.fused_input_projection_enabled
                and self.fused_output_projection_enabled
                and self.fused_residual_enabled
                and hasattr(self.mlp, "forward_from_adaln_residual")
            ):
                x = self.mlp.forward_from_adaln_residual(x, shift_mlp, scale_mlp, gate_mlp, self.norm2.eps)
            elif self.fused_input_projection_enabled and hasattr(self.mlp, "forward_from_adaln"):
                mlp_out = self.mlp.forward_from_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = gated_residual(x, mlp_out, gate_mlp) if self.fused_residual_enabled else x + gate_mlp.unsqueeze(1) * mlp_out
            elif (
                self.fused_epilogue_only_enabled
                and self.fused_residual_enabled
                and hasattr(self.mlp, "forward_residual_epilogue")
            ):
                mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = self.mlp.forward_residual_epilogue(mlp_in, x, gate_mlp)
            elif self.fused_output_projection_enabled and self.fused_residual_enabled and hasattr(self.mlp, "forward_residual"):
                mlp_in = fused_adaln(x, shift_mlp, scale_mlp, self.norm2.eps)
                x = self.mlp.forward_residual(mlp_in, x, gate_mlp)
            else:
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
        fused_input_projection_enabled=False,
        fused_output_projection_enabled=False,
        fused_epilogue_only_enabled=False,
        attention_backend="timm",
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
                fused_input_projection_enabled=fused_input_projection_enabled,
                fused_output_projection_enabled=fused_output_projection_enabled,
                fused_epilogue_only_enabled=fused_epilogue_only_enabled,
                attention_backend=attention_backend,
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


def make_model(
    name: str,
    fused: bool,
    fused_residual: bool = False,
    tk_mlp: bool = False,
    fused_input_projection: bool = False,
    fused_output_projection: bool = False,
    fused_epilogue_only: bool = False,
    attention_backend: str = "timm",
) -> DiT:
    torch.manual_seed(123)
    model = DiT(
        **dit_config(name),
        fused_adaln_enabled=fused,
        fused_residual_enabled=fused_residual,
        tk_mlp_enabled=tk_mlp,
        fused_input_projection_enabled=fused_input_projection,
        fused_output_projection_enabled=fused_output_projection,
        fused_epilogue_only_enabled=fused_epilogue_only,
        attention_backend=attention_backend,
    ).cuda().to(torch.bfloat16).train()
    model.randomize_zero_init_layers()
    return model


from dit_variants import (
    TraceCompareData,
    TraceEvent,
    TraceLane,
    VariantConfig,
    load_compile_fusion_evidence,
    print_ditblock_fusion_plan,
    run_ditblock_fusion_ui,
    selected_variants,
    variant_config,
)

def make_variant_model(model_name: str, config: VariantConfig) -> nn.Module:
    model = make_model(
        model_name,
        fused=config.fused,
        fused_residual=config.fused_residual,
        tk_mlp=config.tk_mlp,
        fused_input_projection=config.fused_input_projection,
        fused_output_projection=config.fused_output_projection,
        fused_epilogue_only=config.fused_epilogue_only,
        attention_backend=config.attention_backend,
    )
    if config.compiled:
        model = torch.compile(model)
    return model


def make_variant_block(model_name: str, config: VariantConfig) -> nn.Module:
    cfg = dit_config(model_name)
    block = DiTBlock(
        cfg["hidden_size"],
        cfg["num_heads"],
        fused_adaln_enabled=config.fused,
        fused_residual_enabled=config.fused_residual,
        tk_mlp_enabled=config.tk_mlp,
        fused_input_projection_enabled=config.fused_input_projection,
        fused_output_projection_enabled=config.fused_output_projection,
        fused_epilogue_only_enabled=config.fused_epilogue_only,
        attention_backend=config.attention_backend,
        **({} if config.attention_backend != "timm" else {
            key: value for key, value in REGULAR_TIMM_BLOCK_KWARGS.items()
            if key not in {
                "fused_adaln_enabled",
                "fused_residual_enabled",
                "tk_mlp_enabled",
                "fused_input_projection_enabled",
                "fused_output_projection_enabled",
                "fused_epilogue_only_enabled",
                "attention_backend",
            }
        }),
    ).cuda().to(torch.bfloat16).train()
    nn.init.xavier_uniform_(block.adaLN_modulation[-1].weight)
    nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
    if config.compiled:
        block = torch.compile(block)
    return block


def make_block_group(batch: int, tokens: int, hidden_size: int, seed: int):
    x = uniform_bf16((batch, tokens, hidden_size), seed, -1.0, 1.0).requires_grad_(True)
    c = uniform_bf16((batch, hidden_size), seed + 1, -1.0, 1.0)
    grad = uniform_bf16((batch, tokens, hidden_size), seed + 2, -1.0, 1.0)
    return x, c, grad


def block_forward_step(block: nn.Module, group):
    x, c, _ = group
    with torch.no_grad():
        block(x, c)


def block_train_step(block: nn.Module, group):
    x, c, grad = group
    out = block(x, c)
    out.backward(grad)
    block.zero_grad(set_to_none=True)
    x.grad = None


def block_profile_step(block: nn.Module, group, include_backward: bool):
    if include_backward:
        block_train_step(block, group)
    else:
        block_forward_step(block, group)


def profile_variant_block_case(
    model_name: str,
    variant_name: str,
    batch: int,
    tokens: int,
    warmup: int,
    iters: int,
    rows: int,
    trace_out: Path | None = None,
    include_backward: bool = False,
) -> None:
    config = variant_config(variant_name)
    block = make_variant_block(model_name, config)
    hidden = dit_config(model_name)["hidden_size"]
    group = make_block_group(batch, tokens, hidden, 88000)
    for _ in range(warmup):
        block_profile_step(block, group, include_backward)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            block_profile_step(block, group, include_backward)
    torch.cuda.synchronize()
    mode = "forward+backward" if include_backward else "forward"
    print(f"\nTorch profiler DiTBlock-{model_name} {variant_name} {mode} B{batch} T{tokens} warmup={warmup} iters={iters}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows))
    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_out))
        print(f"wrote Chrome trace: {trace_out}")


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


def profile_variant_case(
    model_name: str,
    variant_name: str,
    batch: int,
    spatial: tuple[int, int, int],
    warmup: int,
    iters: int,
    rows: int,
    trace_out: Path | None = None,
) -> None:
    model = make_variant_model(model_name, variant_config(variant_name))
    model.pos_embed(spatial, torch.bfloat16, torch.device("cuda"))
    group = make_group(batch, dit_config(model_name)["in_channels"], spatial, 77000)
    for _ in range(warmup):
        train_step(model, group)
    torch.cuda.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(iters):
            train_step(model, group)
    torch.cuda.synchronize()
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(f"\nTorch profiler DiT-{model_name} {variant_name} B{batch} T{tokens} warmup={warmup} iters={iters}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows))
    if trace_out is not None:
        trace_out.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_out))
        print(f"wrote Chrome trace: {trace_out}")



def trace_file_name(model_name: str, variant_name: str, batch: int, spatial: tuple[int, int, int]) -> str:
    tokens = spatial[0] * spatial[1] * spatial[2]
    return f"dit_{model_name}_{variant_name}_b{batch}_t{tokens}.json"


def block_trace_file_name(model_name: str, variant_name: str, batch: int, tokens: int, include_backward: bool = False) -> str:
    mode = "fwd_bwd" if include_backward else "fwd"
    return f"ditblock_{mode}_{model_name}_{variant_name}_b{batch}_t{tokens}.json"


def profile_variant_block_trace(
    model_name: str,
    variant_name: str,
    batch: int,
    tokens: int,
    warmup: int,
    iters: int,
    rows: int,
    trace_dir: Path,
    include_backward: bool = False,
) -> Path:
    trace_path = trace_dir / block_trace_file_name(model_name, variant_name, batch, tokens, include_backward)
    profile_variant_block_case(model_name, variant_name, batch, tokens, warmup, iters, rows, trace_path, include_backward)
    return trace_path


def profile_variant_trace(
    model_name: str,
    variant_name: str,
    batch: int,
    spatial: tuple[int, int, int],
    warmup: int,
    iters: int,
    rows: int,
    trace_dir: Path,
) -> Path:
    trace_path = trace_dir / trace_file_name(model_name, variant_name, batch, spatial)
    profile_variant_case(model_name, variant_name, batch, spatial, warmup, iters, rows, trace_path)
    return trace_path


def compare_compile_trace(eager_trace: Path, compile_trace: Path, rows: int = 10) -> tuple[tuple[str, ...], TraceCompareData]:
    from collections import defaultdict

    from analyze_dit_compile_fusion import (
        build_external_category_map,
        build_external_linear_map,
        cuda_kernel_category,
        load_events,
        semantic_cuda_summary,
    )

    def categorized_kernel_events(events: list[dict]) -> list[TraceEvent]:
        external_linear_map = build_external_linear_map(events)
        out: list[TraceEvent] = []
        for event in events:
            if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
                continue
            name = event.get("name", "")
            start_us = float(event.get("ts", 0.0))
            dur_us = float(event.get("dur", 0.0))
            category, _desc = cuda_kernel_category(name, (event.get("args") or {}).get("External id"), external_linear_map)
            out.append(TraceEvent(start_us, dur_us, category, name))
        if not out:
            return []
        first_us = min(event.start_us for event in out)
        return sorted((TraceEvent(event.start_us - first_us, event.dur_us, event.category, event.name) for event in out), key=lambda event: event.start_us)

    def kernel_rows(events: list[dict], limit: int) -> list[str]:
        grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
        total_us = 0.0
        for event in categorized_kernel_events(events):
            total_us += event.dur_us
            grouped[(event.category, event.name)]["count"] += 1
            grouped[(event.category, event.name)]["dur_us"] += event.dur_us
        out: list[str] = []
        for (category, name), stat in sorted(grouped.items(), key=lambda item: item[1]["dur_us"], reverse=True)[:limit]:
            pct = 100.0 * stat["dur_us"] / total_us if total_us else 0.0
            out.append(f"{category:24s} {int(stat['count']):5d}x {stat['dur_us'] / 1000.0:8.3f} ms {pct:5.1f}%  {name}")
        return out

    def timeline_char(category: str) -> str:
        if category.startswith("fused_"):
            return "F"
        if "attention" in category:
            return "A"
        if "linear" in category or "gemm" in category:
            return "G"
        if "layer_norm" in category or category == "layer_norm":
            return "L"
        if "adaln" in category:
            return "N"
        if "residual" in category:
            return "R"
        if "gelu" in category:
            return "U"
        if "conv" in category or "patch_embed" in category:
            return "P"
        if "memset" in category:
            return "0"
        return "."

    def timeline_lane(label: str, kernels: list[TraceEvent], scale_us: float, width: int = 52) -> str:
        lane = [" "] * width
        if not kernels:
            return f"{label:<7}|{''.join(lane)}|"
        span = max(1.0, scale_us)
        for event in kernels:
            left = max(0, min(width - 1, int(event.start_us / span * width)))
            right = max(left + 1, min(width, int(event.end_us / span * width) + 1))
            lane[(left + right - 1) // 2] = timeline_char(event.category)
        own_span_ms = max(event.end_us for event in kernels) / 1000.0
        return f"{label:<7}|{''.join(lane)}| {own_span_ms:.3f} ms"

    def timeline_rows(eager_events: list[dict], compile_events: list[dict]) -> list[str]:
        eager_kernels = categorized_kernel_events(eager_events)
        compile_kernels = categorized_kernel_events(compile_events)
        if not eager_kernels and not compile_kernels:
            return ["timeline: no CUDA kernels found"]

        def span_us(kernels: list[TraceEvent]) -> float:
            if not kernels:
                return 1.0
            return max(1.0, max(event.end_us for event in kernels))

        scale_us = max(span_us(eager_kernels), span_us(compile_kernels))
        return [
            "timeline, each trace normalized to its first CUDA kernel; shared duration scale",
            f"0.000 ms{' ' * 34}{scale_us / 1000.0:.3f} ms",
            timeline_lane("eager", eager_kernels, scale_us),
            timeline_lane("compile", compile_kernels, scale_us),
            "legend: F=Inductor Triton fused  G=GEMM/linear  A=attention  L=layernorm  N=AdaLN  R=residual  U=GELU  P=conv  0=memset  .=other",
        ]

    eager_events = load_events(eager_trace)
    compile_events = load_events(compile_trace)
    eager_grouped, eager_total_us = semantic_cuda_summary(eager_events, build_external_category_map(eager_events))
    compile_grouped, compile_total_us = semantic_cuda_summary(compile_events, build_external_category_map(compile_events))
    keys = sorted(
        set(eager_grouped) | set(compile_grouped),
        key=lambda k: abs(compile_grouped.get(k, {}).get("dur_us", 0.0) - eager_grouped.get(k, {}).get("dur_us", 0.0)),
        reverse=True,
    )
    lines = [
        "mode: eager vs compile trace browser",
        f"eager trace:   {eager_trace}",
        f"compile trace: {compile_trace}",
        f"total CUDA: eager={eager_total_us / 1000.0:.3f} ms compile={compile_total_us / 1000.0:.3f} ms delta={(compile_total_us - eager_total_us) / 1000.0:+.3f} ms",
        "",
    ]
    lines.extend(timeline_rows(eager_events, compile_events))
    lines.extend(("", "category deltas by CUDA kernel time:"))
    for key in keys:
        eager_us = eager_grouped.get(key, {}).get("dur_us", 0.0)
        compile_us = compile_grouped.get(key, {}).get("dur_us", 0.0)
        lines.append(f"{key:28s} eager={eager_us / 1000.0:.3f} ms  compile={compile_us / 1000.0:.3f} ms  delta={(compile_us - eager_us) / 1000.0:+.3f} ms")
    lines.extend(("", "top compile CUDA kernels:", "category / count / cuda time / pct / kernel"))
    lines.extend(kernel_rows(compile_events, rows))
    lines.extend(("", "top eager CUDA kernels:", "category / count / cuda time / pct / kernel"))
    lines.extend(kernel_rows(eager_events, rows))
    eager_lane = TraceLane("eager", tuple(categorized_kernel_events(eager_events)), eager_total_us)
    compile_lane = TraceLane("compile", tuple(categorized_kernel_events(compile_events)), compile_total_us)
    data = TraceCompareData(eager_trace, compile_trace, eager_lane, compile_lane, tuple(lines))
    return tuple(lines), data


def compare_compile_trace_lines(eager_trace: Path, compile_trace: Path, rows: int = 10) -> tuple[str, ...]:
    return compare_compile_trace(eager_trace, compile_trace, rows)[0]


def check_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float = 1.2e-1, rtol: float = 1.2e-1) -> bool:
    if actual is None or expected is None:
        ok = actual is None and expected is None
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        return ok
    a = actual.detach().float()
    e = expected.detach().float()
    max_abs = (a - e).abs().max().item()
    denom = e.abs().clamp_min(1e-6)
    max_rel = ((a - e).abs() / denom).max().item()
    ok = torch.allclose(a, e, atol=atol, rtol=rtol)
    print(f"  {name}: {'PASS' if ok else 'FAIL'} max_abs={max_abs:.4e} max_rel={max_rel:.4e}")
    return ok


def fused_input_projection_correctness(batch: int = 2, tokens: int = 64, dim: int = 1024, hidden_dim: int = 4096) -> bool:
    print(f"\nFused LN+AdaLN+projection correctness B{batch} T{tokens} D{dim}")
    torch.manual_seed(1234)
    x = uniform_bf16((batch, tokens, dim), 9000, -2.0, 2.0).requires_grad_(True)
    shift = uniform_bf16((batch, dim), 9001, -0.5, 0.5).requires_grad_(True)
    scale = uniform_bf16((batch, dim), 9002, -0.25, 0.25).requires_grad_(True)
    w_qkv = uniform_bf16((3 * dim, dim), 9003, -0.02, 0.02).requires_grad_(True)
    b_qkv = uniform_bf16((3 * dim,), 9004, -0.02, 0.02).requires_grad_(True)
    w_fc1 = uniform_bf16((hidden_dim, dim), 9005, -0.02, 0.02).requires_grad_(True)
    b_fc1 = uniform_bf16((hidden_dim,), 9006, -0.02, 0.02).requires_grad_(True)
    grad_qkv = uniform_bf16((batch, tokens, 3 * dim), 9007, -1.0, 1.0)
    grad_fc1 = uniform_bf16((batch, tokens, hidden_dim), 9008, -1.0, 1.0)

    def clone(t: torch.Tensor) -> torch.Tensor:
        out = t.detach().clone()
        out.requires_grad_(t.requires_grad)
        return out

    ref = [clone(t) for t in (x, shift, scale, w_qkv, b_qkv)]
    z = modulate(torch.nn.functional.layer_norm(ref[0].float(), (dim,), None, None, 1e-6), ref[1], ref[2]).to(torch.bfloat16)
    ref_out = torch.nn.functional.linear(z, ref[3], ref[4])
    ref_out.backward(grad_qkv)

    fused = [clone(t) for t in (x, shift, scale, w_qkv, b_qkv)]
    fused_out = FusedAdaLNLinear.apply(fused[0], fused[1], fused[2], fused[3], fused[4], 1e-6)
    fused_out.backward(grad_qkv)

    ok = check_close("linear output", fused_out, ref_out)
    for name, actual, expected in zip(("x", "shift", "scale", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"linear d{name}", actual, expected) and ok

    ref = [clone(t) for t in (x, shift, scale, w_fc1, b_fc1)]
    z = modulate(torch.nn.functional.layer_norm(ref[0].float(), (dim,), None, None, 1e-6), ref[1], ref[2]).to(torch.bfloat16)
    ref_out = torch.nn.functional.gelu(torch.nn.functional.linear(z, ref[3], ref[4]), approximate="tanh")
    ref_out.backward(grad_fc1)

    fused = [clone(t) for t in (x, shift, scale, w_fc1, b_fc1)]
    fused_out = FusedAdaLNLinearGelu.apply(fused[0], fused[1], fused[2], fused[3], fused[4], 1e-6)
    fused_out.backward(grad_fc1)

    ok = check_close("gelu output", fused_out, ref_out) and ok
    for name, actual, expected in zip(("x", "shift", "scale", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"gelu d{name}", actual, expected) and ok

    h_fc2 = uniform_bf16((batch, tokens, hidden_dim), 9010, -1.0, 1.0).requires_grad_(True)
    residual = uniform_bf16((batch, tokens, dim), 9011, -1.0, 1.0).requires_grad_(True)
    gate = uniform_bf16((batch, dim), 9012, -0.25, 0.25).requires_grad_(True)
    w_fc2 = uniform_bf16((dim, hidden_dim), 9013, -0.02, 0.02).requires_grad_(True)
    b_fc2 = uniform_bf16((dim,), 9014, -0.02, 0.02).requires_grad_(True)
    grad_res = uniform_bf16((batch, tokens, dim), 9015, -1.0, 1.0)

    ref = [clone(t) for t in (h_fc2, residual, gate, w_fc2, b_fc2)]
    ref_out = ref[1] + ref[2].unsqueeze(1) * torch.nn.functional.linear(ref[0], ref[3], ref[4])
    ref_out.backward(grad_res)

    fused = [clone(t) for t in (h_fc2, residual, gate, w_fc2, b_fc2)]
    fused_out = FusedLinearGatedResidual.apply(fused[0], fused[1], fused[2], fused[3], fused[4])
    fused_out.backward(grad_res)

    ok = check_close("linear+residual output", fused_out, ref_out) and ok
    for name, actual, expected in zip(("x", "residual", "gate", "w", "b"), [t.grad for t in fused], [t.grad for t in ref]):
        ok = check_close(f"linear+residual d{name}", actual, expected) and ok

    print(f"Fused LN+AdaLN/projection epilogue correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def residual_group(batch: int, tokens: int, dim: int, seed: int):
    x = uniform_bf16((batch, tokens, dim), seed, -1.0, 1.0)
    h = uniform_bf16((batch, tokens, dim), seed + 1, -1.0, 1.0)
    gate = uniform_bf16((batch, dim), seed + 2, -0.5, 0.5)
    grad = uniform_bf16((batch, tokens, dim), seed + 3, -1.0, 1.0)
    return (
        x,
        h,
        gate,
        grad,
        torch.empty_like(x),
        torch.empty_like(h),
        torch.empty((batch, dim), device="cuda", dtype=torch.float32),
    )


def residual_torch_step(group):
    x, h, gate, grad, out, dh, dgate = group
    out.copy_(x + gate.unsqueeze(1) * h)
    dh.copy_(grad * gate.unsqueeze(1))
    dgate.copy_((grad.float() * h.float()).sum(dim=1))


def residual_fused_step(group):
    x, h, gate, grad, out, dh, dgate = group
    batch, tokens, dim = x.shape
    _C.gated_residual(
        x.reshape(batch * tokens, dim),
        h.reshape(batch * tokens, dim),
        gate,
        out.reshape(batch * tokens, dim),
        tokens,
    )
    _C.gated_residual_backward_no_dx(
        grad.reshape(batch * tokens, dim),
        h.reshape(batch * tokens, dim),
        gate,
        dh.reshape(batch * tokens, dim),
        dgate,
        tokens,
    )


def residual_correctness(batch: int = 2, tokens: int = 64, dim: int = 1024) -> bool:
    print(f"\nStandalone gated residual correctness B{batch} T{tokens} D{dim}")
    ref = residual_group(batch, tokens, dim, 9100)
    fused = tuple(t.clone() if i < 4 else torch.empty_like(t) for i, t in enumerate(ref))
    residual_torch_step(ref)
    residual_fused_step(fused)
    ok = check_close("residual output", fused[4], ref[4])
    ok = check_close("residual dh", fused[5], ref[5]) and ok
    ok = check_close("residual dgate", fused[6], ref[6]) and ok
    print(f"Standalone gated residual correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def bench_residual(tokens_list: list[int], batches: list[int], dim: int, warmup: int, iters: int):
    if not residual_correctness(tokens=max(64, min(tokens_list))):
        raise SystemExit(1)
    for tokens in tokens_list:
        for batch in batches:
            input_bytes = batch * tokens * dim * 2 * 4 + batch * dim * 2
            groups_n = min(input_group_count(input_bytes), 8)
            torch_groups = [residual_group(batch, tokens, dim, 9200 + i * 10) for i in range(groups_n)]
            fused_groups = [residual_group(batch, tokens, dim, 9300 + i * 10) for i in range(groups_n)]
            # x,h,gate,grad reads + out,dh,dgate writes. dx is an alias of grad and does not need materialization.
            bytes_moved = batch * tokens * dim * 2 * 6 + batch * dim * (2 + 4)
            torch_result = profile_groups(
                f"residual torch B{batch} T{tokens} D{dim}",
                torch_groups,
                residual_torch_step,
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_moved,
            )
            fused_result = profile_groups(
                f"residual fused B{batch} T{tokens} D{dim}",
                fused_groups,
                residual_fused_step,
                warmup=warmup,
                iters=iters,
                bytes_moved=bytes_moved,
            )
            print_bench(torch_result)
            print_bench(fused_result)
            print(f"  speedup: {torch_result.us / fused_result.us:.2f}x")


def bench_case(
    model_name: str,
    batch: int,
    spatial: tuple[int, int, int],
    include_compile: bool,
    warmup: int,
    iters: int,
    include_fa3: bool,
    only_variants: set[str] | None = None,
):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    input_bytes = batch * cfg["in_channels"] * tokens * 2
    groups_n = min(input_group_count(input_bytes), 4)
    groups = [make_group(batch, cfg["in_channels"], spatial, 50000 + i * 10) for i in range(groups_n)]

    print(f"\n3D DiT-{model_name}/1 E2E train: batch={batch} tokens={tokens} spatial={spatial} groups={groups_n}")
    variants = selected_variants(
        probe=False,
        include_compile=include_compile,
        include_fa3=include_fa3,
        only_variants=only_variants,
    )
    results = []
    for variant_name in variants:
        print(f"  running {variant_name}...", flush=True)
        config = variant_config(variant_name)
        model = make_variant_model(model_name, config)
        model.pos_embed(spatial, torch.bfloat16, torch.device("cuda"))
        try:
            result = profile_groups(
                f"DiT-{model_name} B{batch} {variant_name} train",
                groups,
                lambda g, current_model=model: train_step(current_model, g),
                warmup=max(1, min(2, warmup)) if config.compiled else warmup,
                iters=iters,
            )
            results.append(result)
            print_bench(result)
        except torch.cuda.OutOfMemoryError as exc:
            print(f"DiT-{model_name} B{batch} {variant_name} train: SKIP OOM ({exc})", flush=True)
        finally:
            del model
            torch.cuda.empty_cache()
    if results:
        base = results[0].us
        print("  speedup: " + ", ".join(f"{r.name} {base / r.us:.2f}x" for r in results[1:]))
    return results


def probe_case(model_name: str, batch: int, spatial: tuple[int, int, int], include_compile: bool = False, include_fa3: bool = False):
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(f"\n3D DiT-{model_name}/1 memory probe: batch={batch} tokens={tokens} spatial={spatial}")
    group = make_group(batch, cfg["in_channels"], spatial, 90000)

    variants = selected_variants(
        probe=True,
        include_compile=include_compile,
        include_fa3=include_fa3,
    )

    for variant_name in variants:
        model = make_variant_model(model_name, variant_config(variant_name))
        memory_probe(model, group, f"DiT-{model_name} B{batch} {variant_name} train")
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
    parser.add_argument("--model", choices=["S", "L", "XL"], default="L")
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 16, 64, 256, 1024])
    parser.add_argument("--spatial", nargs=3, type=int, default=[2, 2, 4])
    parser.add_argument("--tokens", nargs="+", type=int, default=None, help="Token counts to benchmark. Arbitrary counts are mapped to an exact-product 3D shape.")
    parser.add_argument("--sweep", action="store_true", help="Run every token count in --tokens for every batch in --batches.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fa3", action="store_true", help="Include FlashAttention-3 attention variants.")
    parser.add_argument("--variants", nargs="+", default=None, help="Only run the named benchmark variants.")
    parser.add_argument("--profile-variant", default="", help="Run torch profiler for one named variant and exit.")
    parser.add_argument("--profile-rows", type=int, default=30)
    parser.add_argument("--profile-trace-out", type=Path, default=None, help="Write a Chrome trace JSON for --profile-variant.")
    parser.add_argument("--check-fused-input", action="store_true", help="Run isolated LN+AdaLN+projection correctness checks and exit.")
    parser.add_argument("--bench-residual", action="store_true", help="Run isolated standalone gated residual forward+backward benchmarks and exit.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension for isolated residual benchmark.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--probe-memory", action="store_true")
    parser.add_argument("--print-ditblock", action="store_true", help="Print colored DiTBlock fusion plan for selected variants and exit.")
    parser.add_argument("--ditblock-ui", action="store_true", help="Open an interactive DiTBlock variant browser. Use with --print-ditblock.")
    parser.add_argument("--ditblock-detail", action="store_true", help="Include kernel-boundary explanations in --print-ditblock output.")
    parser.add_argument("--ditblock-bwd", action="store_true", help="Profile DiTBlock forward+backward traces in the UI. Default is forward-only.")
    parser.add_argument("--ditblock-compile-trace", type=Path, default=None, help="Chrome trace JSON from a compiled profile run; used to show actual TorchInductor Triton kernels in the DiTBlock UI/printout.")
    parser.add_argument("--ditblock-compile-rows", type=int, default=8, help="Number of trace-derived TorchInductor kernels to show in the DiTBlock UI/printout.")
    parser.add_argument("--ditblock-trace-dir", type=Path, default=Path("profile_artifacts/ditblock_ui_traces"), help="Directory where DiTBlock UI trace reruns are written.")
    args = parser.parse_args()

    if args.print_ditblock:
        variant_names = selected_variants(
            probe=False,
            include_compile=args.compile,
            include_fa3=args.fa3,
            only_variants=set(args.variants) if args.variants else None,
        )
        compile_evidence = None
        if args.ditblock_compile_trace is not None:
            compile_evidence = load_compile_fusion_evidence(args.ditblock_compile_trace, rows=args.ditblock_compile_rows)
        spatial = spatial_for_tokens(args.tokens[0]) if args.tokens else tuple(args.spatial)
        batch = args.batches[0]
        tokens = spatial[0] * spatial[1] * spatial[2]
        trace_config = (
            "trace scope: isolated DiTBlock",
            f"trace mode: {'forward+backward' if args.ditblock_bwd else 'forward-only'} ({'bwd on' if args.ditblock_bwd else 'bwd off'})",
            f"trace shape: model={args.model} batch={batch} tokens={tokens}",
            f"trace profiling: warmup={args.warmup} iters={args.iters} rows={args.profile_rows}",
            f"trace dir: {args.ditblock_trace_dir}",
        )

        def ui_trace_runner(variant_name: str):
            trace_path = profile_variant_block_trace(args.model, variant_name, batch, tokens, args.warmup, args.iters, args.profile_rows, args.ditblock_trace_dir, include_backward=args.ditblock_bwd)
            evidence = load_compile_fusion_evidence(trace_path, rows=args.ditblock_compile_rows) if variant_config(variant_name).compiled else None
            return evidence, (f"wrote trace: {trace_path}",)

        def ui_compare_runner(variant_name: str):
            compile_variant = variant_name if variant_config(variant_name).compiled else "compile"
            eager_trace = profile_variant_block_trace(args.model, "eager", batch, tokens, args.warmup, args.iters, args.profile_rows, args.ditblock_trace_dir, include_backward=args.ditblock_bwd)
            compile_trace = profile_variant_block_trace(args.model, compile_variant, batch, tokens, args.warmup, args.iters, args.profile_rows, args.ditblock_trace_dir, include_backward=args.ditblock_bwd)
            evidence = load_compile_fusion_evidence(compile_trace, rows=args.ditblock_compile_rows)
            lines, data = compare_compile_trace(eager_trace, compile_trace, rows=args.ditblock_compile_rows)
            return evidence, lines, data

        existing_eager_trace = args.ditblock_trace_dir / block_trace_file_name(args.model, "eager", batch, tokens, args.ditblock_bwd)
        existing_compile_trace = args.ditblock_trace_dir / block_trace_file_name(args.model, "compile", batch, tokens, args.ditblock_bwd)
        initial_compare_lines: tuple[str, ...] = ()
        initial_compare_data = None
        initial_show_compare = False
        if existing_eager_trace.exists() and existing_compile_trace.exists():
            initial_compare_lines, initial_compare_data = compare_compile_trace(existing_eager_trace, existing_compile_trace, rows=args.ditblock_compile_rows)
            initial_show_compare = True
            if compile_evidence is None:
                compile_evidence = load_compile_fusion_evidence(existing_compile_trace, rows=args.ditblock_compile_rows)

        if args.ditblock_ui:
            run_ditblock_fusion_ui(
                variant_names,
                compile_evidence=compile_evidence,
                trace_runner=ui_trace_runner,
                compare_runner=ui_compare_runner,
                trace_config=trace_config,
                initial_compare_lines=initial_compare_lines,
                initial_compare_data=initial_compare_data,
                initial_show_compare=initial_show_compare,
            )
        else:
            print_ditblock_fusion_plan(variant_names, detail=args.ditblock_detail, compile_evidence=compile_evidence)
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    a = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.mm(a, b).sum().backward()
    torch.cuda.synchronize()
    if args.check_fused_input:
        if not fused_input_projection_correctness():
            raise SystemExit(1)
        return
    if args.bench_residual:
        bench_residual(args.tokens or [1024], args.batches, args.hidden_dim, args.warmup, args.iters)
        return
    if args.profile_variant:
        spatial = spatial_for_tokens(args.tokens[0]) if args.tokens else tuple(args.spatial)
        profile_variant_case(
            args.model,
            args.profile_variant,
            args.batches[0],
            spatial,
            args.warmup,
            args.iters,
            args.profile_rows,
            args.profile_trace_out,
        )
        return
    all_results = []
    cases = []
    if args.sweep or args.tokens is not None:
        token_counts = args.tokens or [2**p for p in range(9, 17)]
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
                probe_case(args.model, batch, spatial, args.compile, args.fa3)
            else:
                all_results.extend(bench_case(
                    args.model,
                    batch,
                    spatial,
                    args.compile,
                    args.warmup,
                    args.iters,
                    args.fa3,
                    set(args.variants) if args.variants else None,
                ))
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
