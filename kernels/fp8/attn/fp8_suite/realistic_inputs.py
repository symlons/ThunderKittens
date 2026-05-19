"""Realistic / heavy-tail input generators for the FP8 vs INT8 quant ablation.

Two sources:
  - Synthetic heavy-tail distributions where FP8 e4m3's dynamic range is
    *supposed* to beat INT8's uniform grid (Student-t low-df, log-normal,
    Laplace, channel-outlier, mixture-with-outliers).
  - Real-attention Q/K/V activations captured by hooking a pretrained
    Stable-Diffusion UNet (or any diffusers UNet) during a forward pass.
    Cached so the first run downloads ~330 MB and subsequent runs hit disk.

Tensors returned have shape (B, H, N, D) ready for
``fp8_suite.metrics.tensor_metrics`` and the existing quant references.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Synthetic distributions (no download)
# ---------------------------------------------------------------------------


def _laplace(shape, gen):
    # heavier tails than Gaussian, still bounded amax/sigma
    u = torch.rand(shape, device="cuda", generator=gen) - 0.5
    return -torch.sign(u) * torch.log1p(-2 * u.abs().clamp_max(0.999999))


def _student_t(df):
    def _f(shape, gen):
        # Student-t via N / sqrt(chi2/df)
        z = torch.randn(shape, device="cuda", generator=gen)
        # chi2 with df via sum of df squared standard normals (df must be int)
        chi2 = sum(torch.randn(shape, device="cuda", generator=gen) ** 2
                   for _ in range(df))
        return z / (chi2 / df).clamp_min(1e-12).sqrt()
    return _f


def _log_normal(shape, gen):
    # x = exp(N(0, 1)) − E[exp(N)] -> centered, very heavy right tail
    n = torch.randn(shape, device="cuda", generator=gen)
    x = torch.exp(n)
    return x - x.mean()


def _channel_outlier(scale):
    """One channel (last-dim index 0) has ``scale`` × larger std."""
    def _f(shape, gen):
        x = torch.randn(shape, device="cuda", generator=gen)
        x[..., 0] *= scale
        return x
    return _f


def _mix_with_outlier(scale, frac):
    """Mixture of N(0,1) (1-frac) and N(0,scale) (frac)."""
    def _f(shape, gen):
        x = torch.randn(shape, device="cuda", generator=gen)
        m = torch.rand(shape, device="cuda", generator=gen) < frac
        x = torch.where(m, x * scale, x)
        return x
    return _f


SYNTHETIC: Dict[str, Callable] = {
    "gaussian":           lambda s, g: torch.randn(s, device="cuda", generator=g),
    "laplace":            _laplace,
    "student_t_df3":      _student_t(3),
    "student_t_df5":      _student_t(5),
    "log_normal":         _log_normal,
    "channel_outlier_x8": _channel_outlier(8.0),
    "channel_outlier_x32": _channel_outlier(32.0),
    "mix_outlier_5pct_x10": _mix_with_outlier(10.0, 0.05),
    "mix_outlier_1pct_x50": _mix_with_outlier(50.0, 0.01),
}


def make_synthetic(kind, shape, *, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return SYNTHETIC[kind](shape, gen).to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Real Stable-Diffusion UNet Q/K/V activation capture
# ---------------------------------------------------------------------------


@dataclass
class ActivationSample:
    name: str          # e.g. "down_blocks.2.attentions.1.transformer_blocks.0.attn1"
    role: str          # "q" | "k" | "v"
    layer_kind: str    # "self_attn" | "cross_attn"
    tensor: torch.Tensor  # (B, H, N, D), float32, CUDA


def _hook_qkv(unet, samples: List[ActivationSample]):
    """Attach forward hooks on every Attention.to_q/to_k/to_v projection.

    Diffusers' ``Attention`` exposes ``.to_q``, ``.to_k``, ``.to_v`` as Linear
    layers; their *output* tensors are (B, N, inner_dim) where
    ``inner_dim = heads * dim_head``.
    """
    handles = []
    # heads / head_dim per Attention module so we can reshape outputs
    for name, mod in unet.named_modules():
        if mod.__class__.__name__ != "Attention":
            continue
        heads    = getattr(mod, "heads", None)
        head_dim = getattr(mod, "inner_dim", None)
        if heads is None or head_dim is None:
            continue
        head_dim = head_dim // heads
        layer_kind = "cross_attn" if getattr(mod, "is_cross_attention", False) else "self_attn"
        for role, lin in (("q", mod.to_q), ("k", mod.to_k), ("v", mod.to_v)):
            def _mk_hook(name=name, role=role, heads=heads, head_dim=head_dim,
                         layer_kind=layer_kind):
                def _h(_m, _i, out):
                    # out: (B, N, heads*head_dim)
                    b, n, _ = out.shape
                    t = out.detach().reshape(b, n, heads, head_dim).permute(0, 2, 1, 3)
                    samples.append(ActivationSample(name, role, layer_kind,
                                                    t.to(torch.float32).contiguous()))
                return _h
            handles.append(lin.register_forward_hook(_mk_hook()))
    return handles


def _load_diffusion_model(model_id: str, cache_dir):
    """Return (model, runner) where runner(model) does a forward pass.

    Dispatches on model id: Stable-Diffusion-like UNet2DConditionModel or
    CogVideoX-like CogVideoXTransformer3DModel.
    """
    from diffusers import UNet2DConditionModel

    if "cogvideo" in model_id.lower():
        from diffusers import CogVideoXTransformer3DModel
        m = CogVideoXTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,  # 2B fp32 weights = 8 GB → bf16 to fit
        ).to("cuda").eval()
        cfg = m.config
        # latent shape: (B, T, C, H, W). Use a *small* video to keep the
        # forward cheap; numerics on the captured activations are not
        # sensitive to spatial size.
        B, T = 1, 1
        C  = cfg.in_channels
        H  = cfg.sample_height // 2  # half-res for speed
        W  = cfg.sample_width  // 2
        text_seq = cfg.max_text_seq_length
        text_dim = cfg.text_embed_dim
        def _runner(mod):
            x  = torch.randn(B, T, C, H, W, device="cuda", dtype=torch.bfloat16)
            te = torch.randn(B, text_seq, text_dim, device="cuda", dtype=torch.bfloat16)
            ts = torch.tensor([500], device="cuda", dtype=torch.long)
            with torch.no_grad():
                mod(x, te, ts)
        return m, _runner

    # Default: SD-1.x UNet
    m = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir=cache_dir,
        torch_dtype=torch.float32,
    ).to("cuda").eval()
    def _runner(mod):
        x  = torch.randn(2, 4, 64, 64, device="cuda", dtype=torch.float32)
        ts = torch.tensor([10, 50], device="cuda", dtype=torch.long)
        te = torch.randn(2, 77, 768, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            mod(x, ts, encoder_hidden_states=te)
    return m, _runner


def capture_unet_qkv(model_id: str = "segmind/tiny-sd",
                     cache_dir: str = None,
                     max_layers: int = 6,
                     seed: int = 0) -> List[ActivationSample]:
    """Download (if needed) a diffusers UNet or CogVideoX transformer and
    capture Q/K/V from up to ``max_layers`` Attention modules during one
    forward pass with random latents.
    """
    cache_dir = cache_dir or os.environ.get("HF_HOME") or None
    torch.manual_seed(seed)
    model, runner = _load_diffusion_model(model_id, cache_dir)

    samples: List[ActivationSample] = []
    handles = _hook_qkv(model, samples)
    try:
        runner(model)
    finally:
        for h in handles:
            h.remove()

    # Free model memory before returning activations to caller.
    del model
    torch.cuda.empty_cache()

    # Subsample to keep things tractable / diverse: pick one of each role
    # from a few layers spread across the UNet depth.
    by_layer: Dict[str, List[ActivationSample]] = {}
    for s in samples:
        by_layer.setdefault(s.name, []).append(s)
    chosen_layers = list(by_layer.keys())
    step = max(1, len(chosen_layers) // max_layers)
    chosen_layers = chosen_layers[::step][:max_layers]
    out = []
    for ln in chosen_layers:
        for s in by_layer[ln]:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Distribution stats helper (for the report)
# ---------------------------------------------------------------------------


def per_row_stats(x: torch.Tensor) -> Dict[str, float]:
    """``(B, H, N, D)`` -> per-row summary used in the report."""
    flat = x.reshape(-1, x.shape[-1])  # (rows, D)
    row_std  = flat.std(dim=-1)
    row_amax = flat.abs().amax(dim=-1)
    # amax/std is the regime-defining ratio: small (~3-4) -> INT8 wins,
    # large (>10) -> FP8 wins.
    ratio = (row_amax / row_std.clamp_min(1e-12)).cpu()
    return {
        "rows":         flat.shape[0],
        "mean":         float(x.mean()),
        "std":          float(x.std()),
        "p99_abs":      float(torch.quantile(x.abs().flatten(), 0.99)),
        "amax":         float(x.abs().max()),
        "amax_over_std": float(x.abs().max() / x.std().clamp_min(1e-12)),
        "row_amax_over_std_mean": float(ratio.mean()),
        "row_amax_over_std_p95":  float(torch.quantile(ratio, 0.95)),
    }
