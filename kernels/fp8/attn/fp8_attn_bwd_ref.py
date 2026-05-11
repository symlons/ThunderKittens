"""Reference backward for the FP8 forward kernel.

This is the *gold model* a future CUDA backward kernel will be checked
against. Like the forward kernel, it implements the SVQ / SageAttention2
mixed low-bit recipe:

  forward (already implemented in fp8_attn_fwd.cu)
    Q,K        : FP8 e4m3   per-row scale   (RTNE)
    K mean     : per-channel float, subtracted before quant of K
    V          : bf16 (per-channel mean subtracted; mean re-added at end)
    PV         : bf16 mma   (TODO: full FP8)
    softmax    : fp32, online; saves L = log-sum-exp per row

  backward (this file, fp32 reference)
    Reuses the SAME quantized tensors the forward kernel saw:
        Q_eff = Q_q * sq                      (per-row)
        K_eff = K_q * sk + K_mean             (per-row)
        V_eff = V_centered + V_mean           (per-channel)
    Recomputes:
        S    = (Q_eff @ K_eff^T) / sqrt(D)
        P    = exp(S - L)                     (L from forward kernel)
    Quantizes (matching the recipe):
        P    : FP8 e4m3, static 1/448 scale
        V    : FP8 e4m3, per-channel
        dO   : FP8 e4m3, per-row              (sparse spikes ⇒ per-row)
        dS   : FP8 e4m3, per-row
    Uses SR (stochastic rounding) for the dQ/dK upcast paths if requested
    (recipe says SR for backward dQ/dK, RTNE for forward).
    Computes:
        dV = P^T @ dO
        dP = dO @ V^T
        dS = P * (dP - rowsum(P * dP)) / sqrt(D)
        dQ = dS @ K_eff
        dK = dS^T @ Q_eff
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0


# ---------------------------------------------------------------------------
# FP8 quantisation primitives (host side, mirrors smooth_core.py)
# ---------------------------------------------------------------------------


def _fp8_dtype(mode: str):
    return torch.float8_e4m3fn if mode == "fp8_e4m3" else torch.float8_e5m2


def _fp8_max(mode: str) -> float:
    return FP8_E4M3_MAX if mode == "fp8_e4m3" else FP8_E5M2_MAX


def quantize_per_row_fp8(x: torch.Tensor, mode: str = "fp8_e4m3", *,
                         stochastic: bool = False):
    """Per-row FP8 fake-quant with optional stochastic rounding.

    Returns a fp32 dequantized tensor (the same the kernel would see) and
    the per-row scales (... , N).
    """
    fp_max = _fp8_max(mode)
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / fp_max
    y = x / scale
    if stochastic:
        y = _stochastic_round_to_fp8(y, mode)
    else:
        y = y.clamp(-fp_max, fp_max).to(_fp8_dtype(mode)).to(torch.float32)
    return y * scale, scale.squeeze(-1)


def quantize_per_tensor_fp8(x: torch.Tensor, mode: str = "fp8_e4m3", *,
                            stochastic: bool = False):
    fp_max = _fp8_max(mode)
    amax = x.abs().amax().clamp_min(1e-12)
    scale = amax / fp_max
    y = x / scale
    if stochastic:
        y = _stochastic_round_to_fp8(y, mode)
    else:
        y = y.clamp(-fp_max, fp_max).to(_fp8_dtype(mode)).to(torch.float32)
    return y * scale


def quantize_p_static_fp8(p: torch.Tensor, mode: str = "fp8_e4m3"):
    """P ∈ [0,1] post-softmax → FP8 with static scale 1/fp_max.

    Equivalent to: round(p * fp_max) / fp_max  via the FP8 cast.
    """
    fp_max = _fp8_max(mode)
    q = (p * fp_max).clamp(0, fp_max).to(_fp8_dtype(mode)).to(torch.float32)
    return q / fp_max


def quantize_v_per_channel_fp8(v: torch.Tensor, mode: str = "fp8_e4m3"):
    """V[..., N, D] → FP8 per-channel (one scale per D-column)."""
    fp_max = _fp8_max(mode)
    amax = v.abs().amax(dim=-2, keepdim=True).clamp_min(1e-12)
    scale = amax / fp_max
    q = (v / scale).clamp(-fp_max, fp_max).to(_fp8_dtype(mode)).to(torch.float32)
    return q * scale


def _stochastic_round_to_fp8(x: torch.Tensor, mode: str):
    """Approximate stochastic rounding into FP8.

    True SR for FP8 is non-trivial because of the irregular grid. We use a
    practical proxy: cast x to bf16, then to FP8 with random tie-breaking
    via a sub-LSB jitter. For evaluating the *recipe* this is sufficient;
    the kernel will issue native SR via PTX intrinsics.
    """
    fp_max = _fp8_max(mode)
    x = x.clamp(-fp_max, fp_max).to(torch.float32)
    # add uniform jitter scaled to the smallest representable step locally
    # (use bf16 ULP as an approximation since FP8 ULP varies wildly).
    eps = (x.abs().clamp_min(1.0)) * (1.0 / 256.0)  # ~ FP8 e4m3 typical step
    noise = (torch.rand_like(x) - 0.5) * eps
    return (x + noise).to(_fp8_dtype(mode)).to(torch.float32)


# ---------------------------------------------------------------------------
# Reference backward (fp32, "ground truth")
# ---------------------------------------------------------------------------


def reference_backward(Q, K, V, dO, *, causal=False):
    """Standard attention backward in fp32. Returns (dQ, dK, dV)."""
    head_dim = Q.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(head_dim)
    S = (Q @ K.transpose(-2, -1)) * inv_sqrt_d
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), 1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)

    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    rowsum = (P * dP).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)
    dQ = (dS @ K) * inv_sqrt_d
    dK = (dS.transpose(-2, -1) @ Q) * inv_sqrt_d
    return dQ, dK, dV


# ---------------------------------------------------------------------------
# FP8 backward consistent with the kernel forward
# ---------------------------------------------------------------------------


@dataclass
class BwdRecipe:
    """Knobs the ablation sweep tweaks. Defaults match the recommendation."""
    grad_mode: str = "fp8_e4m3"        # FP8 dtype for dO/dS
    grad_granularity: str = "per_row"  # "per_row" | "per_tensor" | "none"
    pv_mode: str = "fp8_e4m3"          # P, V FP8 dtype for the PV-style mmas
    quant_p: bool = True               # FP8 P (static 1/max) for dV path
    quant_v: bool = True               # FP8 V per-channel for dP path
    quant_grads: bool = True           # if False, dO, dS stay in fp32
    stochastic: bool = True            # SR on dQ/dK upcasts (recipe)


def fp8_attention_backward(
    Q_q: torch.Tensor,            # (..., N, D) fp32 (already de-quantized Q)
    K_centered_q: torch.Tensor,   # (..., N, D) fp32 (de-quantized K_centered)
    V_centered: torch.Tensor,     # (..., N, D) fp32 (V - V_mean), kernel input
    L: torch.Tensor,              # (..., N, 1) fp32 log-sum-exp from fwd
    dO: torch.Tensor,             # (..., N, D) fp32 incoming gradient
    *,
    causal: bool = False,
    recipe: Optional[BwdRecipe] = None,
):
    """Mirrors the kernel forward path exactly, then runs the FP8 backward.

    The kernel computed S on the *centered* K and saved
        L = log_sum_exp((Q_q @ K_centered_q^T) / sqrt(D))
    so we recompute on the same centered K to recover P consistently:
        P_ij = exp(S_ij - L_i)            (no shift correction needed)

    Mathematically the K-mean and V-mean smoothing leaves the gradients
    invariant w.r.t. the *original* (un-smoothed) Q,K,V:
      * K_mean is a per-row additive constant in S, killed by softmax,
        and dQ -= dS @ K_mean = 0 since softmax-jacobian rows sum to 0.
      * V_mean is added back to O after PV; treating it as a constant
        (no autograd through the mean) gives dV = P^T @ dO directly.

    Then the SVQ recipe is applied: per-row dO/dS in FP8, FP8 P with
    static 1/max scale, FP8 V per-channel, dS / sqrt(D) is folded into
    the gradient before the kernel-visible matmul.

    Returns (dQ, dK, dV) in fp32, with dQ/dK matching gradients w.r.t.
    the *original* Q,K (smoothing-invariant), and dV w.r.t. the kernel-
    visible V (V_centered) — equal to dV w.r.t. original V when V_mean
    is treated as a detached constant.
    """
    recipe = recipe or BwdRecipe()
    head_dim = Q_q.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(head_dim)

    # ---- Recompute P from saved L on the centered tensors ---------------
    S = (Q_q @ K_centered_q.transpose(-2, -1)) * inv_sqrt_d
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), 1)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.exp(S - L)

    # ---- Quantize P, V, dO for the FP8 mmas -----------------------------
    if recipe.quant_p:
        P_q = quantize_p_static_fp8(P, recipe.pv_mode)
    else:
        P_q = P

    if recipe.quant_v:
        V_q = quantize_v_per_channel_fp8(V_centered, recipe.pv_mode)
    else:
        V_q = V_centered

    if recipe.quant_grads:
        if recipe.grad_granularity == "per_row":
            dO_q, _ = quantize_per_row_fp8(
                dO, recipe.grad_mode, stochastic=recipe.stochastic)
        elif recipe.grad_granularity == "per_tensor":
            dO_q = quantize_per_tensor_fp8(
                dO, recipe.grad_mode, stochastic=recipe.stochastic)
        elif recipe.grad_granularity == "none":
            dO_q = dO
        else:
            raise ValueError(recipe.grad_granularity)
    else:
        dO_q = dO

    # ---- dV = P^T @ dO  (kernel-visible V receives this gradient) -------
    dV = P_q.transpose(-2, -1) @ dO_q

    # ---- dP = dO @ V^T --------------------------------------------------
    dP = dO_q @ V_q.transpose(-2, -1)

    # ---- dS via fp32 softmax-jacobian (always in fp32) -------------------
    rowsum = (P * dP).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum) * inv_sqrt_d

    # ---- dS quantization (per the recipe: per-row, SR, FP8) -------------
    if recipe.quant_grads:
        if recipe.grad_granularity == "per_row":
            dS_q, _ = quantize_per_row_fp8(
                dS, recipe.grad_mode, stochastic=recipe.stochastic)
        elif recipe.grad_granularity == "per_tensor":
            dS_q = quantize_per_tensor_fp8(
                dS, recipe.grad_mode, stochastic=recipe.stochastic)
        else:
            dS_q = dS
    else:
        dS_q = dS

    # ---- dQ, dK : reuse the *same* dequantized K, Q the forward saw ----
    dQ = dS_q @ K_centered_q
    dK = dS_q.transpose(-2, -1) @ Q_q

    return dQ, dK, dV
