"""Backward-pass correctness + ablation suite for the FP8 attention kernel.

This script does NOT call a CUDA backward kernel (none yet). It runs the
Python reference backward (`fp8_attn_bwd_ref.fp8_attention_backward`) on the
exact tensors the forward kernel produced (Q_q, K_q, V_centered, sq, sk,
K_mean, V_mean, L) and compares dQ/dK/dV against:

    ref-fp32    standard fp32 attention backward (PyTorch autograd)
    ref-bwd-fp32  same backward but receiving the kernel's quantized
                  Q,K,V — isolates the dQ/dK/dV quant noise from the
                  forward fwd-quant noise.

Ablations:
    * grad granularity : per_row vs per_tensor vs none
    * grad dtype       : fp8_e4m3 vs fp8_e5m2 vs no-quant
    * P/V dtype        : fp8_e4m3 vs fp8_e5m2 vs no-quant
    * smoothing        : K-mean / V-mean on/off
    * SR vs RTNE       : stochastic rounding on dQ/dK quant

Usage:
    cd attn
    make BUILD_MODE=torch KERNEL=fp8
    python3 test_fp8_bwd.py
    python3 test_fp8_bwd.py --quick
    python3 test_fp8_bwd.py --cogvideox
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from fp8_attn_bwd_ref import (
    BwdRecipe,
    fp8_attention_backward,
    quantize_per_row_fp8,
    reference_backward,
)

FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def metrics(out, ref):
    out = out.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    diff = (out - ref).abs()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    rel_l1 = diff.sum().item() / max(ref.abs().sum().item(), 1e-30)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    sig = (ref * ref).sum().clamp_min(1e-30)
    noise = (diff * diff).sum().clamp_min(1e-30)
    ratio = (sig / noise).item()
    qsnr = 10.0 * math.log10(ratio) if (ratio > 0 and math.isfinite(ratio)) else float("nan")
    return {"max": abs_max, "mean": abs_mean, "rel_L1": rel_l1,
            "cos": cos, "qsnr_dB": qsnr}


def fmt(m):
    return (f"rel-L1={m['rel_L1']:.3e}  cos={m['cos']:.6f}  "
            f"QSNR={m['qsnr_dB']:5.2f} dB  max={m['max']:.3e}")


def fmt_grads(label, dQm, dKm, dVm):
    return (f"{label:<28} "
            f"dQ[QSNR={dQm['qsnr_dB']:5.2f} relL1={dQm['rel_L1']:.2e}] "
            f"dK[QSNR={dKm['qsnr_dB']:5.2f} relL1={dKm['rel_L1']:.2e}] "
            f"dV[QSNR={dVm['qsnr_dB']:5.2f} relL1={dVm['rel_L1']:.2e}]")


# ---------------------------------------------------------------------------
# Forward kernel call (mirrors test_fp8_extensive.kernel_attention) — only
# this time we save and return the kernel's L too.
# ---------------------------------------------------------------------------


def host_quantize_per_row_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale.squeeze(-1)


def kernel_forward(Q, K, V, causal, smooth_k=True, smooth_v=True):
    """Run the FP8 e4m3 forward kernel; return O, L plus all the kernel-
    visible quantized inputs needed by the backward reference.

    The kernel is non-causal / bidirectional only (diffusion attention).
    `causal` here only controls the python-side reference comparisons.
    """
    if causal:
        raise NotImplementedError(
            "FP8 kernel is non-causal / bidirectional only (diffusion attention)")
    import _C
    K_mean = K.mean(dim=-2, keepdim=True) if smooth_k else torch.zeros_like(K[..., :1, :])
    V_mean = V.mean(dim=-2, keepdim=True) if smooth_v else torch.zeros_like(V[..., :1, :])
    K_s = K - K_mean
    V_s = V - V_mean

    Qq, sq = host_quantize_per_row_fp8(Q)
    Kq, sk = host_quantize_per_row_fp8(K_s)
    Vbf = V_s.to(torch.bfloat16).contiguous()
    vm = V_mean.squeeze(-2).to(torch.bfloat16).contiguous()

    o, l = _C.fp8_mha_forward(
        Qq, Kq, Vbf,
        sq.contiguous().to(torch.float32),
        sk.contiguous().to(torch.float32),
        vm,
    )

    # Reconstruct fp32 kernel-visible (centered) tensors. K_centered carries
    # the K-mean shift inside L (cancels in softmax); V_centered is the
    # tile actually streamed into the PV mma; V_mean was added back to O.
    Q_eff = Qq.to(torch.float32) * sq.unsqueeze(-1)
    K_eff_centered = Kq.to(torch.float32) * sk.unsqueeze(-1)
    V_centered_fp32 = Vbf.to(torch.float32)

    # The forward kernel writes:
    #     L_kernel = -sqrt(D) * log_sum_exp(S)   with S = (Q@K_centered^T)/sqrt(D)
    # We need log_sum_exp(S) so that  P = exp(S - L_natural).
    inv_sqrt_d = 1.0 / math.sqrt(Q.shape[-1])
    L = -l.to(torch.float32) * inv_sqrt_d                    # (B,H,N,1)

    return {
        "O": o.to(torch.float32),
        "L": L,
        "Q_eff": Q_eff, "K_eff_centered": K_eff_centered,
        "V_centered": V_centered_fp32,
    }


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def run_one(B, H, N, D, causal, seed, *, ablate=True):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    print(f"\n[B={B} H={H} N={N} D={D} causal={causal} seed={seed}]")

    # ground-truth fp32 backward
    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO, causal=causal)

    # forward kernel — provides L + the kernel-visible quantized tensors
    fwd = kernel_forward(Q, K, V, causal)

    # backward consistent with kernel forward (best recipe)
    dQ, dK, dV = fp8_attention_backward(
        fwd["Q_eff"], fwd["K_eff_centered"], fwd["V_centered"],
        fwd["L"], dO, causal=causal,
        recipe=BwdRecipe(),  # defaults = recommended recipe
    )
    print(" ",
          fmt_grads("recipe (e4m3, per_row, SR)",
                    metrics(dQ, dQ_ref), metrics(dK, dK_ref),
                    metrics(dV, dV_ref)))

    if not ablate:
        return

    # ----- ablations ------------------------------------------------------
    sweeps = [
        ("no grad-quant",
         BwdRecipe(quant_grads=False, quant_p=False, quant_v=False)),
        ("grad e4m3 RTNE per_row",
         BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_row", stochastic=False)),
        ("grad e4m3 SR  per_row",
         BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_row", stochastic=True)),
        ("grad e4m3 SR  per_tensor",
         BwdRecipe(grad_mode="fp8_e4m3", grad_granularity="per_tensor", stochastic=True)),
        ("grad e5m2 SR  per_row",
         BwdRecipe(grad_mode="fp8_e5m2", grad_granularity="per_row", stochastic=True)),
        ("grad e5m2 SR  per_tensor",
         BwdRecipe(grad_mode="fp8_e5m2", grad_granularity="per_tensor", stochastic=True)),
        ("e4m3 SR per_row, V fp32",
         BwdRecipe(quant_v=False)),
        ("e4m3 SR per_row, P fp32",
         BwdRecipe(quant_p=False)),
    ]
    for label, recipe in sweeps:
        dQ, dK, dV = fp8_attention_backward(
            fwd["Q_eff"], fwd["K_eff_centered"], fwd["V_centered"],
            fwd["L"], dO, causal=causal, recipe=recipe,
        )
        print(" ",
              fmt_grads(label,
                        metrics(dQ, dQ_ref), metrics(dK, dK_ref),
                        metrics(dV, dV_ref)))


def run_smoothing_ablation(B, H, N, D, seed):
    """K/V-mean smoothing on biased activations — does smoothing actually
    improve dQ/dK/dV under the FP8 backward recipe?"""
    print(f"\n[bwd smoothing ablation B={B} H={H} N={N} D={D} seed={seed}]")
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    bias_K = torch.randn(1, 1, 1, D, device="cuda") * 4.0
    bias_V = torch.randn(1, 1, 1, D, device="cuda") * 4.0
    K = K + bias_K
    V = V + bias_V
    print(f"  biased K|max|={K.abs().max().item():.2f}  "
          f"V|max|={V.abs().max().item():.2f}")

    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO)

    for sk_on, sv_on in [(False, False), (True, False), (False, True), (True, True)]:
        fwd = kernel_forward(Q, K, V, causal=False,
                             smooth_k=sk_on, smooth_v=sv_on)
        dQ, dK, dV = fp8_attention_backward(
            fwd["Q_eff"], fwd["K_eff_centered"], fwd["V_centered"],
            fwd["L"], dO, recipe=BwdRecipe(),
        )
        label = f"smooth K={int(sk_on)} V={int(sv_on)}"
        print(" ",
              fmt_grads(label,
                        metrics(dQ, dQ_ref), metrics(dK, dK_ref),
                        metrics(dV, dV_ref)))


def run_cogvideox():
    print("\n[CogVideoX-2b real Q,K,V — backward]")
    bundle = torch.load(
        Path(__file__).parent.parent / "captures" / "cogvideox.pt",
        map_location="cuda", weights_only=False,
    )
    Q_full = bundle["Q"]; K_full = bundle["K"]; V_full = bundle["V"]
    if isinstance(Q_full, list):
        Q_full = Q_full[0]; K_full = K_full[0]; V_full = V_full[0]
    Q = Q_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    K = K_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    V = V_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    dO = torch.randn_like(Q)

    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO)

    for sk_on, sv_on in [(False, False), (True, True)]:
        fwd = kernel_forward(Q, K, V, causal=False,
                             smooth_k=sk_on, smooth_v=sv_on)
        dQ, dK, dV = fp8_attention_backward(
            fwd["Q_eff"], fwd["K_eff_centered"], fwd["V_centered"],
            fwd["L"], dO, recipe=BwdRecipe(),
        )
        label = f"K={int(sk_on)} V={int(sv_on)}"
        print(" ",
              fmt_grads(label,
                        metrics(dQ, dQ_ref), metrics(dK, dK_ref),
                        metrics(dV, dV_ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cogvideox", action="store_true")
    args = ap.parse_args()

    try:
        import _C  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Build the FP8 extension first:\n"
            "  cd attn && make BUILD_MODE=torch KERNEL=fp8")

    # FP8 kernel is non-causal / bidirectional only (diffusion attention).
    if args.quick:
        configs = [(1, 8, 1536, 128, False, 0)]
    else:
        configs = [
            (1, 8, 1536, 128, False, 0),
            (1, 8, 1536,  64, False, 0),
            (2, 16, 1536, 128, False, 1),
        ]

    for cfg in configs:
        run_one(*cfg)

    run_smoothing_ablation(1, 8, 1536, 128, seed=0)
    if args.cogvideox:
        run_cogvideox()


if __name__ == "__main__":
    main()
