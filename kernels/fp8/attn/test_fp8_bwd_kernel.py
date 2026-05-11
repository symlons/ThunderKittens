"""End-to-end test of the FP8 backward CUDA kernel.

The kernel is "FP8-recipe-consistent": it consumes Q,K that were FP8
quantized + de-quantized on the host (so the per-row sq, sk scales are
baked into bf16 values the bwd actually sees), V centered (bf16), and
the L from the FP8 forward kernel. It returns dQ, dK, dV in fp32.

Compares dQ/dK/dV against:
  * ref-fp32-bwd       : PyTorch fp32 backward on the original Q,K,V
  * ref-pyfp8-bwd      : the Python `fp8_attn_bwd_ref.fp8_attention_backward`
                         on the same kernel-visible inputs (no FP8 dO/dS
                         quant — pure bf16 path; matches the kernel)

Usage:
    cd attn
    make BUILD_MODE=torch KERNEL=fp8
    python3 test_fp8_bwd_kernel.py
    python3 test_fp8_bwd_kernel.py --quick
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from fp8_attn_bwd_ref import reference_backward

FP8_E4M3_MAX = 448.0


def metrics(out, ref):
    out = out.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    diff = (out - ref).abs()
    abs_max = diff.max().item()
    rel_l1 = diff.sum().item() / max(ref.abs().sum().item(), 1e-30)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    sig = (ref * ref).sum().clamp_min(1e-30)
    noise = (diff * diff).sum().clamp_min(1e-30)
    ratio = (sig / noise).item()
    qsnr = 10.0 * math.log10(ratio) if (ratio > 0 and math.isfinite(ratio)) else float("nan")
    return {"max": abs_max, "rel_L1": rel_l1, "cos": cos, "qsnr_dB": qsnr}


def fmt(label, dQm, dKm, dVm):
    return (f"{label:<28} "
            f"dQ[QSNR={dQm['qsnr_dB']:5.2f} relL1={dQm['rel_L1']:.2e}] "
            f"dK[QSNR={dKm['qsnr_dB']:5.2f} relL1={dKm['rel_L1']:.2e}] "
            f"dV[QSNR={dVm['qsnr_dB']:5.2f} relL1={dVm['rel_L1']:.2e}]")


def host_quantize_per_row_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale.squeeze(-1)


def host_quantize_per_row_fp8_sr_dequant(x, *, gen=None):
    """Per-row FP8 e4m3 SR fake-quant, returning a bf16 dequantized tensor.

    Used for dO on the host side: the bwd kernel's bf16 mmas see the
    FP8-grid values with stochastic rounding. Mathematically equivalent
    to FP8 mma + fp32 accumulator since the kernel accumulators are fp32.

    SR is implemented as: jitter by uniform[-0.5, 0.5] * local_ULP in
    FP8-units before RTNE cast. Local ULP is approximated as
    max(2^floor(log2|y|) - 23, 2^-9) where y = x/scale (in FP8 units).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    y = x / scale
    # crude per-element ULP for FP8 e4m3 (3 mantissa bits): 2^(exp-3)
    # for normal range; 2^-9 for subnormals.
    abs_y = y.abs().clamp_min(2.0 ** -6)
    exp = torch.floor(torch.log2(abs_y))
    ulp = (2.0 ** (exp - 3.0)).clamp_min(2.0 ** -9)
    if gen is None:
        noise = (torch.rand_like(y) - 0.5) * ulp
    else:
        noise = (torch.rand(y.shape, generator=gen,
                            device=y.device, dtype=y.dtype) - 0.5) * ulp
    y_jit = (y + noise).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    xq = y_jit.to(torch.float8_e4m3fn)
    return (xq.to(torch.float32) * scale).to(torch.bfloat16).contiguous()


def host_quantize_per_row_fp8_sr(x, *, gen=None):
    """Per-row FP8 e4m3 SR quant, returning (xq_fp8, scale_per_row)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    y = x / scale
    abs_y = y.abs().clamp_min(2.0 ** -6)
    exp = torch.floor(torch.log2(abs_y))
    ulp = (2.0 ** (exp - 3.0)).clamp_min(2.0 ** -9)
    if gen is None:
        noise = (torch.rand_like(y) - 0.5) * ulp
    else:
        noise = (torch.rand(y.shape, generator=gen,
                            device=y.device, dtype=y.dtype) - 0.5) * ulp
    y_jit = (y + noise).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    xq = y_jit.to(torch.float8_e4m3fn)
    return xq.contiguous(), scale.squeeze(-1).contiguous()


def host_quantize_per_channel_fp8_sr(x, *, gen=None):
    """Per-channel (last-dim across N) FP8 e4m3 SR quant of x with shape
    (..., D, N): each row of x (length N along the last dim) gets its own
    scale = amax / 448. Used for the transposed copies q_t, og_t fed to
    the dV / dK FP8 mmas.

    Returns (xq_fp8 (..., D, N), scale (..., D)).
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    y = x / scale
    abs_y = y.abs().clamp_min(2.0 ** -6)
    exp = torch.floor(torch.log2(abs_y))
    ulp = (2.0 ** (exp - 3.0)).clamp_min(2.0 ** -9)
    if gen is None:
        noise = (torch.rand_like(y) - 0.5) * ulp
    else:
        noise = (torch.rand(y.shape, generator=gen,
                            device=y.device, dtype=y.dtype) - 0.5) * ulp
    y_jit = (y + noise).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    xq = y_jit.to(torch.float8_e4m3fn)
    return xq.contiguous(), scale.squeeze(-1).contiguous()


def kernel_fwd_bwd(Q, K, V, dO, smooth_k=True, smooth_v=True,
                   *, sr_dO=True, fp8_dS_mode=0):
    """Forward+Backward through the FP8 kernel (non-causal / bidirectional).

    Forward consumes Q, K as FP8 e4m3 + per-row scales.

    Backward consumes:
      * Q, K, V, dO as FP8 e4m3 (per-token scales sq, sk, sv, sdo_row)
      * Q_t, dO_t  as FP8 e4m3 transposed copies with per-channel SR scales
        (sq_ch, sdo_ch) — fed to the dV / dK FP8 mma_ABt
      * K_bf bf16 SHADOW copy — fed to the dQ bf16 mma_AtB
        (FP8 mma_AtB unsupported in TK at this commit)
      * O (bf16) and dO (bf16) for the prep kernel that computes D = sum(O*dO)
      * L from the forward (fp32)
    """
    import _C
    K_mean = K.mean(dim=-2, keepdim=True) if smooth_k else torch.zeros_like(K[..., :1, :])
    V_mean = V.mean(dim=-2, keepdim=True) if smooth_v else torch.zeros_like(V[..., :1, :])
    K_s = K - K_mean
    V_s = V - V_mean

    # Forward FP8 quants (per-token Q, K).
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

    # Backward FP8 inputs + scales.
    # Per-token quants for the "primary" operands (S^T uses Qq/Kq, dP^T uses Vq/dOq).
    # We re-use the forward Qq/sq, Kq/sk and add Vq/sv, dOq/sdo_row.
    Vq, sv = host_quantize_per_row_fp8(V_s)
    if sr_dO:
        dOq, sdo_row = host_quantize_per_row_fp8_sr(dO)
    else:
        dOq, sdo_row = host_quantize_per_row_fp8(dO)

    # Per-channel SR transposed FP8 copies (rows = D, cols = N).
    # sq_ch[d] is the per-channel scale of Q (per-row of Q^T).
    Q_T  = Q.transpose(-1, -2).contiguous()   # (B, H, D, N)
    dO_T = dO.transpose(-1, -2).contiguous()  # (B, H, D, N)
    Qq_t,  sq_ch  = host_quantize_per_channel_fp8_sr(Q_T)
    dOq_t, sdo_ch = host_quantize_per_channel_fp8_sr(dO_T)

    # bf16 shadow K (centered) for the dQ bf16 mma_AtB.
    K_bf = K_s.to(torch.bfloat16).contiguous()

    # bf16 dO copy for the prep kernel (snapped to FP8 grid if SR-dO is on).
    if sr_dO:
        dO_bf = host_quantize_per_row_fp8_sr_dequant(dO)
    else:
        dO_bf = dO.to(torch.bfloat16).contiguous()

    qg, kg, vg = _C.fp8_mha_backward(
        Qq, Kq, Vq, dOq,
        Qq_t, dOq_t,
        K_bf,
        o, dO_bf,
        l,
        sq.to(torch.float32).contiguous(),
        sk.to(torch.float32).contiguous(),
        sv.to(torch.float32).contiguous(),
        sdo_row.to(torch.float32).contiguous(),
        sq_ch.to(torch.float32).contiguous(),
        sdo_ch.to(torch.float32).contiguous(),
        fp8_dS_mode=fp8_dS_mode,
    )
    return {
        "O": o.to(torch.float32), "L": l,
        "Q_fp8": Qq, "K_fp8": Kq, "V_fp8": Vq, "dO_fp8": dOq,
        "dQ": qg.to(torch.float32),
        "dK": kg.to(torch.float32),
        "dV": vg.to(torch.float32),
    }


def run_one(B, H, N, D, seed):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    print(f"\n[B={B} H={H} N={N} D={D} seed={seed}]")

    # ground truth (non-causal)
    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO, causal=False)

    # Recipe (SR-dO + FP8 SR-dS in-kernel)
    res_recipe = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=2)
    # SR-dO + FP8 RTNE-dS in-kernel (no SR)
    res_rtne = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=1)
    # SR-dO + bf16 dS (current baseline before the dS-quant work)
    res_sr_dO = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=0)
    # No quant on either dO or dS (cleanest baseline)
    res_no    = kernel_fwd_bwd(Q, K, V, dO, sr_dO=False, fp8_dS_mode=0)

    print(" ", fmt("kernel recipe SR-dO+SR-dS",
                   metrics(res_recipe["dQ"], dQ_ref),
                   metrics(res_recipe["dK"], dK_ref),
                   metrics(res_recipe["dV"], dV_ref)))
    print(" ", fmt("kernel SR-dO + RTNE-dS  ",
                   metrics(res_rtne["dQ"], dQ_ref),
                   metrics(res_rtne["dK"], dK_ref),
                   metrics(res_rtne["dV"], dV_ref)))
    print(" ", fmt("kernel SR-dO + bf16 dS  ",
                   metrics(res_sr_dO["dQ"], dQ_ref),
                   metrics(res_sr_dO["dK"], dK_ref),
                   metrics(res_sr_dO["dV"], dV_ref)))
    print(" ", fmt("kernel no-quant         ",
                   metrics(res_no["dQ"], dQ_ref),
                   metrics(res_no["dK"], dK_ref),
                   metrics(res_no["dV"], dV_ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    try:
        import _C
        if not hasattr(_C, "fp8_mha_backward"):
            raise SystemExit("Rebuild: the _C module has no fp8_mha_backward "
                             "symbol. Run `make BUILD_MODE=torch KERNEL=fp8`.")
    except ImportError:
        raise SystemExit("Build the FP8 extension first.")

    if args.quick:
        configs = [(1, 8, 1536, 128, 0)]
    else:
        configs = [
            (1, 8, 1536, 128, 0),
            (1, 8, 1536,  64, 0),
            (2, 16, 1536, 128, 1),
        ]

    for cfg in configs:
        run_one(*cfg)


if __name__ == "__main__":
    main()
