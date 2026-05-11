"""Extensive correctness suite for the FP8 attention forward kernel.

Compares the kernel against three references:

  ref-fp32   : full-precision attention (the "ground truth" the recipe is
               approximating).
  ref-quant  : same FP8 fake-quantization as the kernel sees (kernel error
               only — accumulation order, bf16 P/V cast).
  ref-sim    : the simulation's smooth_core.quantized_attention with
               qk_mode=fp8_e4m3, smooth_k=True, smooth_v=True,
               granularity=per_token. This is the gold model for the
               recipe.

Reports per (B,H,N,D,causal,seed) combination:
  max|Δ|         peak absolute error
  mean|Δ|        average absolute error
  rel-L1         L1 error / |ref|
  rel-L_inf      max error / max |ref|
  cosine         cosine similarity
  QSNR (dB)      10*log10(|ref|^2 / |Δ|^2)

Also runs a smoothing-on / smoothing-off ablation to confirm smoothing
*helps* (it should: the recipe says it does).

Usage:
    cd attn
    make BUILD_MODE=torch KERNEL=fp8
    python3 test_fp8_extensive.py
    python3 test_fp8_extensive.py --quick   # one config per axis
    python3 test_fp8_extensive.py --cogvideox  # use captured Q,K,V
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Quantisation helpers (match the kernel host-side path)
# ---------------------------------------------------------------------------


def quantize_per_row_fp8(x):
    """x: (..., N, D); returns (xq fp8e4m3, scale (..., N) float32)."""
    amax  = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq    = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale.squeeze(-1)


def reference_attention(Q, K, V, causal=False):
    head_dim = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), 1)
        S = S.masked_fill(mask, float("-inf"))
    P = F.softmax(S, dim=-1)
    return P @ V


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(out, ref):
    out = out.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    diff = (out - ref).abs()
    abs_max = diff.max().item()
    abs_mean = diff.mean().item()
    rel_l1 = diff.sum().item() / ref.abs().sum().clamp_min(1e-30).item()
    rel_linf = abs_max / max(ref.abs().max().item(), 1e-30)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    sig = (ref * ref).sum().clamp_min(1e-30)
    noise = (diff * diff).sum().clamp_min(1e-30)
    qsnr = 10.0 * math.log10((sig / noise).item())
    return {
        "max":     abs_max,
        "mean":    abs_mean,
        "rel_L1":  rel_l1,
        "rel_Linf": rel_linf,
        "cos":     cos,
        "qsnr_dB": qsnr,
    }


def fmt_metrics(m):
    return (f"max={m['max']:.4f}  mean={m['mean']:.5f}  "
            f"rel-L1={m['rel_L1']:.3e}  rel-L∞={m['rel_Linf']:.3e}  "
            f"cos={m['cos']:.6f}  QSNR={m['qsnr_dB']:.2f} dB")


# ---------------------------------------------------------------------------
# Kernel call wrapper applying the recipe (K, V channel-mean smoothing)
# ---------------------------------------------------------------------------


def kernel_attention(Q, K, V, causal, smooth_k=True, smooth_v=True):
    """Run the FP8 e4m3 attention kernel with the SVQ recipe."""
    import _C
    if smooth_k:
        K_mean = K.mean(dim=-2, keepdim=True)
    else:
        K_mean = torch.zeros_like(K[..., :1, :])
    if smooth_v:
        V_mean = V.mean(dim=-2, keepdim=True)
    else:
        V_mean = torch.zeros_like(V[..., :1, :])
    K_s = K - K_mean
    V_s = V - V_mean

    Qq, sq = quantize_per_row_fp8(Q)
    Kq, sk = quantize_per_row_fp8(K_s)
    Vbf    = V_s.to(torch.bfloat16).contiguous()
    vm     = V_mean.squeeze(-2).to(torch.bfloat16).contiguous()

    if causal:
        raise NotImplementedError(
            "FP8 kernel is non-causal / bidirectional only (diffusion attention)")
    o, _ = _C.fp8_mha_forward(
        Qq, Kq, Vbf,
        sq.contiguous().to(torch.float32),
        sk.contiguous().to(torch.float32),
        vm,
    )
    return o.to(torch.float32), (Qq, Kq, sq, sk, K_mean, V_mean)


def fp8_quant_reference(Qq, Kq, sq, sk, K_mean, V, causal):
    """fp32 attention with the *same* fake-quant the kernel sees on Q,K.

    V stays fp32 so this isolates the kernel's own error from the FP8
    quantisation noise on V/P."""
    Q_eq = Qq.to(torch.float32) * sq.unsqueeze(-1)
    K_eq = Kq.to(torch.float32) * sk.unsqueeze(-1) + K_mean
    return reference_attention(Q_eq, K_eq, V, causal=causal)


# ---------------------------------------------------------------------------
# Simulation reference (per-head, sample 0)
# ---------------------------------------------------------------------------


def simulation_attention(Q, K, V, causal, granularity="per_token"):
    """Run smooth_core.quantized_attention per (batch, head). Causal not
    supported by the sim's online softmax path, so this only runs when
    causal=False; otherwise returns None."""
    if causal:
        return None
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from smooth_core import quantized_attention
    B, H, N, D = Q.shape
    out = torch.empty_like(Q)
    for b in range(B):
        for h in range(H):
            out[b, h] = quantized_attention(
                Q[b, h], K[b, h], V[b, h],
                qk_mode="fp8_e4m3",
                smooth_q=False, smooth_k=True, smooth_v=True,
                granularity=granularity,
                pv_mode="fp8_e4m3",
            )
    return out.to(torch.float32)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------


def run_one(B, H, N, D, causal, seed, *, run_sim=False):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    O_ref_fp32 = reference_attention(Q, K, V, causal=causal)

    O_ker, (Qq, Kq, sq, sk, K_mean, V_mean) = kernel_attention(
        Q, K, V, causal, smooth_k=True, smooth_v=True
    )
    O_ref_quant = fp8_quant_reference(Qq, Kq, sq, sk, K_mean, V, causal)

    print(f"\n[B={B} H={H} N={N} D={D} causal={causal} seed={seed}]")
    print(f"  vs ref-fp32   {fmt_metrics(metrics(O_ker, O_ref_fp32))}")
    print(f"  vs ref-quant  {fmt_metrics(metrics(O_ker, O_ref_quant))}")
    if run_sim:
        O_sim = simulation_attention(Q, K, V, causal=causal)
        if O_sim is not None:
            print(f"  vs ref-sim    {fmt_metrics(metrics(O_ker, O_sim))}")
            # The sim itself vs fp32 — sanity check the sim is sane.
            print(f"  sim vs fp32   {fmt_metrics(metrics(O_sim, O_ref_fp32))}")


def run_smoothing_ablation(B, H, N, D, seed):
    """Confirm smoothing helps.

    Two tensor families are exercised:
      (a) random N(0,1) — channel means are already ~0, smoothing is noise
      (b) channel-biased  — adds per-channel offset of ±4σ; this is what
          real activation tensors look like and where smoothing earns its
          keep.
    """
    print("\n[smoothing ablation: B=%d H=%d N=%d D=%d seed=%d, non-causal]"
          % (B, H, N, D, seed))
    torch.manual_seed(seed)
    Q0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V0 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    bias_K = torch.randn(1, 1, 1, D, device="cuda") * 4.0
    bias_V = torch.randn(1, 1, 1, D, device="cuda") * 4.0

    families = [
        ("zero-mean   K,V", Q0, K0, V0),
        ("biased K,V (±4σ)", Q0, K0 + bias_K, V0 + bias_V),
    ]

    for tag, Q, K, V in families:
        print(f"  -- {tag} -- (K|max|={K.abs().max().item():.2f}, "
              f"V|max|={V.abs().max().item():.2f})")
        O_ref = reference_attention(Q, K, V, causal=False)
        for sk_on, sv_on in [(False, False), (True, False),
                              (False, True),  (True, True)]:
            O_ker, _ = kernel_attention(Q, K, V, causal=False,
                                         smooth_k=sk_on, smooth_v=sv_on)
            m = metrics(O_ker, O_ref)
            label = f"smooth K={int(sk_on)} V={int(sv_on)}"
            print(f"     {label:<22} {fmt_metrics(m)}")


def run_cogvideox():
    print("\n[CogVideoX-2b real Q,K,V]")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from smooth_core import make_inputs
    # full bundle, take batch 0
    bundle = torch.load(
        Path(__file__).parent.parent / "captures" / "cogvideox.pt",
        map_location="cuda", weights_only=False,
    )
    Q_full = bundle["Q"]; K_full = bundle["K"]; V_full = bundle["V"]
    if isinstance(Q_full, list):
        Q_full = Q_full[0]; K_full = K_full[0]; V_full = V_full[0]
    # (B=2, H=30, N=4096, D=64) → trim to (B=1, H=8, N=1536, D=64)
    Q = Q_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    K = K_full[:1, :8, :1536, :].to(torch.float32).contiguous()
    V = V_full[:1, :8, :1536, :].to(torch.float32).contiguous()

    O_ref = reference_attention(Q, K, V, causal=False)
    O_ker, _ = kernel_attention(Q, K, V, causal=False)
    print(f"  shape {tuple(Q.shape)}  V|max|={V.abs().max().item():.2f}  "
          f"K|max|={K.abs().max().item():.2f}")
    print(f"  vs ref-fp32   {fmt_metrics(metrics(O_ker, O_ref))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="One config per axis instead of the full sweep")
    ap.add_argument("--no-sim", action="store_true",
                    help="Skip the simulation reference (slow on big N)")
    ap.add_argument("--cogvideox", action="store_true",
                    help="Run on captured CogVideoX-2b Q,K,V")
    args = ap.parse_args()

    try:
        import _C  # noqa: F401
    except ImportError:
        raise SystemExit(
            "Build the FP8 extension first:\n"
            "  cd attn && make BUILD_MODE=torch KERNEL=fp8")

    # N must be a multiple of LCM(3*qo_height=192, kv_height=128) = 384.
    # FP8 kernel is non-causal / bidirectional only (diffusion attention).
    if args.quick:
        configs = [
            (1, 8, 1536, 128, False, 0),
            (1, 8, 1536,  64, False, 0),
        ]
    else:
        configs = []
        for D in (64, 128):
            for N in (384, 768, 1536, 3072):
                for seed in (0, 1):
                    configs.append((1, 8, N, D, False, seed))
        # plus a larger batch
        configs += [
            (2, 16, 1536, 128, False, 0),
            (2, 16, 1536, 64,  False, 0),
        ]

    for cfg in configs:
        run_one(*cfg, run_sim=not args.no_sim)

    run_smoothing_ablation(1, 8, 1536, 128, seed=0)

    if args.cogvideox:
        run_cogvideox()


if __name__ == "__main__":
    main()
