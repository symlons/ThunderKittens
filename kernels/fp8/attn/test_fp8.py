"""Smoke test for the mixed low-bit attention forward kernel.

Recipe applied (matches the kernel doc-comment / smooth_core sim):

    K_s = K - mean_channel(K)            # smoothing
    V_s = V - mean_channel(V)            # smoothing, mean added back to O
    Q   ← fp8 e4m3, per-row scale
    K_s ← fp8 e4m3, per-row scale
    V_s ← bf16 (kernel keeps PV in bf16 in this revision)
    vm  ← V channel-mean in bf16

Run:

    cd attn
    make BUILD_MODE=torch KERNEL=fp8
    python3 test_fp8.py
"""
import math
import torch
import torch.nn.functional as F


FP8_E4M3_MAX = 448.0


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


def main():
    try:
        import _C
    except ImportError as e:
        raise SystemExit(
            "Build the FP8 extension first:\n"
            "  cd attn && make BUILD_MODE=torch KERNEL=fp8\n"
            f"\nimport error: {e}"
        )

    if not hasattr(_C, "fp8_mha_forward"):
        raise SystemExit("loaded _C does not export fp8_mha_forward — "
                          "rebuild with KERNEL=fp8")

    import sys
    # FP8 kernel is non-causal / bidirectional only (diffusion attention).
    causal = False
    D = 64 if "--d64" in sys.argv else 128
    torch.manual_seed(0)
    # N must be divisible by CONSUMER_WARPGROUPS(3) * qo_height(64) = 192.
    B, H, N = 1, 8, 1536
    print(f"B={B} H={H} N={N} D={D} (non-causal)")

    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    # ---- smoothing: K_s = K - mean_chan(K), V_s = V - mean_chan(V) ----------
    K_mean = K.mean(dim=-2, keepdim=True)        # (B, H, 1, D)
    V_mean = V.mean(dim=-2, keepdim=True)        # (B, H, 1, D)
    K_s = K - K_mean
    V_s = V - V_mean

    Qq, sq = quantize_per_row_fp8(Q)             # sq: (B, H, N)
    Kq, sk = quantize_per_row_fp8(K_s)           # sk: (B, H, N)
    Vbf    = V_s.to(torch.bfloat16).contiguous() # bf16 V
    vm     = V_mean.squeeze(-2).to(torch.bfloat16).contiguous()  # (B, H, D)

    o, l = _C.fp8_mha_forward(
        Qq, Kq, Vbf,
        sq.contiguous().to(torch.float32),
        sk.contiguous().to(torch.float32),
        vm,
    )

    # Reference 1: same fake-quanted Q,K (so kernel rounding is the only diff)
    Q_eq = (Qq.to(torch.float32) * sq.unsqueeze(-1))
    K_eq = (Kq.to(torch.float32) * sk.unsqueeze(-1))    # this is K_s_q
    K_eq_full = K_eq + K_mean                            # un-do the smoothing
    O_ref_quant = reference_attention(Q_eq, K_eq_full, V, causal=causal)
    O_ref_fp32  = reference_attention(Q, K, V, causal=causal)
    O_tk        = o.to(torch.float32)

    def report(name, ref):
        diff = (O_tk - ref).abs()
        rel  = diff.sum().item() / ref.abs().sum().clamp_min(1e-12).item()
        print(f"  {name:<30} max|Δ|={diff.max().item():.4f}  rel-L1={rel:.4e}")

    print("kernel output stats:")
    print(f"  O kern : mean={O_tk.mean().item():.4f}  "
          f"max={O_tk.abs().max().item():.4f}  has_nan={torch.isnan(O_tk).any().item()}")
    print(f"  O qref : mean={O_ref_quant.mean().item():.4f}  "
          f"max={O_ref_quant.abs().max().item():.4f}")
    print(f"  O ref  : mean={O_ref_fp32.mean().item():.4f}  "
          f"max={O_ref_fp32.abs().max().item():.4f}")
    print()
    print("error vs references:")
    report("vs fp8-quant reference", O_ref_quant)
    report("vs fp32 reference",      O_ref_fp32)


if __name__ == "__main__":
    main()
