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

import torch
import torch.nn.functional as F

from fp8_attn_bwd_ref import reference_backward

FP8_E4M3_MAX = 448.0


def sdpa_backward(Q, K, V, dO, *, causal=False):
    """PyTorch builtin SDPA/FA2 autograd reference."""
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    O_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=causal)
    O_ref.backward(dO.detach())
    return O_ref.detach(), Q_ref.grad.detach(), K_ref.grad.detach(), V_ref.grad.detach()


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


def attention_fwd_bwd_tflops(B, H, N, D, ms):
    # Non-causal dense attention. Forward has QK^T and PV; backward recomputes
    # QK^T and runs dP, dV, dQ, dK matmuls. Softmax/reductions are not counted.
    flops = 14.0 * B * H * N * N * D
    return flops / (ms * 1.0e-3) / 1.0e12


def attention_fwd_tflops(B, H, N, D, ms):
    flops = 4.0 * B * H * N * N * D
    return flops / (ms * 1.0e-3) / 1.0e12


def attention_bwd_tflops(B, H, N, D, ms):
    flops = 10.0 * B * H * N * N * D
    return flops / (ms * 1.0e-3) / 1.0e12


def gbps(num_bytes, ms):
    return num_bytes / (ms * 1.0e-3) / 1.0e9


def time_ms(fn, *, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def check_metrics(label, dQm, dKm, dVm, *, min_qsnr, max_rel_l1, min_cos):
    failures = []
    for name, m in (("dQ", dQm), ("dK", dKm), ("dV", dVm)):
        if not math.isfinite(m["qsnr_dB"]) or m["qsnr_dB"] < min_qsnr:
            failures.append(f"{name} QSNR {m['qsnr_dB']:.2f} < {min_qsnr:.2f}")
        if m["rel_L1"] > max_rel_l1:
            failures.append(f"{name} relL1 {m['rel_L1']:.3e} > {max_rel_l1:.3e}")
        if m["cos"] < min_cos:
            failures.append(f"{name} cos {m['cos']:.6f} < {min_cos:.6f}")
    if failures:
        raise AssertionError(f"{label} failed: " + "; ".join(failures))


def fmt(label, dQm, dKm, dVm):
    return (f"{label:<28} "
            f"dQ[QSNR={dQm['qsnr_dB']:5.2f} relL1={dQm['rel_L1']:.2e} cos={dQm['cos']:.5f}] "
            f"dK[QSNR={dKm['qsnr_dB']:5.2f} relL1={dKm['rel_L1']:.2e} cos={dKm['cos']:.5f}] "
            f"dV[QSNR={dVm['qsnr_dB']:5.2f} relL1={dVm['rel_L1']:.2e} cos={dVm['cos']:.5f}]")


def host_quantize_per_row_fp8(x):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq, scale.squeeze(-1)


def cuda_quantize_per_token(x):
    import _C
    return _C.fp8_quantize_per_token(x.contiguous())


def cuda_quantize_per_channel(x):
    import _C
    return _C.fp8_quantize_per_channel(x.contiguous())


def cuda_quantize_per_token_out(x, xq, scale):
    import _C
    _C.fp8_quantize_per_token_out(x.contiguous(), xq, scale)


def cuda_quantize_per_channel_out(x, xq, scale):
    import _C
    _C.fp8_quantize_per_channel_out(x.contiguous(), xq, scale)


def check_quant_kernel(label, xq, scale, xq_ref, scale_ref, *, granularity):
    scale_m = metrics(scale, scale_ref)
    if scale_m["rel_L1"] > 1e-6 or scale_m["max"] > 1e-6:
        raise AssertionError(
            f"{label} scale mismatch: max={scale_m['max']:.3e}, relL1={scale_m['rel_L1']:.3e}")
    if granularity == "token":
        deq = xq.to(torch.float32) * scale.unsqueeze(-1)
        ref = xq_ref.to(torch.float32) * scale_ref.unsqueeze(-1)
    elif granularity == "channel":
        deq = xq.to(torch.float32) * scale.unsqueeze(-2)
        ref = xq_ref.to(torch.float32) * scale_ref.unsqueeze(-2)
    else:
        raise ValueError(granularity)
    deq_m = metrics(deq, ref)
    if deq_m["qsnr_dB"] < 50.0 or deq_m["rel_L1"] > 1e-4 or deq_m["cos"] < 0.99999:
        raise AssertionError(
            f"{label} dequant mismatch: QSNR={deq_m['qsnr_dB']:.2f}, "
            f"relL1={deq_m['rel_L1']:.3e}, cos={deq_m['cos']:.6f}")


def host_quantize_per_tensor_fp8(x):
    descale = (x.abs().amax().clamp_min(1e-12) / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / descale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.contiguous(), descale


def host_quantize_per_tensor_fp8_sr(x, *, gen=None):
    """Per-tensor FP8 e4m3 SR quant, returning (xq_fp8, descale_scalar).

    This mirrors cuDNN-style current scaling: the scale/descale is externally
    managed for the whole tensor, not recomputed per row or per block by SDPA.
    """
    descale = (x.abs().amax().clamp_min(1e-12) / FP8_E4M3_MAX).to(torch.float32)
    y = x / descale
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
    return xq.contiguous(), descale


def host_quantize_with_descale_fp8_sr(x, descale, *, gen=None):
    y = x / descale
    abs_y = y.abs().clamp_min(2.0 ** -6)
    exp = torch.floor(torch.log2(abs_y))
    ulp = (2.0 ** (exp - 3.0)).clamp_min(2.0 ** -9)
    if gen is None:
        noise = (torch.rand_like(y) - 0.5) * ulp
    else:
        noise = (torch.rand(y.shape, generator=gen,
                            device=y.device, dtype=y.dtype) - 0.5) * ulp
    y_jit = (y + noise).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return y_jit.to(torch.float8_e4m3fn).contiguous()


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


def host_quantize_per_channel_fp8(x):
    amax = x.abs().amax(dim=-2, keepdim=True).clamp_min(1e-12)
    scale = (amax / FP8_E4M3_MAX).to(torch.float32)
    xq = (x / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return xq.contiguous(), scale.squeeze(-2).contiguous()


def broadcast_descale(descale, like_2d):
    return torch.full_like(like_2d, float(descale), dtype=torch.float32)


def estimate_dS_descale(Qq, Kq, Vq, dOq, sq, sk, sv, sdo_descale):
    """Compute the externally supplied scalar descale for dP/dS.

    A production FP8 manager would normally use the previous launch's reported
    amax. In this test harness we compute the current tensor amax directly so
    the kernel receives the same kind of scalar descale cuDNN SDPA would.
    """
    head_dim = Qq.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(head_dim)
    Q_eff = Qq.to(torch.float32) * sq.unsqueeze(-1)
    K_eff = Kq.to(torch.float32) * sk.unsqueeze(-1)
    V_eff = Vq.to(torch.float32) * sv.unsqueeze(-1)
    dO_eff = dOq.to(torch.float32) * sdo_descale
    P = torch.softmax((Q_eff @ K_eff.transpose(-2, -1)) * inv_sqrt_d, dim=-1)
    dP = dO_eff @ V_eff.transpose(-2, -1)
    rowsum = (P * dP).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum) * inv_sqrt_d
    return (dS.abs().amax().clamp_min(1e-12) / FP8_E4M3_MAX).to(torch.float32)


def prepare_kernel_args(Q, K, V, dO, smooth_k=True, smooth_v=True, *, sr_dO=True):
    K_mean = K.mean(dim=-2, keepdim=True) if smooth_k else torch.zeros_like(K[..., :1, :])
    V_mean = V.mean(dim=-2, keepdim=True) if smooth_v else torch.zeros_like(V[..., :1, :])
    K_s = K - K_mean
    V_s = V - V_mean

    Qq, sq = cuda_quantize_per_token(Q)
    Kq, sk = cuda_quantize_per_token(K_s)
    Vq_ch, sv_ch = cuda_quantize_per_channel(V_s)
    Vbf = (Vq_ch.to(torch.float32) * sv_ch.unsqueeze(-2)).to(torch.bfloat16).contiguous()
    vm = V_mean.squeeze(-2).to(torch.bfloat16).contiguous()

    Vq, sv = cuda_quantize_per_token(V_s)
    if sr_dO:
        dOq, sdo_descale = host_quantize_per_tensor_fp8_sr(dO)
    else:
        dOq, sdo_descale = host_quantize_per_tensor_fp8(dO)
    sdo_row = broadcast_descale(sdo_descale, sq)
    sdp_descale = estimate_dS_descale(Qq, Kq, Vq, dOq, sq, sk, sv, sdo_descale)
    sdp_row = broadcast_descale(sdp_descale, sq)

    Q_T = Q.transpose(-1, -2).contiguous()
    dO_T = dO.transpose(-1, -2).contiguous()
    Qq_t, sq_ch = host_quantize_per_channel_fp8_sr(Q_T)
    dOq_t = host_quantize_with_descale_fp8_sr(dO_T, sdo_descale)
    sdo_ch = broadcast_descale(sdo_descale, sq_ch)
    K_bf = K_s.to(torch.bfloat16).contiguous()
    dO_bf = (dOq.to(torch.float32) * sdo_descale).to(torch.bfloat16).contiguous()

    return {
        "K_s": K_s, "V_s": V_s,
        "Qq": Qq, "Kq": Kq, "Vq_ch": Vq_ch, "Vbf": Vbf,
        "sq": sq, "sk": sk, "sv_ch": sv_ch, "vm": vm,
        "Vq": Vq, "sv": sv, "dOq": dOq,
        "Qq_t": Qq_t, "dOq_t": dOq_t, "K_bf": K_bf, "dO_bf": dO_bf,
        "sdo_row": sdo_row, "sdp_row": sdp_row,
        "sq_ch": sq_ch, "sdo_ch": sdo_ch,
    }


def kernel_fwd_bwd(Q, K, V, dO, smooth_k=True, smooth_v=True,
                   *, sr_dO=True, fp8_dS_mode=0, return_quant=False):
    """Forward+Backward through the FP8 kernel (non-causal / bidirectional).

    Forward consumes Q, K as FP8 e4m3 + per-row scales.

    Backward consumes:
      * Q, K, V as FP8 e4m3 (per-token scales sq, sk, sv)
      * dO as FP8 e4m3 with a scalar externally managed descale
      * Q_t, dO_t  as FP8 e4m3 transposed copies with per-channel SR scales
        (sq_ch, sdo_ch) — fed to the dV / dK FP8 mma_ABt
      * K_bf bf16 SHADOW copy — fed to the dQ bf16 mma_AtB
        (FP8 mma_AtB unsupported in TK at this commit)
      * O (bf16) and dO (bf16) for the prep kernel that computes D = sum(O*dO)
      * L from the forward (fp32)
    """
    import _C
    p = prepare_kernel_args(Q, K, V, dO, smooth_k, smooth_v, sr_dO=sr_dO)

    o, l = _C.fp8_mha_forward(
        p["Qq"], p["Kq"], p["Vbf"],
        p["sq"].contiguous().to(torch.float32),
        p["sk"].contiguous().to(torch.float32),
        p["vm"],
    )

    qg, kg, vg = _C.fp8_mha_backward(
        p["Qq"], p["Kq"], p["Vq"], p["dOq"],
        p["Qq_t"], p["dOq_t"],
        p["K_bf"],
        o, p["dO_bf"],
        l,
        p["sq"].to(torch.float32).contiguous(),
        p["sk"].to(torch.float32).contiguous(),
        p["sv"].to(torch.float32).contiguous(),
        p["sdo_row"].to(torch.float32).contiguous(),
        p["sdp_row"].to(torch.float32).contiguous(),
        p["sq_ch"].to(torch.float32).contiguous(),
        p["sdo_ch"].to(torch.float32).contiguous(),
        fp8_dS_mode=fp8_dS_mode,
    )
    out = {
        "O": o.to(torch.float32), "L": l,
        "Q_fp8": p["Qq"], "K_fp8": p["Kq"], "V_fp8": p["Vq"], "dO_fp8": p["dOq"],
        "dQ": qg.to(torch.float32),
        "dK": kg.to(torch.float32),
        "dV": vg.to(torch.float32),
    }
    if return_quant:
        out.update({
            "sq": p["sq"], "sk": p["sk"], "V_ch_fp8": p["Vq_ch"], "sv_ch": p["sv_ch"],
            "V_token_fp8": p["Vq"], "sv_token": p["sv"],
        })
    return out


def profile_kernels(Q, K, V, dO, *, bench_iters):
    import _C
    B, H, N, D = Q.shape
    K_s = K - K.mean(dim=-2, keepdim=True)
    V_s = V - V.mean(dim=-2, keepdim=True)

    q_out = torch.empty_like(Q, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
    k_out = torch.empty_like(K_s, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)
    v_ch_out = torch.empty_like(V_s, dtype=torch.float8_e4m3fn)
    v_ch_scale = torch.empty((B, H, D), device=Q.device, dtype=torch.float32)

    token_bytes = Q.numel() * (4 + 1) + B * H * N * 4
    channel_bytes = V.numel() * (4 + 1) + B * H * D * 4
    q_ms = time_ms(lambda: cuda_quantize_per_token_out(Q, q_out, q_scale),
                   warmup=5, iters=bench_iters)
    k_ms = time_ms(lambda: cuda_quantize_per_token_out(K_s, k_out, k_scale),
                   warmup=5, iters=bench_iters)
    v_ms = time_ms(lambda: cuda_quantize_per_channel_out(V_s, v_ch_out, v_ch_scale),
                   warmup=5, iters=bench_iters)

    p = prepare_kernel_args(Q, K, V, dO, sr_dO=True)
    fwd = lambda: _C.fp8_mha_forward(
        p["Qq"], p["Kq"], p["Vbf"],
        p["sq"].contiguous().to(torch.float32),
        p["sk"].contiguous().to(torch.float32),
        p["vm"])
    o, l = fwd()
    fwd_ms = time_ms(fwd, warmup=5, iters=bench_iters)

    bwd = lambda: _C.fp8_mha_backward(
        p["Qq"], p["Kq"], p["Vq"], p["dOq"],
        p["Qq_t"], p["dOq_t"],
        p["K_bf"],
        o, p["dO_bf"],
        l,
        p["sq"].to(torch.float32).contiguous(),
        p["sk"].to(torch.float32).contiguous(),
        p["sv"].to(torch.float32).contiguous(),
        p["sdo_row"].to(torch.float32).contiguous(),
        p["sdp_row"].to(torch.float32).contiguous(),
        p["sq_ch"].to(torch.float32).contiguous(),
        p["sdo_ch"].to(torch.float32).contiguous(),
        fp8_dS_mode=2)
    bwd_ms = time_ms(bwd, warmup=5, iters=bench_iters)

    print(f"  {'quant Q token':<28} {q_ms:7.4f} ms  {gbps(token_bytes, q_ms):8.1f} GB/s")
    print(f"  {'quant K token':<28} {k_ms:7.4f} ms  {gbps(token_bytes, k_ms):8.1f} GB/s")
    print(f"  {'quant V channel':<28} {v_ms:7.4f} ms  {gbps(channel_bytes, v_ms):8.1f} GB/s")
    print(f"  {'fp8 attention fwd':<28} {fwd_ms:7.3f} ms  "
          f"{attention_fwd_tflops(B, H, N, D, fwd_ms):7.2f} TFLOP/s")
    print(f"  {'fp8 attention bwd':<28} {bwd_ms:7.3f} ms  "
          f"{attention_bwd_tflops(B, H, N, D, bwd_ms):7.2f} TFLOP/s")


def run_one(B, H, N, D, seed, *, bench_iters):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    print(f"\n[B={B} H={H} N={N} D={D} seed={seed}]")

    # Ground truth (non-causal): compare our algebraic reference against
    # PyTorch's builtin scaled_dot_product_attention autograd path first.
    O_sdpa, dQ_ref, dK_ref, dV_ref = sdpa_backward(Q, K, V, dO, causal=False)
    dQ_alg, dK_alg, dV_alg = reference_backward(Q, K, V, dO, causal=False)
    alg_dQm = metrics(dQ_alg, dQ_ref)
    alg_dKm = metrics(dK_alg, dK_ref)
    alg_dVm = metrics(dV_alg, dV_ref)
    check_metrics("manual fp32 reference vs torch SDPA",
                  alg_dQm, alg_dKm, alg_dVm,
                  min_qsnr=55.0, max_rel_l1=1e-3, min_cos=0.99999)
    print(" ", fmt("manual ref vs torch SDPA",
                   alg_dQm, alg_dKm, alg_dVm))

    # Recipe (SR-dO + FP8 SR-dS in-kernel)
    res_recipe = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=2, return_quant=True)
    Qq_ref, sq_ref = host_quantize_per_row_fp8(Q)
    K_mean = K.mean(dim=-2, keepdim=True)
    Kq_ref, sk_ref = host_quantize_per_row_fp8(K - K_mean)
    V_mean = V.mean(dim=-2, keepdim=True)
    Vq_ch_ref, sv_ch_ref = host_quantize_per_channel_fp8(V - V_mean)
    check_quant_kernel("Q token quant CUDA", res_recipe["Q_fp8"], res_recipe["sq"],
                       Qq_ref, sq_ref, granularity="token")
    check_quant_kernel("K token quant CUDA", res_recipe["K_fp8"], res_recipe["sk"],
                       Kq_ref, sk_ref, granularity="token")
    check_quant_kernel("V channel quant CUDA", res_recipe["V_ch_fp8"], res_recipe["sv_ch"],
                       Vq_ch_ref, sv_ch_ref, granularity="channel")
    # SR-dO + FP8 RTNE-dS in-kernel (no SR)
    res_rtne = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=1)
    # SR-dO + bf16 dS (current baseline before the dS-quant work)
    res_sr_dO = kernel_fwd_bwd(Q, K, V, dO, sr_dO=True, fp8_dS_mode=0)
    # RTNE dO with bf16 dS path.
    res_no    = kernel_fwd_bwd(Q, K, V, dO, sr_dO=False, fp8_dS_mode=0)

    recipe_m = (
        metrics(res_recipe["dQ"], dQ_ref),
        metrics(res_recipe["dK"], dK_ref),
        metrics(res_recipe["dV"], dV_ref),
    )
    rtne_m = (
        metrics(res_rtne["dQ"], dQ_ref),
        metrics(res_rtne["dK"], dK_ref),
        metrics(res_rtne["dV"], dV_ref),
    )
    sr_dO_m = (
        metrics(res_sr_dO["dQ"], dQ_ref),
        metrics(res_sr_dO["dK"], dK_ref),
        metrics(res_sr_dO["dV"], dV_ref),
    )
    rtne_dO_m = (
        metrics(res_no["dQ"], dQ_ref),
        metrics(res_no["dK"], dK_ref),
        metrics(res_no["dV"], dV_ref),
    )
    check_metrics("kernel recipe SR-dO+SR-dS vs torch SDPA",
                  *recipe_m, min_qsnr=18.0, max_rel_l1=0.13, min_cos=0.985)
    check_metrics("kernel SR-dO+RTNE-dS vs torch SDPA",
                  *rtne_m, min_qsnr=18.0, max_rel_l1=0.13, min_cos=0.985)
    check_metrics("kernel SR-dO+bf16-dS vs torch SDPA",
                  *sr_dO_m, min_qsnr=18.0, max_rel_l1=0.13, min_cos=0.985)
    check_metrics("kernel RTNE-dO+bf16-dS vs torch SDPA",
                  *rtne_dO_m, min_qsnr=18.0, max_rel_l1=0.13, min_cos=0.985)

    o_m = metrics(res_recipe["O"], O_sdpa)
    if o_m["qsnr_dB"] < 20.0 or o_m["rel_L1"] > 0.1 or o_m["cos"] < 0.99:
        raise AssertionError(
            "kernel forward O vs torch SDPA failed: "
            f"QSNR={o_m['qsnr_dB']:.2f}, relL1={o_m['rel_L1']:.3e}, cos={o_m['cos']:.6f}")
    print(f"  {'kernel O vs torch SDPA':<28} "
          f"QSNR={o_m['qsnr_dB']:5.2f} relL1={o_m['rel_L1']:.2e} cos={o_m['cos']:.5f}")

    print(" ", fmt("kernel recipe SR-dO+SR-dS",
                   *recipe_m))
    print(" ", fmt("kernel SR-dO + RTNE-dS  ",
                   *rtne_m))
    print(" ", fmt("kernel SR-dO + bf16 dS  ",
                   *sr_dO_m))
    print(" ", fmt("kernel RTNE-dO + bf16 dS",
                   *rtne_dO_m))

    if bench_iters > 0:
        profile_kernels(Q, K, V, dO, bench_iters=bench_iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench-iters", type=int, default=10)
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
        run_one(*cfg, bench_iters=args.bench_iters)


if __name__ == "__main__":
    main()
