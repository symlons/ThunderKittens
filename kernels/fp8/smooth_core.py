import math
from pathlib import Path

import torch
import torch.nn.functional as F


device = "cuda"
dtype = torch.float32
fp8_e4m3 = torch.float8_e4m3fn
fp8_e5m2 = torch.float8_e5m2

N, D = 256, 128
bq, bkv, cw = 128, 64, 4

fp8_e4m3_max = 448.0
fp8_e5m2_max = 57344.0
int4_max = 7.0
int8_max = 127.0

# Q/K dtypes used for QK^T
qk_modes = ["int4", "int8", "fp8_e4m3", "fp8_e5m2"]

# Granularity levels for Q/K quantization
qk_granularities = ["per_tensor", "per_block", "per_token", "per_thread"]

# Smoothing combinations
smooth_cases = [
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


def smooth_label(smooth_q, smooth_k, smooth_v):
    enabled = []
    if smooth_q:
        enabled.append("Q")
    if smooth_k:
        enabled.append("K")
    if smooth_v:
        enabled.append("V")
    return "+".join(enabled) if enabled else "none"


smooth_labels = [smooth_label(*case) for case in smooth_cases]


def rel_l1(a, b):
    return (a - b).abs().sum() / b.abs().sum().clamp_min(1e-12)


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0)


def metrics(x, ref):
    return rel_l1(x, ref).item(), cos(x, ref).item()


def qsnr_db(x, ref):
    sig = (ref.float() ** 2).sum().clamp_min(1e-30)
    noise = ((x - ref).float() ** 2).sum().clamp_min(1e-30)
    return float(10.0 * torch.log10(sig / noise))


def row_center(x):
    return x - x.mean(dim=-1, keepdim=True)


def row_max_center(x):
    return x - x.max(dim=-1, keepdim=True).values


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------


def _mode_max(mode):
    if mode == "int4":
        return int4_max
    if mode == "int8":
        return int8_max
    if mode == "fp8_e4m3":
        return fp8_e4m3_max
    if mode == "fp8_e5m2":
        return fp8_e5m2_max
    raise ValueError(f"unknown quantization mode: {mode}")


def _to_fp(mode):
    if mode == "fp8_e4m3":
        return fp8_e4m3
    if mode == "fp8_e5m2":
        return fp8_e5m2
    raise ValueError(mode)


def _scale_for(x, mode):
    return x.abs().amax().clamp_min(1e-12) / _mode_max(mode)


def _quant_dequant(x, scale, mode):
    if mode in {"int4", "int8"}:
        m = _mode_max(mode)
        return torch.round(x / scale).clamp(-m, m) * scale
    if mode in {"fp8_e4m3", "fp8_e5m2"}:
        m = _mode_max(mode)
        return (x / scale).clamp(-m, m).to(_to_fp(mode)).to(torch.float32) * scale
    raise ValueError(mode)


def qdq_block(x, mode):
    scale = _scale_for(x, mode)
    return _quant_dequant(x, scale, mode)


def quantized_values(x, mode):
    """Return the integer/fp8 representation values (post-rounding, pre-scale)."""
    scale = _scale_for(x, mode)
    if mode in {"int4", "int8"}:
        m = _mode_max(mode)
        return torch.round(x / scale).clamp(-m, m)
    m = _mode_max(mode)
    return (x / scale).clamp(-m, m).to(_to_fp(mode)).to(torch.float32)


# ---------------------------------------------------------------------------
# Granularity-aware fake-quant. All return a dequantized float32 tensor of the
# same shape as the input, simulating per-X quantization of the rows of Q/K.
# ---------------------------------------------------------------------------


def fake_quant_q(x, mode, granularity, warp_count=cw):
    """Quantize a Q-block (tokens x D) with the chosen granularity.

    For Q the row dimension is the token axis, so per-token = per-row.
    Per-thread groups rows mod 8 inside each warp segment.
    """
    if granularity == "per_tensor":
        return qdq_block(x, mode)

    if granularity == "per_block":
        # The whole block shares one scale (same as per_tensor here, since the
        # caller passes a single block at a time).
        return qdq_block(x, mode)

    if granularity == "per_token":
        y = torch.empty_like(x, dtype=torch.float32)
        for r in range(x.shape[0]):
            y[r] = qdq_block(x[r:r + 1], mode)
        return y

    if granularity == "per_thread":
        return _per_thread_q(x, mode, warp_count)

    raise ValueError(f"unknown granularity: {granularity}")


def fake_quant_k(x, mode, granularity):
    if granularity == "per_tensor":
        return qdq_block(x, mode)

    if granularity == "per_block":
        return qdq_block(x, mode)

    if granularity == "per_token":
        y = torch.empty_like(x, dtype=torch.float32)
        for r in range(x.shape[0]):
            y[r] = qdq_block(x[r:r + 1], mode)
        return y

    if granularity == "per_thread":
        return _per_thread_k(x, mode)

    raise ValueError(f"unknown granularity: {granularity}")


def _per_thread_q(x, mode, warp_count):
    n = x.shape[0]
    y = torch.empty_like(x, dtype=torch.float32)
    seg = math.ceil(n / warp_count)
    for w in range(warp_count):
        base = w * seg
        end = min(base + seg, n)
        if base >= end:
            continue
        for i in range(8):
            rows = base + torch.arange(i, end - base, 8, device=x.device)
            if rows.numel() > 0:
                y[rows] = qdq_block(x[rows], mode)
    return y


def _per_thread_k(x, mode):
    n = x.shape[0]
    y = torch.empty_like(x, dtype=torch.float32)
    for i in range(4):
        idx = []
        for k in range(math.ceil(n / 8)):
            r0 = 8 * k + 2 * i
            r1 = r0 + 1
            if r0 < n:
                idx.append(r0)
            if r1 < n:
                idx.append(r1)
        rows = torch.tensor(idx, device=x.device)
        if rows.numel() > 0:
            y[rows] = qdq_block(x[rows], mode)
    return y


# ---------------------------------------------------------------------------
# P/V FP8 quantization (always FP8 in SageAttention2)
# ---------------------------------------------------------------------------


def fp8_quantize_p_static(p, mode="fp8_e4m3"):
    m = _mode_max(mode)
    return (p * m).clamp(0, m).to(_to_fp(mode)).to(torch.float32)


def fp8_quantize_v_per_channel(v, mode="fp8_e4m3"):
    m = _mode_max(mode)
    scale = v.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / m
    vhat = (v / scale).clamp(-m, m).to(_to_fp(mode)).to(torch.float32)
    return vhat, scale


def pv_fp8_block(ptilde, vhat_block, v_scale, mode="fp8_e4m3"):
    phat = fp8_quantize_p_static(ptilde, mode)
    return (phat @ vhat_block) * (v_scale / _mode_max(mode))


# ---------------------------------------------------------------------------
# Smoothing (mean subtraction + SmoothQuant variant)
# ---------------------------------------------------------------------------


def smoothquant_scale(Q, K, alpha=0.5, eps=1e-6):
    """Per-channel SmoothQuant scale s such that Q' = Q / s, K' = K * s."""
    q_max = Q.abs().amax(dim=0).clamp_min(eps)
    k_max = K.abs().amax(dim=0).clamp_min(eps)
    s = q_max.pow(alpha) / k_max.pow(1.0 - alpha)
    return s.clamp_min(eps)


def apply_smoothquant(Q, K, alpha=0.5):
    s = smoothquant_scale(Q, K, alpha=alpha)
    return Q / s, K * s


# ---------------------------------------------------------------------------
# Reference attention
# ---------------------------------------------------------------------------


def reference_attention(Q, K, V):
    head_dim = Q.shape[-1]
    S = (Q @ K.T) / math.sqrt(head_dim)
    P = F.softmax(S, dim=-1)
    O = P @ V
    return S, P, O


# ---------------------------------------------------------------------------
# Quantized scores / attention forward
# ---------------------------------------------------------------------------


def quantized_scores(
    Q,
    K,
    qk_mode="int4",
    smooth_q=False,
    smooth_k=False,
    granularity="per_thread",
    block_q=bq,
    block_k=bkv,
    warp_count=cw,
    smoothquant_alpha=None,
    exact=False,
):
    n, head_dim = Q.shape

    if smoothquant_alpha is not None:
        Q_eff, K_eff = apply_smoothquant(Q, K, alpha=smoothquant_alpha)
    else:
        Q_eff, K_eff = Q, K

    k_mean = (
        K_eff.mean(dim=0, keepdim=True)
        if smooth_k
        else torch.zeros(1, head_dim, device=K.device, dtype=K.dtype)
    )
    Kg = K_eff - k_mean

    S = torch.empty(n, K.shape[0], device=Q.device, dtype=Q.dtype)

    for qs in range(0, n, block_q):
        qe = min(qs + block_q, n)
        Qi = Q_eff[qs:qe]
        q_mean = (
            Qi.mean(dim=0, keepdim=True)
            if smooth_q
            else torch.zeros(1, head_dim, device=Q.device, dtype=Q.dtype)
        )
        Qg = Qi - q_mean
        Qq_dq = fake_quant_q(Qg, qk_mode, granularity, warp_count)

        for ks in range(0, K.shape[0], block_k):
            ke = min(ks + block_k, K.shape[0])
            Kj = Kg[ks:ke]
            Kq_dq = fake_quant_k(Kj, qk_mode, granularity)
            Sij = Qq_dq @ Kq_dq.T

            if smooth_q:
                Sij = Sij + q_mean @ Kj.T
            if exact and smooth_k:
                Sij = Sij + Qi @ k_mean.T

            S[qs:qe, ks:ke] = Sij / math.sqrt(head_dim)

    return S


def quantized_attention(
    Q,
    K,
    V,
    qk_mode="int4",
    smooth_q=False,
    smooth_k=False,
    smooth_v=False,
    granularity="per_thread",
    pv_mode="fp8_e4m3",
    block_q=bq,
    block_k=bkv,
    warp_count=cw,
    smoothquant_alpha=None,
):
    n, head_dim = Q.shape

    if smoothquant_alpha is not None:
        Q_eff, K_eff = apply_smoothquant(Q, K, alpha=smoothquant_alpha)
    else:
        Q_eff, K_eff = Q, K

    k_mean = (
        K_eff.mean(dim=0, keepdim=True)
        if smooth_k
        else torch.zeros(1, head_dim, device=K.device, dtype=K.dtype)
    )
    Kg = K_eff - k_mean

    if smooth_v:
        v_mean = V.mean(dim=0, keepdim=True)
        V_for_pv = V - v_mean
    else:
        v_mean = None
        V_for_pv = V

    vhat, v_scale = fp8_quantize_v_per_channel(V_for_pv, pv_mode)
    O = torch.empty_like(Q)

    for qs in range(0, n, block_q):
        qe = min(qs + block_q, n)
        q_rows = qe - qs
        Qi = Q_eff[qs:qe]
        q_mean = (
            Qi.mean(dim=0, keepdim=True)
            if smooth_q
            else torch.zeros(1, head_dim, device=Q.device, dtype=Q.dtype)
        )
        Qg = Qi - q_mean
        Qq_dq = fake_quant_q(Qg, qk_mode, granularity, warp_count)

        m = torch.full((q_rows,), -float("inf"), device=Q.device, dtype=Q.dtype)
        l = torch.zeros((q_rows,), device=Q.device, dtype=Q.dtype)
        acc = torch.zeros((q_rows, head_dim), device=Q.device, dtype=Q.dtype)

        for ks in range(0, K.shape[0], block_k):
            ke = min(ks + block_k, K.shape[0])
            Kj = Kg[ks:ke]
            Kq_dq = fake_quant_k(Kj, qk_mode, granularity)

            Sij = Qq_dq @ Kq_dq.T
            if smooth_q:
                Sij = Sij + q_mean @ Kj.T
            Sij = Sij / math.sqrt(head_dim)

            m_new = torch.maximum(m, Sij.max(dim=-1).values)
            alpha = torch.exp(m - m_new)
            ptilde = torch.exp(Sij - m_new[:, None])

            R = pv_fp8_block(ptilde, vhat[ks:ke], v_scale, pv_mode)
            acc = alpha[:, None] * acc + R
            l = alpha * l + ptilde.sum(dim=-1)
            m = m_new

        Oi = acc / l[:, None]
        if v_mean is not None:
            Oi = Oi + v_mean
        O[qs:qe] = Oi

    return O


# ---------------------------------------------------------------------------
# Backward pass: reference + quantized
# ---------------------------------------------------------------------------


def reference_backward(Q, K, V, dO):
    """Standard attention backward in fp32. Returns dQ, dK, dV."""
    head_dim = Q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    S = (Q @ K.T) * scale
    P = F.softmax(S, dim=-1)

    dV = P.T @ dO
    dP = dO @ V.T
    # dS_ij = P_ij * (dP_ij - sum_k(P_ik * dP_ik))
    rowsum = (P * dP).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)
    dQ = (dS @ K) * scale
    dK = (dS.T @ Q) * scale
    return dQ, dK, dV


def _row_quant(x, mode):
    """Per-row fake-quant: each row gets its own scale."""
    y = torch.empty_like(x, dtype=torch.float32)
    for r in range(x.shape[0]):
        y[r] = qdq_block(x[r:r + 1], mode)
    return y


def quantized_backward(
    Q,
    K,
    V,
    dO,
    qk_mode="int4",
    smooth_q=False,
    smooth_k=False,
    smooth_v=False,
    granularity="per_thread",
    pv_mode="fp8_e4m3",
    grad_mode="fp8_e4m3",
    smoothquant_alpha=None,
    quantize_grads=True,
):
    """Quantized attention backward.

    Quantization recipe (mirrors the forward):
      * Q,K go through INT4/INT8/FP8 with the chosen granularity for QK^T.
      * P,V are FP8 (channel/static) for both forward PV and backward P^T @ dO.
      * dO and dS are quantized per-row in `grad_mode` (FP8 by default).
      * For dQ = dS @ K and dK = dS^T @ Q we reuse the *same* fake-quanted
        K and Q the forward kernel saw.

    Returns (dQ, dK, dV) in fp32.
    """
    head_dim = Q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # ----- replay the quantized forward to recover the kernel's P -------------
    if smoothquant_alpha is not None:
        Q_eff, K_eff = apply_smoothquant(Q, K, alpha=smoothquant_alpha)
    else:
        Q_eff, K_eff = Q, K

    k_mean = (
        K_eff.mean(dim=0, keepdim=True) if smooth_k else torch.zeros_like(K_eff[:1])
    )
    Kg = K_eff - k_mean

    Q_block_means = []
    Qg_blocks = []
    for qs in range(0, Q.shape[0], bq):
        qe = min(qs + bq, Q.shape[0])
        Qi = Q_eff[qs:qe]
        qm = Qi.mean(dim=0, keepdim=True) if smooth_q else torch.zeros_like(Qi[:1])
        Q_block_means.append(qm)
        Qg_blocks.append(Qi - qm)

    Qg = torch.cat(Qg_blocks, dim=0)
    # The forward fake-quants Qg and Kg with the chosen granularity. We reuse
    # those same dequantized matrices in the backward gemms so dQ/dK see exactly
    # what the kernel sees.
    Qg_q = fake_quant_q(Qg, qk_mode, granularity)
    Kg_q = fake_quant_k(Kg, qk_mode, granularity)

    # K_eff_q reconstructs the kernel-visible K = Kg_q + k_mean (mean is exact).
    K_eff_q = Kg_q + k_mean
    # Q_eff_q reconstructs per-block kernel-visible Q.
    Q_eff_q = Qg_q.clone()
    for i, qm in enumerate(Q_block_means):
        qs = i * bq
        qe = min(qs + bq, Q.shape[0])
        Q_eff_q[qs:qe] = Q_eff_q[qs:qe] + qm

    S = Qg_q @ Kg_q.T
    if smooth_q:
        for i, qm in enumerate(Q_block_means):
            qs = i * bq
            qe = min(qs + bq, Q.shape[0])
            S[qs:qe] = S[qs:qe] + qm @ Kg.T
    S = S * scale
    P = F.softmax(S, dim=-1)

    # ----- V handling (matches forward) --------------------------------------
    if smooth_v:
        v_mean = V.mean(dim=0, keepdim=True)
        V_centered = V - v_mean
    else:
        v_mean = None
        V_centered = V

    if quantize_grads:
        vhat, v_scale = fp8_quantize_v_per_channel(V_centered, pv_mode)
        V_q = vhat * v_scale
        # dO is per-row quantized (kernels do this so each Oi has its own scale)
        dO_q = _row_quant(dO, grad_mode)
        # P is recovered with the static (1/max) scale used in the forward
        P_q = fp8_quantize_p_static(P, pv_mode) / _mode_max(pv_mode)
    else:
        V_q = V_centered
        dO_q = dO
        P_q = P

    # dV = P^T @ dO  (FP8 P x FP8 dO)
    dV = P_q.T @ dO_q

    # dP = dO @ V^T  (FP8 dO x FP8 V)
    dP = dO_q @ V_q.T

    rowsum = (P * dP).sum(dim=-1, keepdim=True)
    dS = P * (dP - rowsum)

    if quantize_grads:
        # dS rows have wildly different magnitudes (sparse near argmax). Per-row
        # quantization is essential, otherwise per-tensor zeros most rows.
        dS_q = _row_quant(dS, grad_mode)
    else:
        dS_q = dS

    # dQ_eff = dS @ K_eff   (use kernel-visible K)
    dQ_eff = (dS_q @ K_eff_q) * scale
    # dK_eff = dS^T @ Q_eff (use kernel-visible Q)
    dK_eff = (dS_q.T @ Q_eff_q) * scale

    if smoothquant_alpha is not None:
        s = smoothquant_scale(Q, K, alpha=smoothquant_alpha)
        dQ = dQ_eff / s
        dK = dK_eff * s
    else:
        dQ = dQ_eff
        dK = dK_eff

    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Inputs (synthetic and real)
# ---------------------------------------------------------------------------


def make_inputs(seed=0, n=N, d=D, source="synthetic", path=None, head_index=0):
    if source == "synthetic":
        torch.manual_seed(seed)
        shared_q = torch.randn(1, d, device=device, dtype=dtype) * 8.0
        shared_k = torch.randn(1, d, device=device, dtype=dtype) * 8.0
        shared_v = torch.randn(1, d, device=device, dtype=dtype) * 8.0
        Q = shared_q + torch.randn(n, d, device=device, dtype=dtype) * 0.2
        K = shared_k + torch.randn(n, d, device=device, dtype=dtype) * 0.2
        V = shared_v + torch.randn(n, d, device=device, dtype=dtype) * 2.0
        return Q, K, V

    if source == "cogvideox":
        if path is None:
            path = Path(__file__).parent / "captures" / "cogvideox.pt"
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CogVideoX capture not found at {path}. "
                f"Run: python3 capture_cogvideox.py"
            )
        bundle = torch.load(path, map_location=device, weights_only=False)
        # bundle: {"Q": [batch, heads, n, d] or list, "K": ..., "V": ...}
        Q_full = bundle["Q"]
        K_full = bundle["K"]
        V_full = bundle["V"]
        if isinstance(Q_full, list):
            Q_full = Q_full[0]
            K_full = K_full[0]
            V_full = V_full[0]
        # take a single head
        if Q_full.dim() == 4:
            Q_full = Q_full[0, head_index]
            K_full = K_full[0, head_index]
            V_full = V_full[0, head_index]
        elif Q_full.dim() == 3:
            Q_full = Q_full[head_index]
            K_full = K_full[head_index]
            V_full = V_full[head_index]
        # truncate to requested N if needed
        if n is not None and Q_full.shape[0] > n:
            Q_full = Q_full[:n]
            K_full = K_full[:n]
            V_full = V_full[:n]
        return (
            Q_full.to(device=device, dtype=dtype).contiguous(),
            K_full.to(device=device, dtype=dtype).contiguous(),
            V_full.to(device=device, dtype=dtype).contiguous(),
        )

    raise ValueError(f"unknown source: {source}")


# ---------------------------------------------------------------------------
# Convenience wrappers used by smooth.py
# ---------------------------------------------------------------------------


def raw_int4_scores(Q, K):
    return quantized_scores(Q, K, "int4", False, False)


def raw_int4_fp8_attention(Q, K, V):
    return quantized_attention(Q, K, V, "int4", False, False, False)


def sa2_scores(Q, K, exact=False):
    return quantized_scores(Q, K, "int4", True, True, exact=exact)


def sa2_attention(Q, K, V, smooth_v=False):
    return quantized_attention(Q, K, V, "int4", True, True, smooth_v)


# ---------------------------------------------------------------------------
# Ablation tables
# ---------------------------------------------------------------------------


def _row(qk_mode, smooth_str, granularity, S_ref, P_ref, O_ref, S, P, O, extra=None):
    row = {
        "qk_mode": qk_mode,
        "smooth": smooth_str,
        "granularity": granularity,
        "score_l1": rel_l1(row_max_center(S), row_max_center(S_ref)).item(),
        "softmax_l1": rel_l1(P, P_ref).item(),
        "out_l1": rel_l1(O, O_ref).item(),
        "out_cos": cos(O, O_ref).item(),
        "out_qsnr": qsnr_db(O, O_ref),
    }
    if extra:
        row.update(extra)
    return row


def build_ablation_rows(Q, K, V, S_ref, P_ref, O_ref, granularity="per_thread"):
    rows = []
    for qk_mode in qk_modes:
        for smooth_q, smooth_k, smooth_v in smooth_cases:
            S = quantized_scores(Q, K, qk_mode, smooth_q, smooth_k, granularity=granularity)
            P = F.softmax(S, dim=-1)
            O = quantized_attention(
                Q, K, V, qk_mode, smooth_q, smooth_k, smooth_v, granularity=granularity
            )
            rows.append(_row(
                qk_mode, smooth_label(smooth_q, smooth_k, smooth_v), granularity,
                S_ref, P_ref, O_ref, S, P, O,
            ))
    return rows


def build_granularity_rows(Q, K, V, S_ref, P_ref, O_ref, smooth=(True, True, False)):
    """Sweep quantization granularity for each Q/K dtype, with one smoothing setup."""
    sq, sk, sv = smooth
    rows = []
    for qk_mode in qk_modes:
        for granularity in qk_granularities:
            S = quantized_scores(Q, K, qk_mode, sq, sk, granularity=granularity)
            P = F.softmax(S, dim=-1)
            O = quantized_attention(
                Q, K, V, qk_mode, sq, sk, sv, granularity=granularity
            )
            rows.append(_row(
                qk_mode, smooth_label(sq, sk, sv), granularity,
                S_ref, P_ref, O_ref, S, P, O,
            ))
    return rows


def build_smoothquant_rows(Q, K, V, S_ref, P_ref, O_ref, qk_mode="int4",
                            granularity="per_thread", alphas=None):
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = []
    for alpha in alphas:
        S = quantized_scores(Q, K, qk_mode, False, False,
                             granularity=granularity, smoothquant_alpha=alpha)
        P = F.softmax(S, dim=-1)
        O = quantized_attention(Q, K, V, qk_mode, False, False, False,
                                granularity=granularity, smoothquant_alpha=alpha)
        rows.append(_row(
            qk_mode, f"smoothquant a={alpha:.2f}", granularity,
            S_ref, P_ref, O_ref, S, P, O,
            extra={"alpha": alpha},
        ))
    return rows


def build_backward_rows(Q, K, V, dO, granularity="per_thread"):
    """Compare quantized backward against fp32 reference."""
    dQ_ref, dK_ref, dV_ref = reference_backward(Q, K, V, dO)
    rows = []
    for qk_mode in qk_modes:
        for smooth_q, smooth_k, smooth_v in smooth_cases:
            dQ, dK, dV = quantized_backward(
                Q, K, V, dO, qk_mode=qk_mode,
                smooth_q=smooth_q, smooth_k=smooth_k, smooth_v=smooth_v,
                granularity=granularity,
            )
            rows.append({
                "qk_mode": qk_mode,
                "smooth": smooth_label(smooth_q, smooth_k, smooth_v),
                "granularity": granularity,
                "dQ_l1": rel_l1(dQ, dQ_ref).item(),
                "dK_l1": rel_l1(dK, dK_ref).item(),
                "dV_l1": rel_l1(dV, dV_ref).item(),
                "dQ_qsnr": qsnr_db(dQ, dQ_ref),
                "dK_qsnr": qsnr_db(dK, dK_ref),
                "dV_qsnr": qsnr_db(dV, dV_ref),
            })
    return rows
