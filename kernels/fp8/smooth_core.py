import math

import torch
import torch.nn.functional as F


device = "cuda"
dtype = torch.float32
fp8_dtype = torch.float8_e4m3fn

N, D = 256, 128
bq, bkv, cw = 128, 64, 4

fp8_max = 448.0
int4_max = 7.0
int8_max = 127.0

qk_modes = ["int4", "int8", "fp8_e4m3"]
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


def row_center(x):
    return x - x.mean(dim=-1, keepdim=True)


def row_max_center(x):
    return x - x.max(dim=-1, keepdim=True).values


def qdq_block(x, mode):
    if mode == "int4":
        scale = x.abs().max().clamp_min(1e-12) / int4_max
        q = torch.round(x / scale).clamp(-int4_max, int4_max)
        return q * scale

    if mode == "int8":
        scale = x.abs().max().clamp_min(1e-12) / int8_max
        q = torch.round(x / scale).clamp(-int8_max, int8_max)
        return q * scale

    if mode == "fp8_e4m3":
        scale = x.abs().max().clamp_min(1e-12) / fp8_max
        q = (x / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype).to(torch.float32)
        return q * scale

    raise ValueError(f"unknown quantization mode: {mode}")


def quantized_values(x, mode):
    if mode == "int4":
        scale = x.abs().max().clamp_min(1e-12) / int4_max
        return torch.round(x / scale).clamp(-int4_max, int4_max)

    if mode == "int8":
        scale = x.abs().max().clamp_min(1e-12) / int8_max
        return torch.round(x / scale).clamp(-int8_max, int8_max)

    if mode == "fp8_e4m3":
        scale = x.abs().max().clamp_min(1e-12) / fp8_max
        return (x / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype).to(torch.float32)

    raise ValueError(f"unknown quantization mode: {mode}")


def qdq_q_per_thread(x, warp_count, mode):
    n, _ = x.shape
    y = torch.empty_like(x)
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


def qdq_k_per_thread(x, mode):
    n, _ = x.shape
    y = torch.empty_like(x)

    for i in range(4):
        idx = []
        for k in range(math.ceil(n / 8)):
            r0 = 8 * k + 2 * i
            r1 = 8 * k + 2 * i + 1
            if r0 < n:
                idx.append(r0)
            if r1 < n:
                idx.append(r1)

        rows = torch.tensor(idx, device=x.device)
        if rows.numel() > 0:
            y[rows] = qdq_block(x[rows], mode)

    return y


def fp8_quantize_p_static(p):
    return (p * fp8_max).clamp(0, fp8_max).to(fp8_dtype).to(torch.float32)


def fp8_quantize_v_per_channel(v):
    scale = v.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / fp8_max
    vhat = (v / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return vhat, scale


def pv_fp8_block(ptilde, vhat_block, v_scale):
    phat = fp8_quantize_p_static(ptilde)
    return (phat @ vhat_block.to(torch.float32)) * (v_scale / fp8_max)


def reference_attention(Q, K, V):
    head_dim = Q.shape[-1]
    S = (Q @ K.T) / math.sqrt(head_dim)
    P = F.softmax(S, dim=-1)
    O = P @ V
    return S, P, O


def quantized_scores(
    Q,
    K,
    qk_mode="int4",
    smooth_q=False,
    smooth_k=False,
    block_q=bq,
    block_k=bkv,
    warp_count=cw,
    exact=False,
):
    n, head_dim = Q.shape
    S = torch.empty(n, K.shape[0], device=Q.device, dtype=Q.dtype)
    k_mean = K.mean(dim=0, keepdim=True) if smooth_k else torch.zeros(1, head_dim, device=K.device, dtype=K.dtype)
    Kg = K - k_mean

    for qs in range(0, n, block_q):
        qe = min(qs + block_q, n)
        Qi = Q[qs:qe]
        q_mean = Qi.mean(dim=0, keepdim=True) if smooth_q else torch.zeros(1, head_dim, device=Q.device, dtype=Q.dtype)
        Qg = Qi - q_mean
        Qdq = qdq_q_per_thread(Qg, warp_count, qk_mode)

        for ks in range(0, K.shape[0], block_k):
            ke = min(ks + block_k, K.shape[0])
            Kj = Kg[ks:ke]
            Kdq = qdq_k_per_thread(Kj, qk_mode)
            Sij = Qdq @ Kdq.T

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
    block_q=bq,
    block_k=bkv,
    warp_count=cw,
):
    n, head_dim = Q.shape
    k_mean = K.mean(dim=0, keepdim=True) if smooth_k else torch.zeros(1, head_dim, device=K.device, dtype=K.dtype)
    Kg = K - k_mean

    if smooth_v:
        v_mean = V.mean(dim=0, keepdim=True)
        V_for_pv = V - v_mean
    else:
        v_mean = None
        V_for_pv = V

    vhat, v_scale = fp8_quantize_v_per_channel(V_for_pv)
    O = torch.empty_like(Q)

    for qs in range(0, n, block_q):
        qe = min(qs + block_q, n)
        q_rows = qe - qs
        Qi = Q[qs:qe]
        q_mean = Qi.mean(dim=0, keepdim=True) if smooth_q else torch.zeros(1, head_dim, device=Q.device, dtype=Q.dtype)
        Qg = Qi - q_mean
        Qdq = qdq_q_per_thread(Qg, warp_count, qk_mode)

        m = torch.full((q_rows,), -float("inf"), device=Q.device, dtype=Q.dtype)
        l = torch.zeros((q_rows,), device=Q.device, dtype=Q.dtype)
        acc = torch.zeros((q_rows, head_dim), device=Q.device, dtype=Q.dtype)

        for ks in range(0, K.shape[0], block_k):
            ke = min(ks + block_k, K.shape[0])
            Kj = Kg[ks:ke]
            Kdq = qdq_k_per_thread(Kj, qk_mode)

            Sij = Qdq @ Kdq.T
            if smooth_q:
                Sij = Sij + q_mean @ Kj.T
            Sij = Sij / math.sqrt(head_dim)

            m_new = torch.maximum(m, Sij.max(dim=-1).values)
            alpha = torch.exp(m - m_new)
            ptilde = torch.exp(Sij - m_new[:, None])

            R = pv_fp8_block(ptilde, vhat[ks:ke], v_scale)

            acc = alpha[:, None] * acc + R
            l = alpha * l + ptilde.sum(dim=-1)
            m = m_new

        Oi = acc / l[:, None]
        if v_mean is not None:
            Oi = Oi + v_mean
        O[qs:qe] = Oi

    return O


def make_inputs(seed=0):
    torch.manual_seed(seed)
    shared_q = torch.randn(1, D, device=device, dtype=dtype) * 8.0
    shared_k = torch.randn(1, D, device=device, dtype=dtype) * 8.0
    shared_v = torch.randn(1, D, device=device, dtype=dtype) * 8.0

    Q = shared_q + torch.randn(N, D, device=device, dtype=dtype) * 0.2
    K = shared_k + torch.randn(N, D, device=device, dtype=dtype) * 0.2
    V = shared_v + torch.randn(N, D, device=device, dtype=dtype) * 2.0
    return Q, K, V


def raw_int4_scores(Q, K):
    return quantized_scores(Q, K, "int4", False, False)


def raw_int4_fp8_attention(Q, K, V):
    return quantized_attention(Q, K, V, "int4", False, False, False)


def sa2_scores(Q, K, exact=False):
    return quantized_scores(Q, K, "int4", True, True, exact=exact)


def sa2_attention(Q, K, V, smooth_v=False):
    return quantized_attention(Q, K, V, "int4", True, True, smooth_v)


def build_ablation_rows(Q, K, V, S_ref, P_ref, O_ref):
    rows = []

    for qk_mode in qk_modes:
        for smooth_q, smooth_k, smooth_v in smooth_cases:
            S = quantized_scores(Q, K, qk_mode, smooth_q, smooth_k)
            P = F.softmax(S, dim=-1)
            O = quantized_attention(Q, K, V, qk_mode, smooth_q, smooth_k, smooth_v)
            rows.append({
                "qk_mode": qk_mode,
                "smooth": smooth_label(smooth_q, smooth_k, smooth_v),
                "score_l1": rel_l1(row_max_center(S), row_max_center(S_ref)).item(),
                "softmax_l1": rel_l1(P, P_ref).item(),
                "out_l1": rel_l1(O, O_ref).item(),
                "out_cos": cos(O, O_ref).item(),
            })

    return rows
