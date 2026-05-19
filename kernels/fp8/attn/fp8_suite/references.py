import math

import torch
import torch.nn.functional as F

from fp8_attn_bwd_ref import BwdRecipe, fp8_attention_backward, reference_backward


def reference_attention(Q, K, V, causal=False):
    S = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if causal:
        N = S.shape[-1]
        mask = torch.triu(torch.ones(N, N, device=S.device, dtype=torch.bool), 1)
        S = S.masked_fill(mask, float("-inf"))
    return F.softmax(S, dim=-1) @ V


def sdpa_backward(Q, K, V, dO, *, causal=False):
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    O_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=causal)
    O_ref.backward(dO.detach())
    return O_ref.detach(), Q_ref.grad.detach(), K_ref.grad.detach(), V_ref.grad.detach()


def fp8_quant_reference(Qq, Kq, sq, sk, K_mean, V, causal=False):
    Q_eq = Qq.to(torch.float32) * sq.unsqueeze(-1)
    K_eq = Kq.to(torch.float32) * sk.unsqueeze(-1) + K_mean
    return reference_attention(Q_eq, K_eq, V, causal=causal)

