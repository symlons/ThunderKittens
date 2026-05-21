import math
from dataclasses import dataclass

import torch

from .kernel_api import (
    cuda_quantize_per_channel,
    cuda_quantize_per_token,
    cuda_quantize_per_token_int8,
)
from .quant import (
    broadcast_descale,
    quantize_per_channel_fp8,
    quantize_per_channel_fp8_sr,
    quantize_per_row_fp8,
    quantize_per_row_int8,
    quantize_per_tensor_fp8,
    quantize_per_tensor_fp8_sr,
    quantize_with_descale_fp8_sr,
)


@dataclass
class ForwardInputs:
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    K_mean: torch.Tensor
    V_mean: torch.Tensor
    K_s: torch.Tensor
    V_s: torch.Tensor
    Qq: torch.Tensor
    Kq: torch.Tensor
    Vbf: torch.Tensor
    sq: torch.Tensor
    sk: torch.Tensor
    vm: torch.Tensor

    @property
    def Q_eff(self):
        return self.Qq.to(torch.float32) * self.sq.unsqueeze(-1)

    @property
    def K_eff_centered(self):
        return self.Kq.to(torch.float32) * self.sk.unsqueeze(-1)

    @property
    def V_centered(self):
        return self.Vbf.to(torch.float32)


@dataclass
class BackwardInputs:
    fwd: ForwardInputs
    dO: torch.Tensor
    O_bf: torch.Tensor
    L_raw: torch.Tensor
    Qq: torch.Tensor
    Kq: torch.Tensor
    Vq: torch.Tensor
    dOq: torch.Tensor
    Qq_t: torch.Tensor
    dOq_t: torch.Tensor
    Kq_t: torch.Tensor          # FP8 K_s^T (B, H_kv, D, N) per-D-channel SR
    dO_bf: torch.Tensor
    sq: torch.Tensor
    sk: torch.Tensor
    sv: torch.Tensor
    sdo_row: torch.Tensor
    sdp_row: torch.Tensor
    sq_ch: torch.Tensor
    sdo_ch: torch.Tensor
    sk_ch: torch.Tensor         # per-D-channel scale of K_s^T (for FP8 dQ)
    Vq_ch: torch.Tensor
    sv_ch: torch.Tensor


def prepare_forward_inputs(Q, K, V, *, smooth_k=True, smooth_v=True,
                           use_cuda_quant=False, quant_dtype="fp8"):
    """Quantize Q,K for the FP8 (default) or INT8 forward kernel.

    ``quant_dtype`` selects the GEMM1 quantization:
      - "fp8":  per-token FP8 e4m3 (SageAttention2)
      - "int8": per-token symmetric INT8 (SageBwd-style INT8 GEMM1)
    """
    K_mean = K.mean(dim=-2, keepdim=True) if smooth_k else torch.zeros_like(K[..., :1, :])
    V_mean = V.mean(dim=-2, keepdim=True) if smooth_v else torch.zeros_like(V[..., :1, :])
    K_s = K - K_mean
    V_s = V - V_mean

    if quant_dtype == "fp8":
        quant = cuda_quantize_per_token if use_cuda_quant else quantize_per_row_fp8
    elif quant_dtype == "int8":
        quant = cuda_quantize_per_token_int8 if use_cuda_quant else quantize_per_row_int8
    else:
        raise ValueError(f"unknown quant_dtype {quant_dtype!r}")
    Qq, sq = quant(Q)
    Kq, sk = quant(K_s)
    return ForwardInputs(
        Q=Q,
        K=K,
        V=V,
        K_mean=K_mean,
        V_mean=V_mean,
        K_s=K_s,
        V_s=V_s,
        Qq=Qq,
        Kq=Kq,
        Vbf=V_s.to(torch.bfloat16).contiguous(),
        sq=sq.contiguous().to(torch.float32),
        sk=sk.contiguous().to(torch.float32),
        vm=V_mean.squeeze(-2).to(torch.bfloat16).contiguous(),
    )


def natural_lse_from_kernel_l(L_raw, D):
    return -L_raw.to(torch.float32) / math.sqrt(D)


def estimate_dS_descale(Qq, Kq, Vq, dOq, sq, sk, sv, sdo_descale):
    inv_sqrt_d = 1.0 / math.sqrt(Qq.shape[-1])
    Q_eff = Qq.to(torch.float32) * sq.unsqueeze(-1)
    K_eff = Kq.to(torch.float32) * sk.unsqueeze(-1)
    V_eff = Vq.to(torch.float32) * sv.unsqueeze(-1)
    dO_eff = dOq.to(torch.float32) * sdo_descale
    P = torch.softmax((Q_eff @ K_eff.transpose(-2, -1)) * inv_sqrt_d, dim=-1)
    dP = dO_eff @ V_eff.transpose(-2, -1)
    dS = P * (dP - (P * dP).sum(dim=-1, keepdim=True)) * inv_sqrt_d
    return (dS.abs().amax().clamp_min(1e-12) / 448.0).to(torch.float32)


def prepare_backward_inputs(fwd, O_bf, L_raw, dO, *, sr_dO=True, use_cuda_quant=True,
                            sdp_descale_mode="estimate"):
    token_quant = cuda_quantize_per_token if use_cuda_quant else quantize_per_row_fp8
    channel_quant = cuda_quantize_per_channel if use_cuda_quant else quantize_per_channel_fp8

    Vq_ch, sv_ch = channel_quant(fwd.V_s)
    Vq, sv = token_quant(fwd.V_s)
    dOq, sdo_descale = quantize_per_tensor_fp8_sr(dO) if sr_dO else quantize_per_tensor_fp8(dO)

    Q_T = fwd.Q.transpose(-1, -2).contiguous()
    dO_T = dO.transpose(-1, -2).contiguous()
    K_T = fwd.K_s.transpose(-1, -2).contiguous()        # smoothed K, transposed for FP8 dQ
    Qq_t, sq_ch = quantize_per_channel_fp8_sr(Q_T)
    dOq_t = quantize_with_descale_fp8_sr(dO_T, sdo_descale)
    Kq_t, sk_ch = quantize_per_channel_fp8_sr(K_T)      # per-D-channel scale, SR token-wise

    sdo_row = broadcast_descale(sdo_descale, fwd.sq)
    if sdp_descale_mode == "estimate":
        sdp_descale = estimate_dS_descale(fwd.Qq, fwd.Kq, Vq, dOq, fwd.sq, fwd.sk, sv, sdo_descale)
    elif sdp_descale_mode == "constant":
        # Profile-only path: avoids the O(N^2) Python reference used to
        # estimate dS range. This preserves kernel timing but is not a
        # correctness-quality configuration.
        sdp_descale = torch.ones((), device=fwd.Qq.device, dtype=torch.float32)
    else:
        raise ValueError(f"unknown sdp_descale_mode {sdp_descale_mode!r}")
    sdp_row = broadcast_descale(sdp_descale, fwd.sq)

    return BackwardInputs(
        fwd=fwd,
        dO=dO,
        O_bf=O_bf,
        L_raw=L_raw,
        Qq=fwd.Qq,
        Kq=fwd.Kq,
        Vq=Vq,
        dOq=dOq,
        Qq_t=Qq_t,
        dOq_t=dOq_t,
        Kq_t=Kq_t,
        dO_bf=(dOq.to(torch.float32) * sdo_descale).to(torch.bfloat16).contiguous(),
        sq=fwd.sq,
        sk=fwd.sk,
        sv=sv.contiguous().to(torch.float32),
        sdo_row=sdo_row,
        sdp_row=sdp_row,
        sq_ch=sq_ch.contiguous().to(torch.float32),
        sdo_ch=broadcast_descale(sdo_descale, sq_ch),
        sk_ch=sk_ch.contiguous().to(torch.float32),
        Vq_ch=Vq_ch,
        sv_ch=sv_ch.contiguous().to(torch.float32),
    )

