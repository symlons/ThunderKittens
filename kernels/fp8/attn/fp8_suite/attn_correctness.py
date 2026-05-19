"""Reusable correctness building blocks for the FP8 attention kernels.

Wraps the existing forward/backward kernel call paths from
`fp8_suite.test_forward` and `fp8_suite.test_backward_kernel` and
returns metric dicts (built on `tensor_metrics`) instead of raising,
so a sweep can aggregate across many cases.

Forward kernel output O is compared against three baselines:

  - `fp32`   : manual SDPA reference in fp32
  - `bf16`   : torch SDPA in bf16, cast to fp32
  - `quant`  : `fp8_quant_reference` (dequant Q,K -> fp32 SDPA),
               which is the *best the kernel could hope for given the
               input quantization* — measures the kernel-internal loss.

Backward kernel gradients (dQ, dK, dV) are compared against:

  - `fp32`   : torch SDPA fp32 backward
  - `manual` : the local `reference_backward` recipe
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .kernel_api import fp8_forward, require_extension
from .metrics import tensor_metrics
from .recipe import prepare_forward_inputs
from .references import (
    fp8_quant_reference,
    reference_attention,
    reference_backward,
    sdpa_backward,
)
from .test_backward_kernel import run_kernel as run_bwd_kernel


# ---------------------------------------------------------------------------
# Shape sweep
# ---------------------------------------------------------------------------


# Forward kernel requires N % 384 == 0 (LCM(3*qo_height=192, kv_height=128)).
def attn_correctness_cases(quick: bool = False):
    if quick:
        return [(1, 8, 384, 128), (1, 8, 1536, 128), (1, 8, 1536, 64)]
    cases = []
    for D in (64, 128):
        for N in (384, 768, 1536, 3072, 6144):
            cases.append((1, 8, N, D))
        cases += [(2, 16, 1536, D), (4, 8, 1536, D)]
    return cases


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _bf16_sdpa(Q, K, V):
    return F.scaled_dot_product_attention(
        Q.to(torch.bfloat16),
        K.to(torch.bfloat16),
        V.to(torch.bfloat16),
        is_causal=False,
    ).to(torch.float32)


def forward_metrics(B, H, N, D, *, seed):
    """Run the FP8 forward kernel and compare to three baselines."""
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    fwd = prepare_forward_inputs(Q, K, V)
    O_kern, _ = fp8_forward(fwd)
    O_kern = O_kern.to(torch.float32)

    O_fp32 = reference_attention(Q, K, V, causal=False)
    O_bf16 = _bf16_sdpa(Q, K, V)
    O_quant = fp8_quant_reference(fwd.Qq, fwd.Kq, fwd.sq, fwd.sk, fwd.K_mean, V, causal=False)

    return {
        "shape": (B, H, N, D),
        "seed": seed,
        "vs_fp32":  tensor_metrics(O_kern, O_fp32),
        "vs_bf16":  tensor_metrics(O_kern, O_bf16),
        "vs_quant": tensor_metrics(O_kern, O_quant),
        # Baseline: how much error bf16 SDPA itself introduces vs fp32.
        "bf16_vs_fp32": tensor_metrics(O_bf16, O_fp32),
    }


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


def backward_metrics(B, H, N, D, *, seed, fp8_dS_mode: int = 2, sr_dO: bool = True):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    K = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    V = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    dO = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)

    _, _, _, (dQ, dK, dV) = run_bwd_kernel(Q, K, V, dO, sr_dO=sr_dO, fp8_dS_mode=fp8_dS_mode)
    _, dQr, dKr, dVr = sdpa_backward(Q, K, V, dO, causal=False)
    dQa, dKa, dVa = reference_backward(Q, K, V, dO, causal=False)

    return {
        "shape": (B, H, N, D),
        "seed": seed,
        "fp8_dS_mode": fp8_dS_mode,
        "sr_dO": sr_dO,
        "vs_fp32":   {"dQ": tensor_metrics(dQ, dQr),
                      "dK": tensor_metrics(dK, dKr),
                      "dV": tensor_metrics(dV, dVr)},
        "vs_manual": {"dQ": tensor_metrics(dQ, dQa),
                      "dK": tensor_metrics(dK, dKa),
                      "dV": tensor_metrics(dV, dVa)},
        # Baseline: manual fp32 ref vs torch fp32 SDPA -> floor of the
        # comparison itself (should be ~120 dB if everything is sane).
        "manual_vs_fp32": {"dQ": tensor_metrics(dQa, dQr),
                           "dK": tensor_metrics(dKa, dKr),
                           "dV": tensor_metrics(dVa, dVr)},
    }


# ---------------------------------------------------------------------------
# Pass / fail bounds (in line with existing test_backward_kernel thresholds)
# ---------------------------------------------------------------------------


FWD_CRITERIA = {
    "vs_fp32":  {"min_qsnr": 18.0, "max_rel_l1": 0.25, "min_cos": 0.98},
    "vs_quant": {"min_qsnr": 25.0, "max_rel_l1": 0.10, "min_cos": 0.995},
}

BWD_CRITERIA = {
    "vs_fp32":  {"min_qsnr": 18.0, "max_rel_l1": 0.13, "min_cos": 0.985},
}


def _check(m, bounds):
    return (
        m["qsnr_dB"] >= bounds["min_qsnr"]
        and m["rel_L1"] <= bounds["max_rel_l1"]
        and m["cos"] >= bounds["min_cos"]
    )


def fwd_passed(r):
    return _check(r["vs_fp32"], FWD_CRITERIA["vs_fp32"]) and \
           _check(r["vs_quant"], FWD_CRITERIA["vs_quant"])


def bwd_passed(r):
    return all(_check(r["vs_fp32"][k], BWD_CRITERIA["vs_fp32"]) for k in ("dQ", "dK", "dV"))


def require_kernels():
    require_extension("fp8_mha_forward", "fp8_mha_backward")
