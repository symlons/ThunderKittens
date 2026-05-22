"""Reusable building blocks for FP8 and INT8 quantization correctness sweeps.

Provides:
  - INPUT_DISTRIBUTIONS / make_input: synthetic inputs that stress
    different aspects of dynamic per-row / per-channel quantization.
  - correctness_cases: shape sweep with chunk-boundary stress.
  - quant_kernel_metrics: scale + dequant + quantized-code metrics for a
    single (xq, scale, xq_ref, scale_ref) tuple, built on top of the
    existing `tensor_metrics`.
  - CRITERIA / passed: shared pass/fail thresholds (same bounds as
    the existing `check_quant_kernel` in test_backward_kernel.py).
"""

from collections import OrderedDict

import torch

from .metrics import tensor_metrics


# ---------------------------------------------------------------------------
# Input distributions
# ---------------------------------------------------------------------------


def _uniform(shape, gen):
    return torch.empty(shape, device="cuda").uniform_(-1.0, 1.0, generator=gen)


def _normal(shape, gen):
    return torch.randn(shape, device="cuda", generator=gen)


def _scaled_normal(scale):
    def _f(shape, gen):
        return torch.randn(shape, device="cuda", generator=gen) * scale
    return _f


def _positive(shape, gen):
    return torch.empty(shape, device="cuda").uniform_(0.0, 5.0, generator=gen)


def _negative(shape, gen):
    return -torch.empty(shape, device="cuda").uniform_(0.0, 5.0, generator=gen)


def _zeros(shape, gen):
    return torch.zeros(shape, device="cuda")


def _mixed(shape, gen):
    """5% outliers ~500x, 95% values ~0.01x — exercises amax dynamics."""
    x = torch.randn(shape, device="cuda", generator=gen)
    mask = torch.rand(shape, device="cuda", generator=gen) < 0.05
    return torch.where(mask, x * 500.0, x * 0.01)


def _one_outlier_per_row(shape, gen):
    """One huge value per (B,H,N) row — per-token amax dictated by one element."""
    D = shape[-1]
    x = torch.empty(shape, device="cuda").uniform_(-0.1, 0.1, generator=gen)
    idx = torch.randint(0, D, shape[:-1], device="cuda", generator=gen)
    flat = x.view(-1, D)
    flat[torch.arange(flat.size(0), device="cuda"), idx.view(-1)] = 300.0
    return x


INPUT_DISTRIBUTIONS = OrderedDict([
    ("uniform", _uniform),
    ("normal", _normal),
    ("large", _scaled_normal(1000.0)),
    ("tiny", _scaled_normal(1e-4)),
    ("positive", _positive),
    ("negative", _negative),
    ("zeros", _zeros),
    ("mixed", _mixed),
    ("one_outlier_per_row", _one_outlier_per_row),
])


def make_input(kind, shape, *, generator):
    if kind not in INPUT_DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {kind}; "
                         f"choose from {list(INPUT_DISTRIBUTIONS)}")
    return INPUT_DISTRIBUTIONS[kind](shape, generator).to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Shape sweep
# ---------------------------------------------------------------------------


def correctness_cases(quick=False):
    if quick:
        return [(1, 8, 1536, 128), (1, 8, 1536, 64), (2, 16, 3072, 128)]
    shapes = []
    for B in (1, 4, 16):
        for H in (1, 16):
            # N values include chunk-boundary stress around CHUNK_ROWS=256.
            for N in (1, 16, 31, 128, 255, 256, 257, 512, 1024, 2048, 4096, 16384):
                for D in (64, 128):
                    shapes.append((B, H, N, D))
    # Extra per-channel chunk-boundary cases at non-trivial (B,H).
    shapes += [
        (2, 4, 255, 128), (2, 4, 257, 128),
        (2, 4, 511, 128), (2, 4, 513, 128),
        (2, 4, 4101, 128),
    ]
    return shapes


# ---------------------------------------------------------------------------
# Metrics (built on tensor_metrics from fp8_suite.metrics)
# ---------------------------------------------------------------------------


CRITERIA = {
    "scale_max": 1e-6,
    "scale_rel_L1": 1e-6,
    "deq_qsnr_dB": 50.0,
    "deq_rel_L1": 1e-4,
    "deq_cos": 0.99999,
    "code_byte_delta": 1,
}


def _dequant(xq, scale, granularity):
    if granularity == "token":
        return xq.to(torch.float32) * scale.unsqueeze(-1)
    if granularity == "channel":
        return xq.to(torch.float32) * scale.unsqueeze(-2)
    raise ValueError(f"unknown granularity {granularity!r}")


def quant_kernel_metrics(xq, scale, xq_ref, scale_ref, *, granularity, x_fp32=None):
    """Return {scale, dequant, code byte[, vs_fp32]} metrics for one case.

    Mirrors check_quant_kernel: compares (a) the fp32 scale tensors and
    (b) the dequantized values xq*scale. Also reports quantized-code byte
    agreement. If `x_fp32` is provided, additionally reports intrinsic
    quantization noise via `tensor_metrics(dequant_kernel, x_fp32)`.
    """
    deq_got = _dequant(xq, scale, granularity)
    deq_ref = _dequant(xq_ref, scale_ref, granularity)
    a = xq.view(torch.uint8)
    b = xq_ref.view(torch.uint8)
    out = {
        "scale": tensor_metrics(scale, scale_ref),
        "dequant": tensor_metrics(deq_got, deq_ref),
        "code_byte_exact_frac": (a == b).float().mean().item(),
        "code_byte_max_delta": (a.to(torch.int16) - b.to(torch.int16))
                               .abs().max().item(),
    }
    if x_fp32 is not None:
        out["vs_fp32"] = tensor_metrics(deq_got, x_fp32)
    return out


def passed(m):
    s = m["scale"]
    d = m["dequant"]
    # Bit-identical dequant tensors -> pass regardless of scale numerics.
    # This covers the zero-input edge case where the kernel's floor
    # (clamp scale >= 1e-12) differs from the reference's floor
    # (clamp amax >= 1e-12 then divide by 448).
    if d["max"] == 0.0 and m["code_byte_max_delta"] == 0:
        return True
    deq_ok = (
        d["rel_L1"] <= CRITERIA["deq_rel_L1"]
        and d["qsnr_dB"] >= CRITERIA["deq_qsnr_dB"]
        and d["cos"] >= CRITERIA["deq_cos"]
    )
    return (
        s["max"] <= CRITERIA["scale_max"]
        and s["rel_L1"] <= CRITERIA["scale_rel_L1"]
        and deq_ok
        and m["code_byte_max_delta"] <= CRITERIA["code_byte_delta"]
    )
