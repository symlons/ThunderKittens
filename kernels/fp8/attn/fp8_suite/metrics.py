import math

import torch
import torch.nn.functional as F


def tensor_metrics(out, ref):
    out = out.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    diff = (out - ref).abs()
    sq_err = diff * diff
    sig = (ref * ref).sum().clamp_min(1e-30)
    noise = sq_err.sum().clamp_min(1e-30)
    ratio = (sig / noise).item()
    rmse = math.sqrt(sq_err.mean().item())
    return {
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "rmse": rmse,
        "rel_L1": diff.sum().item() / max(ref.abs().sum().item(), 1e-30),
        "rel_Linf": diff.max().item() / max(ref.abs().max().item(), 1e-30),
        "cos": F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item(),
        "qsnr_dB": 10.0 * math.log10(ratio) if ratio > 0 and math.isfinite(ratio) else float("nan"),
    }


def check_grad_metrics(label, dQm, dKm, dVm, *, min_qsnr, max_rel_l1, min_cos):
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


def fmt_forward(m):
    return (
        f"max={m['max']:.4f}  mean={m['mean']:.5f}  "
        f"rel-L1={m['rel_L1']:.3e}  rel-Linf={m['rel_Linf']:.3e}  "
        f"RMSE={m['rmse']:.3e}  cos={m['cos']:.6f}  QSNR={m['qsnr_dB']:.2f} dB"
    )


def _fmt_grad_one(name, m):
    return (
        f"{name}[QSNR={m['qsnr_dB']:5.2f} relL1={m['rel_L1']:.2e} "
        f"RMSE={m['rmse']:.2e} cos={m['cos']:.5f}]"
    )


def fmt_grad(label, dQm, dKm, dVm):
    return (
        f"{label:<28} "
        f"{_fmt_grad_one('dQ', dQm)} "
        f"{_fmt_grad_one('dK', dKm)} "
        f"{_fmt_grad_one('dV', dVm)}"
    )


def attention_fwd_tflops(B, H, N, D, ms):
    return (4.0 * B * H * N * N * D) / (ms * 1.0e-3) / 1.0e12


def attention_bwd_tflops(B, H, N, D, ms):
    return (10.0 * B * H * N * N * D) / (ms * 1.0e-3) / 1.0e12


def gbps(num_bytes, ms):
    return num_bytes / (ms * 1.0e-3) / 1.0e9

