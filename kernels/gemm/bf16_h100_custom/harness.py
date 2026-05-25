from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import _C
import _gelu_bwd
import _linear_bwd_fused
from tk_bench import BenchResult, check_close, input_group_count, print_bench, profile_groups, uniform_bf16

_FA3_FUNC = None
_FA3_ERROR = None


def get_fa3_func():
    global _FA3_FUNC, _FA3_ERROR
    if _FA3_FUNC is not None:
        return _FA3_FUNC
    if _FA3_ERROR is not None:
        raise RuntimeError(_FA3_ERROR)
    try:
        from kernels import get_kernel

        fa3_module = get_kernel("kernels-community/flash-attn3", version=1)
        _FA3_FUNC = fa3_module.flash_attn_func
        return _FA3_FUNC
    except Exception as exc:
        _FA3_ERROR = repr(exc)
        raise


def _batch_index(batch: int, tokens_per_sample: int, device: torch.device) -> torch.Tensor:
    return torch.arange(batch * tokens_per_sample, device=device) // tokens_per_sample


def reference_forward(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens_per_sample: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_idx = _batch_index(shift.shape[0], tokens_per_sample, x.device)
    xf = x.float()
    mean = xf.mean(dim=1)
    var = (xf * xf).mean(dim=1) - mean * mean
    rstd = torch.rsqrt(torch.clamp(var, min=0.0) + eps)
    xhat = (xf - mean[:, None]) * rstd[:, None]
    out = xhat * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()
    return out.to(torch.bfloat16), mean, rstd


def reference_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    tokens_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = scale.shape[0]
    batch_idx = _batch_index(batch, tokens_per_sample, x.device)
    gf = grad.float()
    xhat = (x.float() - mean[:, None]) * rstd[:, None]
    dshift = torch.zeros((batch, x.shape[1]), device=x.device, dtype=torch.float32)
    dscale = torch.zeros_like(dshift)
    dshift.index_add_(0, batch_idx, gf)
    dscale.index_add_(0, batch_idx, gf * xhat)

    dnorm = gf * (1.0 + scale[batch_idx].float())
    s1 = dnorm.sum(dim=1, keepdim=True)
    s2 = (dnorm * xhat).sum(dim=1, keepdim=True)
    inv_k = 1.0 / x.shape[1]
    dx = (dnorm - s1 * inv_k - xhat * s2 * inv_k) * rstd[:, None]
    return dx.to(torch.bfloat16), dshift, dscale


def run_fused_forward(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens_per_sample: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    mean = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    rstd = torch.empty_like(mean)
    _C.layernorm_adaln(x, shift, scale, out, mean, rstd, tokens_per_sample, eps)
    return out, mean, rstd


def run_fused_backward(
    grad: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    tokens_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    dshift = torch.empty_like(scale, dtype=torch.float32)
    dscale = torch.empty_like(scale, dtype=torch.float32)
    _C.layernorm_adaln_backward(grad, x, scale, mean, rstd, dx, dshift, dscale, tokens_per_sample)
    return dx, dshift, dscale


class FusedAdaLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float) -> torch.Tensor:
        out, mean, rstd = run_fused_forward(x.contiguous(), shift.contiguous(), scale.contiguous(), tokens, eps)
        ctx.save_for_backward(x, scale, mean, rstd)
        ctx.tokens = tokens
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, scale, mean, rstd = ctx.saved_tensors
        dx, dshift, dscale = run_fused_backward(
            grad_out.contiguous(),
            x.contiguous(),
            scale.contiguous(),
            mean,
            rstd,
            ctx.tokens,
        )
        return dx, dshift, dscale, None, None


class FusedGatedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor, tokens: int) -> torch.Tensor:
        x_c = x.contiguous()
        h_c = h.contiguous()
        gate_c = gate.contiguous()
        out = torch.empty_like(x_c)
        _C.gated_residual(x_c, h_c, gate_c, out, tokens)
        ctx.save_for_backward(h_c, gate_c)
        ctx.tokens = tokens
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h, gate = ctx.saved_tensors
        grad = grad_out.contiguous()
        dx = torch.empty_like(grad)
        dh = torch.empty_like(grad)
        dgate = torch.empty_like(gate, dtype=torch.float32)
        _C.gated_residual_backward(grad, h, gate, dx, dh, dgate, ctx.tokens)
        return dx, dh, dgate, None


def gated_residual(x: torch.Tensor, h: torch.Tensor, gate: torch.Tensor, tokens: int) -> torch.Tensor:
    if (
        x.is_cuda
        and h.is_cuda
        and gate.is_cuda
        and x.dtype == torch.bfloat16
        and h.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.shape == h.shape
        and x.shape[-1] == gate.shape[-1]
        and x.numel() == gate.shape[0] * tokens * gate.shape[1]
    ):
        flat_x = x.reshape(-1, x.shape[-1])
        flat_h = h.reshape(-1, h.shape[-1])
        return FusedGatedResidual.apply(flat_x, flat_h, gate, tokens).reshape_as(x)

    batch_idx = _batch_index(gate.shape[0], tokens, x.device)
    gate_view = gate[batch_idx].to(h.dtype).reshape(x.shape)
    return x + gate_view * h


class TkGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return F.gelu(x, approximate="tanh")

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        _gelu_bwd.gelu_backward(grad_out.contiguous(), x.contiguous(), grad_input)
        return grad_input


class TkLinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        out = torch.empty((x_c.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        preact = torch.empty_like(out)
        _C.gemm_custom_native(x_c, w.contiguous(), out, b.contiguous(), preact)
        ctx.save_for_backward(x_c, w, preact)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w, preact = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dz = torch.empty_like(grad_out)
        db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
        _linear_bwd_fused.gelu_bwd_bias(grad_out, preact, dz, db)

        dw = torch.empty_like(w)
        dx = torch.empty_like(x)
        _linear_bwd_fused.dw_gemm(dz, x, dw)
        _linear_bwd_fused.dx_gemm_native(dz, w.contiguous(), dx)
        return dx, dw, db.to(grad_out.dtype)


class TkLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        ctx.save_for_backward(x_c, w)
        out = torch.empty((x_c.shape[0], w.shape[0]), device=x.device, dtype=x.dtype)
        _C.gemm_linear_native(x_c, w.contiguous(), out, b.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        dw = torch.empty_like(w)
        dx = torch.empty_like(x)
        _linear_bwd_fused.dw_gemm(grad_out, x, dw)
        _linear_bwd_fused.dx_gemm_native(grad_out, w.contiguous(), dx)
        db = torch.empty((grad_out.shape[1],), device=grad_out.device, dtype=torch.float32)
        _linear_bwd_fused.bias_reduce(grad_out, db)
        return dx, dw, db.to(grad_out.dtype)


def tk_mlp(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    h = TkLinearGelu.apply(x, w1, b1)
    return TkLinear.apply(h, w2, b2)


def correctness_case(batch: int, tokens_per_sample: int, dim: int, eps: float) -> bool:
    x = uniform_bf16((batch * tokens_per_sample, dim), 123, -2.0, 2.0)
    shift = uniform_bf16((batch, dim), 456, -0.5, 0.5)
    scale = uniform_bf16((batch, dim), 789, -0.25, 0.25)
    grad = uniform_bf16(x.shape, 321, -1.0, 1.0)

    out, mean, rstd = run_fused_forward(x, shift, scale, tokens_per_sample, eps)
    ref_out, ref_mean, ref_rstd = reference_forward(x, shift, scale, tokens_per_sample, eps)
    dx, dshift, dscale = run_fused_backward(grad, x, scale, mean, rstd, tokens_per_sample)
    ref_dx, ref_dshift, ref_dscale = reference_backward(grad, x, scale, ref_mean, ref_rstd, tokens_per_sample)

    print(f"\nCorrectness batch={batch} tokens={tokens_per_sample} dim={dim}")
    checks = [
        check_close("forward", out, ref_out, atol=1.6e-2),
        check_close("mean", mean, ref_mean, atol=2.5e-4, rtol=2e-3),
        check_close("rstd", rstd, ref_rstd, atol=2.5e-3, rtol=2e-3),
        check_close("dx", dx, ref_dx, atol=2.0e-2),
        check_close("dshift", dshift, ref_dshift, atol=1.5e-2, rtol=2e-3),
        check_close("dscale", dscale, ref_dscale, atol=2.5e-2, rtol=3e-3),
    ]
    return all(checks)


def run_tests() -> bool:
    torch.manual_seed(0)
    cases = [
        (1, 17, 257, 1e-5),
        (2, 128, 1024, 1e-5),
        (4, 256, 4096, 1e-5),
        (8, 64, 4096, 1e-5),
        (64, 32, 4096, 1e-5),
        (256, 16, 4096, 1e-5),
        (1024, 16, 4096, 1e-5),
        (4, 1024, 1024, 1e-5),
        (64, 32, 1024, 1e-5),
        (256, 16, 1024, 1e-5),
        (1024, 16, 1024, 1e-5),
    ]
    ok = all(correctness_case(*case) for case in cases)
    ok = mlp_branch_correctness(4, 128, 1024, 1e-6) and ok
    ok = mlp_branch_correctness(64, 16, 1024, 1e-6) and ok
    return ok


def pytorch_forward(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, tokens: int, eps: float) -> torch.Tensor:
    batch_idx = _batch_index(shift.shape[0], tokens, x.device)
    y = torch.nn.functional.layer_norm(x.float(), (x.shape[1],), None, None, eps)
    return (y * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()).to(torch.bfloat16)


def torch_mlp_branch(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    tokens: int,
    eps: float,
    tk_gelu: bool = False,
    tk_full_mlp: bool = False,
    fused_residual: bool = False,
) -> torch.Tensor:
    batch_idx = _batch_index(shift.shape[0], tokens, x.device)
    z = torch.nn.functional.layer_norm(x.float(), (x.shape[1],), None, None, eps)
    z = (z * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()).to(torch.bfloat16)
    if tk_full_mlp:
        h = tk_mlp(z, w1, b1, w2, b2)
    else:
        h = F.linear(z, w1, b1)
        h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
        h = F.linear(h, w2, b2)
    if fused_residual:
        return gated_residual(x, h, gate, tokens)
    return x + gate[batch_idx].to(h.dtype) * h


def fused_mlp_branch(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    tokens: int,
    eps: float,
    tk_gelu: bool = False,
    tk_full_mlp: bool = False,
    fused_residual: bool = True,
) -> torch.Tensor:
    batch_idx = _batch_index(shift.shape[0], tokens, x.device)
    z = FusedAdaLN.apply(x, shift, scale, tokens, eps)
    if tk_full_mlp:
        h = tk_mlp(z, w1, b1, w2, b2)
    else:
        h = F.linear(z, w1, b1)
        h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
        h = F.linear(h, w2, b2)
    if fused_residual:
        return gated_residual(x, h, gate, tokens)
    return x + gate[batch_idx].to(h.dtype) * h


def make_mlp_group(batch: int, tokens: int, dim: int, hidden_dim: int, seed: int) -> tuple[torch.Tensor, ...]:
    m = batch * tokens
    x = uniform_bf16((m, dim), seed + 0, -2.0, 2.0).requires_grad_(True)
    shift = uniform_bf16((batch, dim), seed + 1, -0.5, 0.5).requires_grad_(True)
    scale = uniform_bf16((batch, dim), seed + 2, -0.25, 0.25).requires_grad_(True)
    gate = uniform_bf16((batch, dim), seed + 3, -0.5, 0.5).requires_grad_(True)
    w1 = uniform_bf16((hidden_dim, dim), seed + 4, -0.02, 0.02).requires_grad_(True)
    b1 = uniform_bf16((hidden_dim,), seed + 5, -0.02, 0.02).requires_grad_(True)
    w2 = uniform_bf16((dim, hidden_dim), seed + 6, -0.02, 0.02).requires_grad_(True)
    b2 = uniform_bf16((dim,), seed + 7, -0.02, 0.02).requires_grad_(True)
    grad = uniform_bf16((m, dim), seed + 8, -1.0, 1.0)
    return x, shift, scale, gate, w1, b1, w2, b2, grad


def clone_mlp_group(group: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned = []
    for tensor in group:
        t = tensor.detach().clone()
        if tensor.requires_grad:
            t.requires_grad_(True)
        cloned.append(t)
    return tuple(cloned)


def zero_mlp_group_grads(group: tuple[torch.Tensor, ...]) -> None:
    for tensor in group[:-1]:
        tensor.grad = None


def mlp_branch_step(fn, group: tuple[torch.Tensor, ...], tokens: int, eps: float) -> None:
    y = fn(*group[:-1], tokens, eps)
    y.backward(group[-1])
    zero_mlp_group_grads(group)


def mlp_branch_step_tk_gelu(fn, group: tuple[torch.Tensor, ...], tokens: int, eps: float) -> None:
    y = fn(*group[:-1], tokens, eps, True)
    y.backward(group[-1])
    zero_mlp_group_grads(group)


def mlp_branch_step_tk_full_mlp(fn, group: tuple[torch.Tensor, ...], tokens: int, eps: float) -> None:
    y = fn(*group[:-1], tokens, eps, False, True)
    y.backward(group[-1])
    zero_mlp_group_grads(group)


def mlp_branch_correctness(batch: int, tokens: int, dim: int, eps: float) -> bool:
    hidden_dim = dim * 4
    base = make_mlp_group(batch, tokens, dim, hidden_dim, 9000)
    names = ("x", "shift", "scale", "gate", "w1", "b1", "w2", "b2")
    print(f"\nMLP branch correctness batch={batch} tokens={tokens} dim={dim}")
    ref_group = clone_mlp_group(base)
    y_ref = torch_mlp_branch(*ref_group[:-1], tokens, eps)
    y_ref.backward(ref_group[-1])

    variants = {
        "fused_adaln": (fused_mlp_branch, False, False),
        "tk_full_mlp": (torch_mlp_branch, False, True),
        "fused_adaln_tk_full_mlp": (fused_mlp_branch, False, True),
    }
    ok = True
    for variant, (fn, tk_gelu, tk_full_mlp) in variants.items():
        group = clone_mlp_group(base)
        y = fn(*group[:-1], tokens, eps, tk_gelu, tk_full_mlp)
        y.backward(group[-1])
        checks = [check_close(f"{variant} out", y, y_ref, atol=3.0e-2, rtol=8e-2)]
        for name, ref_t, fused_t in zip(names, ref_group[:-1], group[:-1], strict=True):
            checks.append(check_close(f"{variant} d{name}", fused_t.grad, ref_t.grad, atol=8.0e-2, rtol=1.2e-1))
        ok = all(checks) and ok
    return ok


def benchmark_case(batch: int, tokens: int, dim: int, eps: float, label: str) -> list[BenchResult]:
    m = batch * tokens
    input_bytes = (m * dim * 2) * 3 + (batch * dim * 2) * 2
    groups_n = min(input_group_count(input_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((m, dim), 1000 + i, -2.0, 2.0)
        shift = uniform_bf16((batch, dim), 2000 + i, -0.5, 0.5)
        scale = uniform_bf16((batch, dim), 3000 + i, -0.25, 0.25)
        grad = uniform_bf16((m, dim), 4000 + i, -1.0, 1.0)
        out, mean, rstd = run_fused_forward(x, shift, scale, tokens, eps)
        groups.append((x, shift, scale, grad, mean, rstd, out))

    bytes_forward = m * dim * 2 * 2 + batch * dim * 2 * 2 + m * 4 * 2
    bytes_backward = m * dim * 2 * 4 + batch * dim * 2 + m * 4 * 2 + batch * dim * 4 * 2

    print(f"\nShape ({label}): batch={batch} tokens={tokens} dim={dim}; input groups={groups_n}")

    results = [
        profile_groups(
            f"{label} tk forward",
            groups,
            lambda g: run_fused_forward(g[0], g[1], g[2], tokens, eps),
            warmup=500,
            iters=100,
            bytes_moved=bytes_forward,
        ),
        profile_groups(
            f"{label} tk backward",
            groups,
            lambda g: run_fused_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=500,
            iters=100,
            bytes_moved=bytes_backward,
        ),
        profile_groups(
            f"{label} torch forward",
            groups,
            lambda g: pytorch_forward(g[0], g[1], g[2], tokens, eps),
            warmup=500,
            iters=100,
            bytes_moved=bytes_forward,
        ),
        profile_groups(
            f"{label} torch backward",
            groups,
            lambda g: reference_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=500,
            iters=100,
            bytes_moved=bytes_backward,
        ),
    ]
    for result in results:
        print_bench(result)

    tk_fwd, tk_bwd, torch_fwd, torch_bwd = results
    print(
        f"  speedup: forward {torch_fwd.us / tk_fwd.us:.2f}x, "
        f"backward {torch_bwd.us / tk_bwd.us:.2f}x"
    )
    return results


def benchmark_big_adaln_case(batch: int, tokens: int, dim: int, eps: float, label: str) -> list[BenchResult]:
    m = batch * tokens
    x = uniform_bf16((m, dim), 21000, -2.0, 2.0)
    shift = uniform_bf16((batch, dim), 21001, -0.5, 0.5)
    scale = uniform_bf16((batch, dim), 21002, -0.25, 0.25)
    grad = uniform_bf16((m, dim), 21003, -1.0, 1.0)
    _, mean, rstd = run_fused_forward(x, shift, scale, tokens, eps)
    group = [(x, shift, scale, grad, mean, rstd)]

    bytes_forward = m * dim * 2 * 2 + batch * dim * 2 * 2 + m * 4 * 2
    bytes_backward = m * dim * 2 * 4 + batch * dim * 2 + m * 4 * 2 + batch * dim * 4 * 2
    print(f"\nBig AdaLN ({label}): batch={batch} tokens={tokens} dim={dim}; elements={m * dim:,}")
    results = [
        profile_groups(
            f"{label} big_adaln tk forward",
            group,
            lambda g: run_fused_forward(g[0], g[1], g[2], tokens, eps),
            warmup=500,
            iters=100,
            bytes_moved=bytes_forward,
        ),
        profile_groups(
            f"{label} big_adaln tk backward",
            group,
            lambda g: run_fused_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=500,
            iters=100,
            bytes_moved=bytes_backward,
        ),
    ]
    for result in results:
        print_bench(result)
    return results


def benchmark_big_adaln_suite() -> list[BenchResult]:
    print(
        "\nBig AdaLN bandwidth recipe: uniform BF16 inputs, one naturally cold large input group, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, cooldown between kernels."
    )
    cases = [
        (2048, 16, 1024, "D1024-B2048T16"),
        (4096, 16, 1024, "D1024-B4096T16"),
        (8192, 16, 1024, "D1024-B8192T16"),
        (1024, 16, 4096, "D4096-B1024T16"),
        (2048, 16, 4096, "D4096-B2048T16"),
        (4096, 16, 4096, "D4096-B4096T16"),
        (8192, 16, 4096, "D4096-B8192T16"),
    ]
    results: list[BenchResult] = []
    for batch, tokens, dim, label in cases:
        results.extend(benchmark_big_adaln_case(batch, tokens, dim, 1e-6, label))
    fwd = [r for r in results if "forward" in r.name]
    bwd = [r for r in results if "backward" in r.name]
    if fwd:
        peak = max(fwd, key=lambda r: r.bandwidth_tb_s or 0)
        print(f"\nPeak forward BW: {peak.bandwidth_tb_s:.3f} TB/s ({peak.name}, {peak.us:.2f} us)")
    if bwd:
        peak = max(bwd, key=lambda r: r.bandwidth_tb_s or 0)
        print(f"Peak backward BW: {peak.bandwidth_tb_s:.3f} TB/s ({peak.name}, {peak.us:.2f} us)")
    return results


def compile_fn(fn):
    try:
        return torch.compile(fn)
    except Exception as exc:
        print(f"torch.compile unavailable for {getattr(fn, '__name__', repr(fn))}: {exc!r}")
        return None


def benchmark_adaln_compile_case(batch: int, tokens: int, dim: int, eps: float, label: str) -> list[BenchResult]:
    m = batch * tokens
    input_bytes = (m * dim * 2) * 3 + (batch * dim * 2) * 2
    groups_n = min(input_group_count(input_bytes), 8)
    groups = []
    for i in range(groups_n):
        x = uniform_bf16((m, dim), 30000 + i, -2.0, 2.0)
        shift = uniform_bf16((batch, dim), 31000 + i, -0.5, 0.5)
        scale = uniform_bf16((batch, dim), 32000 + i, -0.25, 0.25)
        grad = uniform_bf16((m, dim), 33000 + i, -1.0, 1.0)
        out, mean, rstd = run_fused_forward(x, shift, scale, tokens, eps)
        groups.append((x, shift, scale, grad, mean, rstd, out))

    compiled_forward = compile_fn(lambda x, shift, scale: pytorch_forward(x, shift, scale, tokens, eps))
    compiled_backward = compile_fn(lambda grad, x, scale, mean, rstd: reference_backward(grad, x, scale, mean, rstd, tokens))

    bytes_forward = m * dim * 2 * 2 + batch * dim * 2 * 2 + m * 4 * 2
    bytes_backward = m * dim * 2 * 4 + batch * dim * 2 + m * 4 * 2 + batch * dim * 4 * 2

    print(f"\nAdaLN compile comparison ({label}): batch={batch} tokens={tokens} dim={dim}; input groups={groups_n}")
    results = [
        profile_groups(
            f"{label} adaln eager forward",
            groups,
            lambda g: pytorch_forward(g[0], g[1], g[2], tokens, eps),
            warmup=100,
            iters=50,
            bytes_moved=bytes_forward,
        )
    ]
    if compiled_forward is not None:
        results.append(profile_groups(
            f"{label} adaln compile forward",
            groups,
            lambda g: compiled_forward(g[0], g[1], g[2]),
            warmup=10,
            iters=50,
            bytes_moved=bytes_forward,
        ))
    results.append(profile_groups(
        f"{label} adaln tk forward",
        groups,
        lambda g: run_fused_forward(g[0], g[1], g[2], tokens, eps),
        warmup=100,
        iters=50,
        bytes_moved=bytes_forward,
    ))
    results.append(profile_groups(
        f"{label} adaln eager backward",
        groups,
        lambda g: reference_backward(g[3], g[0], g[2], g[4], g[5], tokens),
        warmup=100,
        iters=50,
        bytes_moved=bytes_backward,
    ))
    if compiled_backward is not None:
        results.append(profile_groups(
            f"{label} adaln compile backward",
            groups,
            lambda g: compiled_backward(g[3], g[0], g[2], g[4], g[5]),
            warmup=10,
            iters=50,
            bytes_moved=bytes_backward,
        ))
    results.append(profile_groups(
        f"{label} adaln tk backward",
        groups,
        lambda g: run_fused_backward(g[3], g[0], g[2], g[4], g[5], tokens),
        warmup=100,
        iters=50,
        bytes_moved=bytes_backward,
    ))
    for result in results:
        print_bench(result)
    return results


def benchmark_mlp_branch_case(batch: int, tokens: int, dim: int, eps: float, label: str) -> list[BenchResult]:
    hidden_dim = dim * 4
    m = batch * tokens
    group_bytes = (
        m * dim * 2 * 2
        + batch * dim * 2 * 3
        + hidden_dim * dim * 2
        + dim * hidden_dim * 2
        + hidden_dim * 2
        + dim * 2
    )
    groups_n = min(input_group_count(group_bytes), 8)
    torch_groups = [make_mlp_group(batch, tokens, dim, hidden_dim, 12000 + i * 100) for i in range(groups_n)]
    fused_groups = [clone_mlp_group(group) for group in torch_groups]
    fused_tk_gelu_groups = [clone_mlp_group(group) for group in torch_groups]
    tk_full_mlp_groups = [clone_mlp_group(group) for group in torch_groups]
    fused_tk_full_mlp_groups = [clone_mlp_group(group) for group in torch_groups]

    flops = 4.0 * m * dim * hidden_dim
    bytes_forward = group_bytes
    bytes_train = group_bytes * 3

    print(f"\nMLP branch ({label}): batch={batch} tokens={tokens} dim={dim} hidden={hidden_dim}; input groups={groups_n}")
    results = [
        profile_groups(
            f"{label} mlp torch forward",
            torch_groups,
            lambda g: torch_mlp_branch(*g[:-1], tokens, eps),
            warmup=500,
            iters=100,
            flops=flops,
            bytes_moved=bytes_forward,
        ),
        profile_groups(
            f"{label} mlp fused forward",
            fused_groups,
            lambda g: fused_mlp_branch(*g[:-1], tokens, eps),
            warmup=500,
            iters=100,
            flops=flops,
            bytes_moved=bytes_forward,
        ),
        profile_groups(
            f"{label} mlp torch train",
            torch_groups,
            lambda g: mlp_branch_step(torch_mlp_branch, g, tokens, eps),
            warmup=500,
            iters=100,
            flops=flops * 3.0,
            bytes_moved=bytes_train,
        ),
        profile_groups(
            f"{label} mlp fused train",
            fused_groups,
            lambda g: mlp_branch_step(fused_mlp_branch, g, tokens, eps),
            warmup=500,
            iters=100,
            flops=flops * 3.0,
            bytes_moved=bytes_train,
        ),
        profile_groups(
            f"{label} mlp fused+tk_gelu train",
            fused_tk_gelu_groups,
            lambda g: mlp_branch_step_tk_gelu(fused_mlp_branch, g, tokens, eps),
            warmup=500,
            iters=100,
            flops=flops * 3.0,
            bytes_moved=bytes_train,
        ),
        profile_groups(
            f"{label} mlp tk_full_mlp train",
            tk_full_mlp_groups,
            lambda g: mlp_branch_step_tk_full_mlp(torch_mlp_branch, g, tokens, eps),
            warmup=500,
            iters=100,
            flops=flops * 3.0,
            bytes_moved=bytes_train,
        ),
        profile_groups(
            f"{label} mlp fused+tk_full_mlp train",
            fused_tk_full_mlp_groups,
            lambda g: mlp_branch_step_tk_full_mlp(fused_mlp_branch, g, tokens, eps),
            warmup=500,
            iters=100,
            flops=flops * 3.0,
            bytes_moved=bytes_train,
        ),
    ]
    for result in results:
        print_bench(result)
    torch_fwd, fused_fwd, torch_train, fused_train, fused_tk_gelu_train, tk_full_mlp_train, fused_tk_full_mlp_train = results
    print(
        f"  MLP branch speedup: forward {torch_fwd.us / fused_fwd.us:.2f}x, "
        f"train {torch_train.us / fused_train.us:.2f}x, "
        f"train+tk_gelu {torch_train.us / fused_tk_gelu_train.us:.2f}x, "
        f"train+tk_full_mlp {torch_train.us / tk_full_mlp_train.us:.2f}x, "
        f"train+fused_adaln+tk_full_mlp {torch_train.us / fused_tk_full_mlp_train.us:.2f}x"
    )
    return results


def block_train_step_factory(heads: int, eps: float, fused_msa: bool, fused_mlp: bool, tk_gelu: bool = False, tk_full_mlp: bool = False, fused_residual: bool = False):
    def step(*args):
        *inputs, grad = args
        y = dit_block_forward(*inputs, heads, eps, fused_msa, fused_mlp, tk_gelu, tk_full_mlp, False, fused_residual)
        grads = torch.autograd.grad(y, inputs, grad, allow_unused=True)
        return grads
    return step


def benchmark_block_compile_case(batch: int, tokens: int, dim: int, heads: int, eps: float, label: str) -> list[BenchResult]:
    hidden_dim = dim * 4
    m = batch * tokens
    param_bytes = (
        (6 * dim * dim + 6 * dim)
        + (3 * dim * dim + 3 * dim)
        + (dim * dim + dim)
        + (hidden_dim * dim + hidden_dim)
        + (dim * hidden_dim + dim)
    ) * 2
    activation_bytes = (m * dim * 2 * 4) + (batch * dim * 2)
    group_bytes = param_bytes + activation_bytes
    groups_n = min(input_group_count(group_bytes), 4)
    base_groups = [make_block_group(batch, tokens, dim, heads, hidden_dim, 34000 + i * 100) for i in range(groups_n)]
    eager_groups = [clone_block_group(g) for g in base_groups]
    compile_groups = [clone_block_group(g) for g in base_groups]
    tk_groups = [clone_block_group(g) for g in base_groups]

    eager_step = block_train_step_factory(heads, eps, False, False)
    compile_step = compile_fn(eager_step)
    tk_step = block_train_step_factory(heads, eps, True, True, True)

    flops_fwd = (
        8.0 * m * dim * dim
        + 4.0 * m * dim * hidden_dim
        + 4.0 * batch * heads * tokens * tokens * (dim // heads)
    )
    print(f"\nDiT block compile comparison ({label}): batch={batch} tokens={tokens} dim={dim} heads={heads}; input groups={groups_n}")
    results = [
        profile_groups(
            f"{label} block eager train",
            eager_groups,
            lambda g: eager_step(*g),
            warmup=20,
            iters=20,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        )
    ]
    if compile_step is not None:
        results.append(profile_groups(
            f"{label} block compile train",
            compile_groups,
            lambda g: compile_step(*g),
            warmup=3,
            iters=20,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ))
    results.append(profile_groups(
        f"{label} block tk fused train",
        tk_groups,
        lambda g: tk_step(*g),
        warmup=20,
        iters=20,
        flops=flops_fwd * 3.0,
        bytes_moved=group_bytes * 3,
    ))
    for result in results:
        print_bench(result)
    baseline = results[0].us
    print("  block train speedup: " + ", ".join(f"{result.name} {baseline / result.us:.2f}x" for result in results[1:]))
    return results


def make_block_group(batch: int, tokens: int, dim: int, heads: int, hidden_dim: int, seed: int) -> tuple[torch.Tensor, ...]:
    x = uniform_bf16((batch, tokens, dim), seed + 0, -2.0, 2.0).requires_grad_(True)
    c = uniform_bf16((batch, dim), seed + 1, -1.0, 1.0).requires_grad_(True)
    adaln_w = uniform_bf16((6 * dim, dim), seed + 2, -0.02, 0.02).requires_grad_(True)
    adaln_b = uniform_bf16((6 * dim,), seed + 3, -0.02, 0.02).requires_grad_(True)
    qkv_w = uniform_bf16((3 * dim, dim), seed + 4, -0.02, 0.02).requires_grad_(True)
    qkv_b = uniform_bf16((3 * dim,), seed + 5, -0.02, 0.02).requires_grad_(True)
    proj_w = uniform_bf16((dim, dim), seed + 6, -0.02, 0.02).requires_grad_(True)
    proj_b = uniform_bf16((dim,), seed + 7, -0.02, 0.02).requires_grad_(True)
    w1 = uniform_bf16((hidden_dim, dim), seed + 8, -0.02, 0.02).requires_grad_(True)
    b1 = uniform_bf16((hidden_dim,), seed + 9, -0.02, 0.02).requires_grad_(True)
    w2 = uniform_bf16((dim, hidden_dim), seed + 10, -0.02, 0.02).requires_grad_(True)
    b2 = uniform_bf16((dim,), seed + 11, -0.02, 0.02).requires_grad_(True)
    grad = uniform_bf16((batch, tokens, dim), seed + 12, -1.0, 1.0)
    return x, c, adaln_w, adaln_b, qkv_w, qkv_b, proj_w, proj_b, w1, b1, w2, b2, grad


def clone_block_group(group: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    cloned = []
    for tensor in group:
        t = tensor.detach().clone()
        if tensor.requires_grad:
            t.requires_grad_(True)
        cloned.append(t)
    return tuple(cloned)


def zero_block_group_grads(group: tuple[torch.Tensor, ...]) -> None:
    for tensor in group[:-1]:
        tensor.grad = None


def adaln_input(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    tokens: int,
    eps: float,
    fused: bool,
) -> torch.Tensor:
    batch, seq, dim = x.shape
    if fused:
        flat = FusedAdaLN.apply(x.reshape(batch * seq, dim).contiguous(), shift, scale, tokens, eps)
        return flat.reshape(batch, seq, dim)
    y = torch.nn.functional.layer_norm(x.float(), (dim,), None, None, eps)
    return (y * (1.0 + scale[:, None, :].float()) + shift[:, None, :].float()).to(torch.bfloat16)


def attention_forward(
    x: torch.Tensor,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    heads: int,
    fa3: bool = False,
) -> torch.Tensor:
    batch, seq, dim = x.shape
    head_dim = dim // heads
    qkv = F.linear(x, qkv_w, qkv_b)
    qkv = qkv.reshape(batch, seq, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if fa3:
        flash_attn_func = get_fa3_func()
        y = flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            causal=False,
        )
        y = y.reshape(batch, seq, dim)
    else:
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        y = y.transpose(1, 2).reshape(batch, seq, dim)
    return F.linear(y, proj_w, proj_b)


def dit_block_forward(
    x: torch.Tensor,
    c: torch.Tensor,
    adaln_w: torch.Tensor,
    adaln_b: torch.Tensor,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    heads: int,
    eps: float,
    fused_msa: bool,
    fused_mlp: bool,
    tk_gelu: bool = False,
    tk_full_mlp: bool = False,
    fa3: bool = False,
    fused_residual: bool = False,
) -> torch.Tensor:
    batch, tokens, _ = x.shape
    params = F.linear(F.silu(c), adaln_w, adaln_b)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)

    attn_in = adaln_input(x, shift_msa, scale_msa, tokens, eps, fused_msa)
    attn_out = attention_forward(attn_in, qkv_w, qkv_b, proj_w, proj_b, heads, fa3)
    if fused_residual:
        x = gated_residual(x, attn_out, gate_msa, tokens)
    else:
        x = x + gate_msa[:, None, :].to(attn_out.dtype) * attn_out

    mlp_in = adaln_input(x, shift_mlp, scale_mlp, tokens, eps, fused_mlp)
    flat_mlp = mlp_in.reshape(batch * tokens, -1).contiguous()
    if tk_full_mlp:
        h = tk_mlp(flat_mlp, w1, b1, w2, b2).reshape(batch, tokens, -1)
    else:
        h = F.linear(mlp_in, w1, b1)
        h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
        h = F.linear(h, w2, b2)
    if fused_residual:
        return gated_residual(x, h, gate_mlp, tokens)
    return x + gate_mlp[:, None, :].to(h.dtype) * h


def block_step(
    group: tuple[torch.Tensor, ...],
    heads: int,
    eps: float,
    fused_msa: bool,
    fused_mlp: bool,
    tk_gelu: bool = False,
    tk_full_mlp: bool = False,
    fa3: bool = False,
    fused_residual: bool = False,
) -> None:
    y = dit_block_forward(*group[:-1], heads, eps, fused_msa, fused_mlp, tk_gelu, tk_full_mlp, fa3, fused_residual)
    y.backward(group[-1])
    zero_block_group_grads(group)


def block_correctness(batch: int, tokens: int, dim: int, heads: int, eps: float) -> bool:
    hidden_dim = dim * 4
    base = make_block_group(batch, tokens, dim, heads, hidden_dim, 15000)
    variants = {
        "tk_full_mlp": (False, False, False, True, False, True),
        "fused_mlp": (False, True, False, False, False, True),
        "fused_msa": (True, False, False, False, False, True),
        "fused_both": (True, True, False, False, False, True),
        "fused_both_tk_gelu": (True, True, True, False, False, True),
        "fused_both_tk_full_mlp": (True, True, False, True, False, True),
    }
    ref_group = clone_block_group(base)
    y_ref = dit_block_forward(*ref_group[:-1], heads, eps, False, False)
    y_ref.backward(ref_group[-1])
    ok = True
    print(f"\nDiT block correctness batch={batch} tokens={tokens} dim={dim}")
    for name, flags in variants.items():
        group = clone_block_group(base)
        y = dit_block_forward(*group[:-1], heads, eps, *flags)
        y.backward(group[-1])
        checks = [check_close(f"{name} out", y, y_ref, atol=3.0e-2, rtol=8e-2)]
        for grad_name, idx in (("x", 0), ("c", 1), ("adaln_w", 2), ("qkv_w", 4), ("w1", 8), ("w2", 10)):
            checks.append(check_close(f"{name} d{grad_name}", group[idx].grad, ref_group[idx].grad, atol=8.0e-2, rtol=1.2e-1))
        ok = all(checks) and ok
    return ok


def benchmark_block_case(batch: int, tokens: int, dim: int, heads: int, eps: float, label: str) -> list[BenchResult]:
    hidden_dim = dim * 4
    m = batch * tokens
    param_bytes = (
        (6 * dim * dim + 6 * dim)
        + (3 * dim * dim + 3 * dim)
        + (dim * dim + dim)
        + (hidden_dim * dim + hidden_dim)
        + (dim * hidden_dim + dim)
    ) * 2
    activation_bytes = (m * dim * 2 * 4) + (batch * dim * 2)
    group_bytes = param_bytes + activation_bytes
    groups_n = min(input_group_count(group_bytes), 4)
    base_groups = [make_block_group(batch, tokens, dim, heads, hidden_dim, 18000 + i * 100) for i in range(groups_n)]
    groups = {
        "torch": [clone_block_group(g) for g in base_groups],
        "tk_full_mlp": [clone_block_group(g) for g in base_groups],
        "fused_mlp": [clone_block_group(g) for g in base_groups],
        "fused_msa": [clone_block_group(g) for g in base_groups],
        "fused_both": [clone_block_group(g) for g in base_groups],
        "fused_both_tk_gelu": [clone_block_group(g) for g in base_groups],
        "fused_both_tk_full_mlp": [clone_block_group(g) for g in base_groups],
        "fused_both_fa3": [clone_block_group(g) for g in base_groups],
        "fused_both_fa3_tk_gelu": [clone_block_group(g) for g in base_groups],
        "fused_both_fa3_tk_full_mlp": [clone_block_group(g) for g in base_groups],
    }

    # Approximate forward FLOPs: attention projections/proj + MLP + QK/AV attention.
    flops_fwd = (
        8.0 * m * dim * dim
        + 4.0 * m * dim * hidden_dim
        + 4.0 * batch * heads * tokens * tokens * (dim // heads)
    )
    print(f"\nDiT block ({label}): batch={batch} tokens={tokens} dim={dim} heads={heads}; input groups={groups_n}")
    specs = [
        ("torch", False, False, False, False, False, False),
        ("tk_full_mlp", False, False, False, True, False, True),
        ("fused_mlp", False, True, False, False, False, True),
        ("fused_msa", True, False, False, False, False, True),
        ("fused_both", True, True, False, False, False, True),
        ("fused_both_tk_gelu", True, True, True, False, False, True),
        ("fused_both_tk_full_mlp", True, True, False, True, False, True),
        ("fused_both_fa3", True, True, False, False, True, True),
        ("fused_both_fa3_tk_gelu", True, True, True, False, True, True),
        ("fused_both_fa3_tk_full_mlp", True, True, False, True, True, True),
    ]
    results = []
    skipped = []
    for name, fused_msa, fused_mlp, tk_gelu, tk_full_mlp, fa3, fused_residual in specs:
        try:
            result = profile_groups(
                    f"{label} block {name} train",
                    groups[name],
                    lambda g, fm=fused_msa, fp=fused_mlp, tg=tk_gelu, tm=tk_full_mlp, f3=fa3, fr=fused_residual: block_step(g, heads, eps, fm, fp, tg, tm, f3, fr),
                    warmup=500,
                    iters=100,
                    flops=flops_fwd * 3.0,
                    bytes_moved=group_bytes * 3,
            )
        except Exception as exc:
            print(f"\n{name}: SKIP ({exc!r})")
            skipped.append(name)
            continue
        results.append(result)
        print_bench(result)

    baseline = results[0].us
    speedups = []
    for result in results[1:]:
        speedups.append(f"{result.name.split(' block ', 1)[1].rsplit(' train', 1)[0]} {baseline / result.us:.2f}x")
    if skipped:
        speedups.extend(f"{name} skipped" for name in skipped)
    print("  DiT block train speedup: " + ", ".join(speedups))
    return results


def benchmark() -> list[BenchResult]:
    print(
        "\nBenchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, cooldown between kernels."
    )
    results = benchmark_case(4, 1024, 4096, 1e-5, "base")
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_case(batch, 16, 4096, 1e-5, f"batch{batch}"))
    results.extend(benchmark_case(4, 1024, 1024, 1e-5, "L-D1024-base"))
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_case(batch, 16, 1024, 1e-5, f"L-D1024-batch{batch}"))
    results.extend(benchmark_mlp_branch_case(4, 1024, 1024, 1e-6, "L-D1024-base"))
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_mlp_branch_case(batch, 16, 1024, 1e-6, f"L-D1024-batch{batch}"))
    results.extend(benchmark_block_case(4, 1024, 1024, 16, 1e-6, "L-D1024-base"))
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_block_case(batch, 16, 1024, 16, 1e-6, f"L-D1024-batch{batch}"))
    return results


def run_block_tests() -> bool:
    ok = block_correctness(2, 64, 1024, 16, 1e-6)
    ok = block_correctness(16, 16, 1024, 16, 1e-6) and ok
    return ok


def benchmark_block_suite() -> list[BenchResult]:
    print(
        "\nBenchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, cooldown between kernels."
    )
    results = benchmark_block_case(4, 1024, 1024, 16, 1e-6, "L-D1024-base")
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_block_case(batch, 16, 1024, 16, 1e-6, f"L-D1024-batch{batch}"))
    return results


def benchmark_residual_sweep_case(batch: int, tokens: int, dim: int, heads: int, eps: float, label: str) -> list[BenchResult]:
    hidden_dim = dim * 4
    m = batch * tokens
    param_bytes = (
        (6 * dim * dim + 6 * dim)
        + (3 * dim * dim + 3 * dim)
        + (dim * dim + dim)
        + (hidden_dim * dim + hidden_dim)
        + (dim * hidden_dim + dim)
    ) * 2
    activation_bytes = (m * dim * 2 * 4) + (batch * dim * 2)
    group_bytes = param_bytes + activation_bytes
    groups_n = min(input_group_count(group_bytes), 4)
    base_groups = [make_block_group(batch, tokens, dim, heads, hidden_dim, 24000 + i * 100) for i in range(groups_n)]
    nores_groups = [clone_block_group(g) for g in base_groups]
    res_groups = [clone_block_group(g) for g in base_groups]

    flops_fwd = (
        8.0 * m * dim * dim
        + 4.0 * m * dim * hidden_dim
        + 4.0 * batch * heads * tokens * tokens * (dim // heads)
    )
    print(f"\nResidual sweep ({label}): batch={batch} tokens={tokens} total_tokens={m} dim={dim}; input groups={groups_n}")
    results = [
        profile_groups(
            f"{label} block fused_both_nores train",
            nores_groups,
            lambda g: block_step(g, heads, eps, True, True, False, False, False, False),
            warmup=500,
            iters=100,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ),
        profile_groups(
            f"{label} block fused_both_residual train",
            res_groups,
            lambda g: block_step(g, heads, eps, True, True, False, False, False, True),
            warmup=500,
            iters=100,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ),
    ]
    for result in results:
        print_bench(result)
    nores, res = results
    print(f"  residual fusion speedup: {nores.us / res.us:.2f}x")
    return results


def benchmark_residual_sweep_suite() -> list[BenchResult]:
    print(
        "\nResidual-stream benchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, 2s cooldown between kernels."
    )
    results: list[BenchResult] = []
    for tokens in (1, 2, 4, 8, 16):
        results.extend(benchmark_residual_sweep_case(1024, tokens, 1024, 16, 1e-6, f"L-D1024-B1024-T{tokens}"))
    for tokens in (1024, 2048, 4096, 8192, 16384):
        try:
            results.extend(benchmark_residual_sweep_case(1, tokens, 1024, 16, 1e-6, f"L-D1024-B1-T{tokens}"))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"\nL-D1024-B1-T{tokens}: SKIP OOM ({exc})")
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\nL-D1024-B1-T{tokens}: SKIP OOM ({exc})")
                torch.cuda.empty_cache()
            else:
                raise
    return results


def benchmark_long_batch_case(batch: int, tokens: int, dim: int, heads: int, eps: float, label: str) -> list[BenchResult]:
    hidden_dim = dim * 4
    m = batch * tokens
    param_bytes = (
        (6 * dim * dim + 6 * dim)
        + (3 * dim * dim + 3 * dim)
        + (dim * dim + dim)
        + (hidden_dim * dim + hidden_dim)
        + (dim * hidden_dim + dim)
    ) * 2
    activation_bytes = (m * dim * 2 * 4) + (batch * dim * 2)
    group_bytes = param_bytes + activation_bytes
    groups_n = min(input_group_count(group_bytes), 4)
    base_groups = [make_block_group(batch, tokens, dim, heads, hidden_dim, 28000 + i * 100) for i in range(groups_n)]
    torch_groups = [clone_block_group(g) for g in base_groups]
    fused_nores_groups = [clone_block_group(g) for g in base_groups]
    fused_res_groups = [clone_block_group(g) for g in base_groups]

    flops_fwd = (
        8.0 * m * dim * dim
        + 4.0 * m * dim * hidden_dim
        + 4.0 * batch * heads * tokens * tokens * (dim // heads)
    )
    print(f"\nLong-token batch sweep ({label}): batch={batch} tokens={tokens} total_tokens={m} dim={dim}; input groups={groups_n}")
    results = [
        profile_groups(
            f"{label} block torch train",
            torch_groups,
            lambda g: block_step(g, heads, eps, False, False, False, False, False, False),
            warmup=500,
            iters=100,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ),
        profile_groups(
            f"{label} block fused_adaln_nores train",
            fused_nores_groups,
            lambda g: block_step(g, heads, eps, True, True, False, False, False, False),
            warmup=500,
            iters=100,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ),
        profile_groups(
            f"{label} block fused_adaln_residual train",
            fused_res_groups,
            lambda g: block_step(g, heads, eps, True, True, False, False, False, True),
            warmup=500,
            iters=100,
            flops=flops_fwd * 3.0,
            bytes_moved=group_bytes * 3,
        ),
    ]
    for result in results:
        print_bench(result)
    torch_result, nores, res = results
    print(
        f"  total AdaLN speedup: nores {torch_result.us / nores.us:.2f}x, "
        f"with_residual {torch_result.us / res.us:.2f}x; residual-only {nores.us / res.us:.2f}x"
    )
    return results


def benchmark_long_batch_suite() -> list[BenchResult]:
    print(
        "\nLong-token/batch benchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, 2s cooldown between kernels."
    )
    cases = [
        (1, 1024), (2, 1024), (4, 1024), (8, 1024), (16, 1024), (32, 1024),
        (1, 2048), (2, 2048), (4, 2048), (8, 2048), (16, 2048),
        (1, 4096), (2, 4096), (4, 4096), (8, 4096),
        (1, 8192), (2, 8192), (4, 8192),
        (1, 16384), (2, 16384),
        (1024, 16),
    ]
    results: list[BenchResult] = []
    for batch, tokens in cases:
        label = f"L-D1024-B{batch}-T{tokens}"
        try:
            results.extend(benchmark_long_batch_case(batch, tokens, 1024, 16, 1e-6, label))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n{label}: SKIP OOM ({exc})")
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() or "invalid configuration" in str(exc).lower():
                print(f"\n{label}: SKIP ({exc})")
                torch.cuda.empty_cache()
            else:
                raise
    return results


def benchmark_b1024_tokens_suite() -> list[BenchResult]:
    print(
        "\nB1024 token benchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, 2s cooldown between kernels."
    )
    results: list[BenchResult] = []
    for tokens in (1, 2, 4, 8, 16, 32, 64):
        label = f"L-D1024-B1024-T{tokens}"
        try:
            results.extend(benchmark_long_batch_case(1024, tokens, 1024, 16, 1e-6, label))
        except torch.cuda.OutOfMemoryError as exc:
            print(f"\n{label}: SKIP OOM ({exc})")
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"\n{label}: SKIP OOM ({exc})")
                torch.cuda.empty_cache()
            else:
                raise
    return results


def benchmark_mlp_suite(tokens_filter: int | None = None) -> list[BenchResult]:
    print(
        "\nMLP-only benchmark recipe: uniform BF16 inputs, natural L2 eviction via input groups, "
        "500 warmups, 100 measured back-to-back launches, two CUDA events, cooldown between kernels."
    )
    results = []
    token_cases = (tokens_filter,) if tokens_filter is not None else (256, 512, 1024, 2048, 4096, 8192, 16384)
    for tokens in token_cases:
        for batch in (1, 2, 4, 8, 16, 32, 64, 128):
            label = f"L-D1024-B{batch}-T{tokens}"
            try:
                results.extend(benchmark_mlp_branch_case(batch, tokens, 1024, 1e-6, label))
            except torch.cuda.OutOfMemoryError as exc:
                print(f"\n{label}: SKIP OOM ({exc})")
                torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"\n{label}: SKIP OOM ({exc})")
                    torch.cuda.empty_cache()
                else:
                    raise
    return results


def benchmark_compile_suite() -> list[BenchResult]:
    print(
        "\nTorch eager / torch.compile / TK comparison: uniform BF16 inputs, natural L2 eviction via input groups, "
        "CUDA event timing. Compile timings exclude first-use compile by warming compiled functions before measurement."
    )
    results: list[BenchResult] = []
    results.extend(benchmark_adaln_compile_case(4, 1024, 1024, 1e-6, "L-D1024-base"))
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_adaln_compile_case(batch, 16, 1024, 1e-6, f"L-D1024-batch{batch}"))
    results.extend(benchmark_block_compile_case(4, 1024, 1024, 16, 1e-6, "L-D1024-base"))
    for batch in (4, 16, 64, 256, 1024):
        results.extend(benchmark_block_compile_case(batch, 16, 1024, 16, 1e-6, f"L-D1024-batch{batch}"))
    return results


def write_report(path: Path, ok: bool, results: list[BenchResult]) -> None:
    lines = ["# AdaLN LayerNorm Fusion Report", "", f"correctness: {'PASS' if ok else 'FAIL'}", ""]
    for result in results:
        bw = "" if result.bandwidth_tb_s is None else f", {result.bandwidth_tb_s:.3f} TB/s"
        lines.append(f"- {result.name}: {result.us:.2f} us{bw}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=[
        "test", "bench", "all", "block", "mlp", "mlp_t256", "mlp_t512", "mlp_t1024",
        "mlp_t2048", "mlp_t4096", "mlp_t8192", "mlp_t16384", "big_adaln",
        "compile", "residual_sweep", "long_batch", "b1024_tokens"
    ], nargs="?", default="all")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    ok = True
    results: list[BenchResult] = []
    if args.mode == "block":
        ok = run_block_tests()
        results = benchmark_block_suite()
    elif args.mode.startswith("mlp"):
        ok = mlp_branch_correctness(4, 128, 1024, 1e-6)
        ok = mlp_branch_correctness(64, 16, 1024, 1e-6) and ok
        token_filter = int(args.mode.split("_t", 1)[1]) if "_t" in args.mode else None
        results = benchmark_mlp_suite(token_filter)
    elif args.mode == "big_adaln":
        results = benchmark_big_adaln_suite()
    elif args.mode == "compile":
        ok = run_block_tests()
        results = benchmark_compile_suite()
    elif args.mode == "residual_sweep":
        ok = run_block_tests()
        results = benchmark_residual_sweep_suite()
    elif args.mode == "long_batch":
        ok = run_block_tests()
        results = benchmark_long_batch_suite()
    elif args.mode == "b1024_tokens":
        ok = run_block_tests()
        results = benchmark_b1024_tokens_suite()
    elif args.mode in ("test", "all"):
        ok = run_tests()
        if args.mode == "all":
            results = benchmark()
    elif args.mode == "bench":
        results = benchmark()
    if args.report is not None:
        write_report(args.report, ok, results)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
