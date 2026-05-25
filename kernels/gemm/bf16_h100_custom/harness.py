from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import _C
import _gelu_bwd
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
) -> torch.Tensor:
    batch_idx = _batch_index(shift.shape[0], tokens, x.device)
    z = torch.nn.functional.layer_norm(x.float(), (x.shape[1],), None, None, eps)
    z = (z * (1.0 + scale[batch_idx].float()) + shift[batch_idx].float()).to(torch.bfloat16)
    h = F.linear(z, w1, b1)
    h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
    h = F.linear(h, w2, b2)
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
) -> torch.Tensor:
    batch_idx = _batch_index(shift.shape[0], tokens, x.device)
    z = FusedAdaLN.apply(x, shift, scale, tokens, eps)
    h = F.linear(z, w1, b1)
    h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
    h = F.linear(h, w2, b2)
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


def mlp_branch_correctness(batch: int, tokens: int, dim: int, eps: float) -> bool:
    hidden_dim = dim * 4
    torch_group = make_mlp_group(batch, tokens, dim, hidden_dim, 9000)
    fused_group = clone_mlp_group(torch_group)
    y_ref = torch_mlp_branch(*torch_group[:-1], tokens, eps)
    y = fused_mlp_branch(*fused_group[:-1], tokens, eps)
    y_ref.backward(torch_group[-1])
    y.backward(fused_group[-1])

    names = ("x", "shift", "scale", "gate", "w1", "b1", "w2", "b2")
    print(f"\nMLP branch correctness batch={batch} tokens={tokens} dim={dim}")
    checks = [check_close("out", y, y_ref, atol=2.5e-2)]
    for name, ref_t, fused_t in zip(names, torch_group[:-1], fused_group[:-1], strict=True):
        checks.append(check_close(f"d{name}", fused_t.grad, ref_t.grad, atol=4.0e-2, rtol=8e-2))
    return all(checks)


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
    ]
    for result in results:
        print_bench(result)
    torch_fwd, fused_fwd, torch_train, fused_train, fused_tk_gelu_train = results
    print(
        f"  MLP branch speedup: forward {torch_fwd.us / fused_fwd.us:.2f}x, "
        f"train {torch_train.us / fused_train.us:.2f}x, "
        f"train+tk_gelu {torch_train.us / fused_tk_gelu_train.us:.2f}x"
    )
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
    fa3: bool = False,
) -> torch.Tensor:
    batch, tokens, _ = x.shape
    params = F.linear(F.silu(c), adaln_w, adaln_b)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)

    attn_in = adaln_input(x, shift_msa, scale_msa, tokens, eps, fused_msa)
    x = x + gate_msa[:, None, :].to(x.dtype) * attention_forward(attn_in, qkv_w, qkv_b, proj_w, proj_b, heads, fa3)

    mlp_in = adaln_input(x, shift_mlp, scale_mlp, tokens, eps, fused_mlp)
    h = F.linear(mlp_in, w1, b1)
    h = TkGelu.apply(h) if tk_gelu else F.gelu(h, approximate="tanh")
    h = F.linear(h, w2, b2)
    return x + gate_mlp[:, None, :].to(h.dtype) * h


def block_step(
    group: tuple[torch.Tensor, ...],
    heads: int,
    eps: float,
    fused_msa: bool,
    fused_mlp: bool,
    tk_gelu: bool = False,
    fa3: bool = False,
) -> None:
    y = dit_block_forward(*group[:-1], heads, eps, fused_msa, fused_mlp, tk_gelu, fa3)
    y.backward(group[-1])
    zero_block_group_grads(group)


def block_correctness(batch: int, tokens: int, dim: int, heads: int, eps: float) -> bool:
    hidden_dim = dim * 4
    base = make_block_group(batch, tokens, dim, heads, hidden_dim, 15000)
    variants = {
        "fused_mlp": (False, True, False, False),
        "fused_msa": (True, False, False, False),
        "fused_both": (True, True, False, False),
        "fused_both_tk_gelu": (True, True, True, False),
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
        "fused_mlp": [clone_block_group(g) for g in base_groups],
        "fused_msa": [clone_block_group(g) for g in base_groups],
        "fused_both": [clone_block_group(g) for g in base_groups],
        "fused_both_tk_gelu": [clone_block_group(g) for g in base_groups],
        "fused_both_fa3": [clone_block_group(g) for g in base_groups],
        "fused_both_fa3_tk_gelu": [clone_block_group(g) for g in base_groups],
    }

    # Approximate forward FLOPs: attention projections/proj + MLP + QK/AV attention.
    flops_fwd = (
        8.0 * m * dim * dim
        + 4.0 * m * dim * hidden_dim
        + 4.0 * batch * heads * tokens * tokens * (dim // heads)
    )
    print(f"\nDiT block ({label}): batch={batch} tokens={tokens} dim={dim} heads={heads}; input groups={groups_n}")
    specs = [
        ("torch", False, False, False, False),
        ("fused_mlp", False, True, False, False),
        ("fused_msa", True, False, False, False),
        ("fused_both", True, True, False, False),
        ("fused_both_tk_gelu", True, True, True, False),
        ("fused_both_fa3", True, True, False, True),
        ("fused_both_fa3_tk_gelu", True, True, True, True),
    ]
    results = []
    skipped = []
    for name, fused_msa, fused_mlp, tk_gelu, fa3 in specs:
        try:
            result = profile_groups(
                    f"{label} block {name} train",
                    groups[name],
                    lambda g, fm=fused_msa, fp=fused_mlp, tg=tk_gelu, f3=fa3: block_step(g, heads, eps, fm, fp, tg, f3),
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


def write_report(path: Path, ok: bool, results: list[BenchResult]) -> None:
    lines = ["# AdaLN LayerNorm Fusion Report", "", f"correctness: {'PASS' if ok else 'FAIL'}", ""]
    for result in results:
        bw = "" if result.bandwidth_tb_s is None else f", {result.bandwidth_tb_s:.3f} TB/s"
        lines.append(f"- {result.name}: {result.us:.2f} us{bw}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "bench", "all", "block", "big_adaln"], nargs="?", default="all")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    ok = True
    results: list[BenchResult] = []
    if args.mode == "block":
        ok = run_block_tests()
        results = benchmark_block_suite()
    elif args.mode == "big_adaln":
        results = benchmark_big_adaln_suite()
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
