from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import torch.nn.functional as F

from dit3d_e2e_bench import (
    FusedAdaLNLinear,
    FusedAdaLNLinearGelu,
    FusedLinearGatedResidual,
    fused_adaln_linear_gelu,
    fused_linear_gated_residual,
    modulate,
    tk_gemm_gelu_ln_adaln_op,
    tk_gemm_linear_gated_residual_op,
    tk_gemm_linear_ln_adaln_op,
)
from tk_bench import profile_groups, uniform_bf16


def clone_requires(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def zero_grads(group: tuple[torch.Tensor, ...]) -> None:
    for tensor in group:
        if tensor.grad is not None:
            tensor.grad = None


def train_step(fn: Callable[..., torch.Tensor], group: tuple[torch.Tensor, ...]) -> None:
    *inputs, grad = group
    out = fn(*inputs)
    out.backward(grad)
    zero_grads(group)


def compile_fn(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    return torch.compile(fn, fullgraph=False)


def torch_ln_linear(x, shift, scale, w, b, eps):
    return F.linear(modulate(F.layer_norm(x, (x.shape[-1],), eps=eps), shift, scale), w, b)


def torch_ln_linear_gelu(x, shift, scale, w, b, eps):
    return F.gelu(torch_ln_linear(x, shift, scale, w, b, eps), approximate="tanh")


def torch_linear_gated_residual(h, residual, gate, w, b):
    return residual + gate.unsqueeze(1) * F.linear(h, w, b)


def torch_mlp_branch(x, shift, scale, gate, w1, b1, w2, b2, eps):
    h = torch_ln_linear_gelu(x, shift, scale, w1, b1, eps)
    return torch_linear_gated_residual(h, x, gate, w2, b2)


def custom_ln_linear(x, shift, scale, w, b, eps):
    batch, tokens, dim = x.shape
    out, _, _, _ = tk_gemm_linear_ln_adaln_op(
        x.reshape(batch * tokens, dim).contiguous(),
        w.contiguous(),
        b.contiguous(),
        shift.contiguous(),
        scale.contiguous(),
        tokens,
        eps,
    )
    return out.reshape(batch, tokens, w.shape[0])


def custom_ln_linear_gelu(x, shift, scale, w, b, eps):
    batch, tokens, dim = x.shape
    out, _, _, _ = tk_gemm_gelu_ln_adaln_op(
        x.reshape(batch * tokens, dim).contiguous(),
        w.contiguous(),
        b.contiguous(),
        shift.contiguous(),
        scale.contiguous(),
        tokens,
        eps,
    )
    return out.reshape(batch, tokens, w.shape[0])


def custom_linear_gated_residual(h, residual, gate, w, b):
    batch, tokens, _ = h.shape
    out, _ = tk_gemm_linear_gated_residual_op(
        h.reshape(batch * tokens, h.shape[-1]).contiguous(),
        w.contiguous(),
        residual.reshape(batch * tokens, residual.shape[-1]).contiguous(),
        gate.contiguous(),
        b.contiguous(),
        tokens,
    )
    return out.reshape_as(residual)


def custom_mlp_branch(x, shift, scale, gate, w1, b1, w2, b2, eps):
    h = fused_adaln_linear_gelu(x, shift, scale, _linear_from_tensors(w1, b1), eps)
    return fused_linear_gated_residual(h, x, gate, _linear_from_tensors(w2, b2))


class _linear_from_tensors:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight = weight
        self.bias = bias


def make_base(batch: int, tokens: int, dim: int, out_dim: int, seed: int) -> tuple[torch.Tensor, ...]:
    x = uniform_bf16((batch, tokens, dim), seed, -2.0, 2.0).requires_grad_(True)
    shift = uniform_bf16((batch, dim), seed + 1, -0.5, 0.5).requires_grad_(True)
    scale = uniform_bf16((batch, dim), seed + 2, -0.25, 0.25).requires_grad_(True)
    w = uniform_bf16((out_dim, dim), seed + 3, -0.02, 0.02).requires_grad_(True)
    b = uniform_bf16((out_dim,), seed + 4, -0.02, 0.02).requires_grad_(True)
    grad = uniform_bf16((batch, tokens, out_dim), seed + 5, -1.0, 1.0)
    return x, shift, scale, w, b, grad


def clone_ln_group(group: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    *inputs, grad = group
    return tuple(clone_requires(t) for t in inputs) + (grad.detach().clone(),)


def make_residual_base(batch: int, tokens: int, dim: int, seed: int) -> tuple[torch.Tensor, ...]:
    h = uniform_bf16((batch, tokens, dim), seed, -2.0, 2.0).requires_grad_(True)
    residual = uniform_bf16((batch, tokens, dim), seed + 1, -2.0, 2.0).requires_grad_(True)
    gate = uniform_bf16((batch, dim), seed + 2, -0.5, 0.5).requires_grad_(True)
    w = uniform_bf16((dim, dim), seed + 3, -0.02, 0.02).requires_grad_(True)
    b = uniform_bf16((dim,), seed + 4, -0.02, 0.02).requires_grad_(True)
    grad = uniform_bf16((batch, tokens, dim), seed + 5, -1.0, 1.0)
    return h, residual, gate, w, b, grad


def make_mlp_base(batch: int, tokens: int, dim: int, seed: int) -> tuple[torch.Tensor, ...]:
    hidden = dim * 4
    x = uniform_bf16((batch, tokens, dim), seed, -2.0, 2.0).requires_grad_(True)
    shift = uniform_bf16((batch, dim), seed + 1, -0.5, 0.5).requires_grad_(True)
    scale = uniform_bf16((batch, dim), seed + 2, -0.25, 0.25).requires_grad_(True)
    gate = uniform_bf16((batch, dim), seed + 3, -0.5, 0.5).requires_grad_(True)
    w1 = uniform_bf16((hidden, dim), seed + 4, -0.02, 0.02).requires_grad_(True)
    b1 = uniform_bf16((hidden,), seed + 5, -0.02, 0.02).requires_grad_(True)
    w2 = uniform_bf16((dim, hidden), seed + 6, -0.02, 0.02).requires_grad_(True)
    b2 = uniform_bf16((dim,), seed + 7, -0.02, 0.02).requires_grad_(True)
    grad = uniform_bf16((batch, tokens, dim), seed + 8, -1.0, 1.0)
    return x, shift, scale, gate, w1, b1, w2, b2, grad


def check_close(label: str, a: torch.Tensor, b: torch.Tensor, detailed: bool = False) -> None:
    diff = (a.float() - b.float()).abs()
    print(f"  correctness {label}: max={diff.max().item():.6g} mean={diff.mean().item():.6g}", flush=True)
    if not detailed:
        return
    flat = diff.flatten()
    for threshold in (0.125, 0.5, 1.0):
        count = (flat > threshold).sum().item()
        print(f"    count>|{threshold}|: {count}/{flat.numel()}", flush=True)
    top_k = min(8, flat.numel())
    vals, idxs = torch.topk(flat, top_k)
    a_flat = a.flatten()
    b_flat = b.flatten()
    for rank, (val, idx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        coord = tuple(int(v) for v in torch.unravel_index(torch.tensor(idx, device=diff.device), diff.shape))
        print(
            f"    top{rank}: idx={coord} diff={val:.6g} custom={a_flat[idx].float().item():.6g} torch={b_flat[idx].float().item():.6g}",
            flush=True,
        )


def detailed_correctness(batch: int, tokens: int, dim: int, eps: float) -> None:
    print(f"\n=== Detailed correctness B={batch} T={tokens} D={dim} ===", flush=True)
    if (batch * tokens) % 128 != 0:
        print("SKIP: current custom GEMM kernels require rows=B*T divisible by 128", flush=True)
        return

    cases = [
        (
            "pre_qkv_ln_adaln_linear",
            torch_ln_linear,
            custom_ln_linear,
            make_base(batch, tokens, dim, 3 * dim, 51000 + tokens + batch),
            eps,
            ("x", "shift", "scale", "w", "b"),
        ),
        (
            "mlp_fc1_ln_adaln_linear_gelu",
            torch_ln_linear_gelu,
            custom_ln_linear_gelu,
            make_base(batch, tokens, dim, 4 * dim, 52000 + tokens + batch),
            eps,
            ("x", "shift", "scale", "w", "b"),
        ),
        (
            "post_linear_gated_residual",
            torch_linear_gated_residual,
            custom_linear_gated_residual,
            make_residual_base(batch, tokens, dim, 53000 + tokens + batch),
            None,
            ("h", "residual", "gate", "w", "b"),
        ),
        (
            "full_mlp_branch",
            torch_mlp_branch,
            custom_mlp_branch,
            make_mlp_base(batch, tokens, dim, 54000 + tokens + batch),
            eps,
            ("x", "shift", "scale", "gate", "w1", "b1", "w2", "b2"),
        ),
    ]

    for label, torch_fn, custom_fn, base_group, case_eps, grad_names in cases:
        print(f"\n{label}", flush=True)
        torch_group = clone_ln_group(base_group)
        custom_group = clone_ln_group(base_group)
        if case_eps is None:
            torch_call = torch_fn
            custom_call = custom_fn
        elif len(base_group) == 6:
            def torch_call(x, shift, scale, w, b, fn=torch_fn, eps_value=case_eps):
                return fn(x, shift, scale, w, b, eps_value)

            def custom_call(x, shift, scale, w, b, fn=custom_fn, eps_value=case_eps):
                return fn(x, shift, scale, w, b, eps_value)
        else:
            def torch_call(x, shift, scale, gate, w1, b1, w2, b2, fn=torch_fn, eps_value=case_eps):
                return fn(x, shift, scale, gate, w1, b1, w2, b2, eps_value)

            def custom_call(x, shift, scale, gate, w1, b1, w2, b2, fn=custom_fn, eps_value=case_eps):
                return fn(x, shift, scale, gate, w1, b1, w2, b2, eps_value)

        torch_out = torch_call(*torch_group[:-1])
        custom_out = custom_call(*custom_group[:-1])
        check_close(f"{label} output", custom_out, torch_out, detailed=True)
        torch_out.backward(torch_group[-1])
        custom_out.backward(custom_group[-1])
        for name, custom_tensor, torch_tensor in zip(grad_names, custom_group[:-1], torch_group[:-1]):
            check_close(f"{label} d{name}", custom_tensor.grad, torch_tensor.grad, detailed=True)


def compare_case(
    label: str,
    torch_fn: Callable[..., torch.Tensor],
    custom_fn: Callable[..., torch.Tensor],
    base_group: tuple[torch.Tensor, ...],
    eps: float | None,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    torch_group = clone_ln_group(base_group)
    compile_group = clone_ln_group(base_group)
    custom_group = clone_ln_group(base_group)
    custom_compile_group = clone_ln_group(base_group)

    if eps is None:
        torch_call = torch_fn
        custom_call = custom_fn
    elif len(base_group) == 6:
        def torch_call(x, shift, scale, w, b):
            return torch_fn(x, shift, scale, w, b, eps)

        def custom_call(x, shift, scale, w, b):
            return custom_fn(x, shift, scale, w, b, eps)
    elif len(base_group) == 9:
        def torch_call(x, shift, scale, gate, w1, b1, w2, b2):
            return torch_fn(x, shift, scale, gate, w1, b1, w2, b2, eps)

        def custom_call(x, shift, scale, gate, w1, b1, w2, b2):
            return custom_fn(x, shift, scale, gate, w1, b1, w2, b2, eps)
    else:
        raise ValueError(f"unsupported group arity for eps case: {len(base_group)}")
    compiled_call = compile_fn(torch_call)
    compiled_custom_call = compile_fn(custom_call)

    with torch.no_grad():
        check_close(label, custom_call(*custom_group[:-1]), torch_call(*torch_group[:-1]))

    compile_result = profile_groups(
        f"{label} torch.compile train",
        [compile_group],
        lambda g: train_step(compiled_call, g),
        warmup=max(3, min(warmup, 10)),
        iters=iters,
        cooldown_s=0.0,
    )
    custom_result = profile_groups(
        f"{label} custom train",
        [custom_group],
        lambda g: train_step(custom_call, g),
        warmup=warmup,
        iters=iters,
        cooldown_s=0.0,
    )
    custom_compile_result = profile_groups(
        f"{label} custom+torch.compile train",
        [custom_compile_group],
        lambda g: train_step(compiled_custom_call, g),
        warmup=max(3, min(warmup, 10)),
        iters=iters,
        cooldown_s=0.0,
    )
    custom_speedup = compile_result.us / custom_result.us
    custom_compile_speedup = compile_result.us / custom_compile_result.us
    print(
        f"RESULT {label}: compile={compile_result.us:.2f}us "
        f"custom={custom_result.us:.2f}us custom_speedup={custom_speedup:.2f}x "
        f"custom_compile={custom_compile_result.us:.2f}us custom_compile_speedup={custom_compile_speedup:.2f}x",
        flush=True,
    )
    return compile_result.us, custom_compile_result.us


def run_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float) -> None:
    run_shape_cases(batch, tokens, dim, warmup, iters, eps, {"pre_qkv", "fc1_gelu", "post_residual", "full_mlp"})


def run_shape_cases(batch: int, tokens: int, dim: int, warmup: int, iters: int, eps: float, cases: set[str]) -> None:
    print(f"\n=== Custom vs torch.compile B={batch} T={tokens} D={dim} ===", flush=True)
    if (batch * tokens) % 128 != 0:
        print(
            f"SKIP B={batch} T={tokens}: current custom GEMM kernels require rows=B*T divisible by 128",
            flush=True,
        )
        return
    if "pre_qkv" in cases:
        compare_case(
            "pre_qkv_ln_adaln_linear",
            torch_ln_linear,
            custom_ln_linear,
            make_base(batch, tokens, dim, 3 * dim, 1000 + tokens),
            eps,
            warmup,
            iters,
        )
    if "fc1_gelu" in cases:
        compare_case(
            "mlp_fc1_ln_adaln_linear_gelu",
            torch_ln_linear_gelu,
            custom_ln_linear_gelu,
            make_base(batch, tokens, dim, 4 * dim, 2000 + tokens),
            eps,
            warmup,
            iters,
        )
    if "post_residual" in cases:
        compare_case(
            "post_linear_gated_residual",
            torch_linear_gated_residual,
            custom_linear_gated_residual,
            make_residual_base(batch, tokens, dim, 3000 + tokens),
            None,
            warmup,
            iters,
        )
    if "full_mlp" in cases:
        compare_case(
            "full_mlp_branch",
            torch_mlp_branch,
            custom_mlp_branch,
            make_mlp_base(batch, tokens, dim, 4000 + tokens),
            eps,
            warmup,
            iters,
        )


def parse_shape_pair(value: str) -> tuple[int, int]:
    for sep in ("x", ":", ","):
        if sep not in value:
            continue
        batch_s, tokens_s = value.split(sep, 1)
        return int(batch_s), int(tokens_s)
    raise argparse.ArgumentTypeError(f"shape must be BxT, B:T, or B,T; got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--batches", type=int, nargs="+", default=None)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 128, 1024])
    parser.add_argument(
        "--shapes",
        type=parse_shape_pair,
        nargs="+",
        default=None,
        help="Explicit batch/token pairs as BxT, B:T, or B,T. Overrides --batch/--batches x --tokens.",
    )
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["pre_qkv", "fc1_gelu", "post_residual", "full_mlp"],
        choices=["pre_qkv", "fc1_gelu", "post_residual", "full_mlp"],
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    shape_pairs = args.shapes
    if shape_pairs is None:
        batches = args.batches or [args.batch or 4]
        shape_pairs = [(batch, tokens) for batch in batches for tokens in args.tokens]
    for batch, tokens in shape_pairs:
        if args.correctness_only:
            detailed_correctness(batch, tokens, args.dim, args.eps)
        else:
            run_shape_cases(batch, tokens, args.dim, args.warmup, args.iters, args.eps, set(args.cases))


if __name__ == "__main__":
    main()
