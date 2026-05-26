from __future__ import annotations

import argparse

from dit3d_e2e_bench import tk_layernorm_adaln_backward_op, tk_layernorm_adaln_op
from harness import run_fused_backward, run_fused_forward
from tk_bench import print_bench, profile_groups, uniform_bf16


def make_groups(batch: int, tokens: int, dim: int):
    rows = batch * tokens
    x = uniform_bf16((rows, dim), 81000, -2.0, 2.0)
    shift = uniform_bf16((batch, dim), 81001, -0.5, 0.5)
    scale = uniform_bf16((batch, dim), 81002, -0.25, 0.25)
    grad = uniform_bf16((rows, dim), 81003, -1.0, 1.0)
    _, mean, rstd = run_fused_forward(x, shift, scale, tokens, 1e-6)
    return [(x, shift, scale, grad, mean, rstd)]


def run_shape(batch: int, tokens: int, dim: int, warmup: int, iters: int):
    label = f"B{batch}T{tokens}D{dim}"
    groups = make_groups(batch, tokens, dim)
    print(f"\nRaw pybind vs torch.library custom_op {label}", flush=True)
    results = [
        profile_groups(
            f"{label} raw pybind forward",
            groups,
            lambda g: run_fused_forward(g[0], g[1], g[2], tokens, 1e-6),
            warmup=warmup,
            iters=iters,
        ),
        profile_groups(
            f"{label} custom_op forward",
            groups,
            lambda g: tk_layernorm_adaln_op(g[0], g[1], g[2], tokens, 1e-6),
            warmup=warmup,
            iters=iters,
        ),
        profile_groups(
            f"{label} raw pybind backward",
            groups,
            lambda g: run_fused_backward(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=warmup,
            iters=iters,
        ),
        profile_groups(
            f"{label} custom_op backward",
            groups,
            lambda g: tk_layernorm_adaln_backward_op(g[3], g[0], g[2], g[4], g[5], tokens),
            warmup=warmup,
            iters=iters,
        ),
    ]
    for result in results:
        print_bench(result)
    by_name = {result.name: result for result in results}
    raw_fwd = by_name[f"{label} raw pybind forward"]
    op_fwd = by_name[f"{label} custom_op forward"]
    raw_bwd = by_name[f"{label} raw pybind backward"]
    op_bwd = by_name[f"{label} custom_op backward"]
    print(f"RESULT {label} custom_op_vs_raw_forward={raw_fwd.us / op_fwd.us:.3f}x", flush=True)
    print(f"RESULT {label} custom_op_vs_raw_backward={raw_bwd.us / op_bwd.us:.3f}x", flush=True)


def parse_shape(text: str) -> tuple[int, int]:
    batch, tokens = text.lower().replace("b", "").replace("t", "").split("x")
    return int(batch), int(tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=["64x1024", "16x4096"])
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        run_shape(batch, tokens, args.dim, args.warmup, args.iters)


if __name__ == "__main__":
    main()
