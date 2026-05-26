import argparse
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from bench_common import (
    DEFAULT_SHAPES,
    MeasuredPart,
    PROFILE_MODES,
    WorkloadPart,
    compile_module,
    device_hbm_used_bytes,
    export_profiler_trace,
    make_bwd_case,
    make_deploy_case,
    make_fwd_bwd_case,
    make_fwd_case,
    make_input_groups,
    parse_dtype,
    parse_shape,
    prompt_choice,
    prompt_compile_options,
    prompt_diagnostic_options,
    prompt_int,
    prompt_shapes,
    print_architecture_measurements,
    print_architecture_workload,
    run_dynamo_explain,
    run_profile,
    should_prompt,
    trace_path,
)


MLP_RATIO = 4
MODES = PROFILE_MODES


def make_mlp(dim: int, hidden_dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> nn.Module:
    mlp = Mlp(
        in_features=dim,
        hidden_features=hidden_dim,
        act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
        drop=0,
    ).to(device=device, dtype=dtype)
    mlp.train(train)
    return mlp


def matmul_flops(batch: int, tokens: int, dim: int, hidden_dim: int) -> float:
    elems = batch * tokens
    return float(2 * elems * dim * hidden_dim + 2 * elems * hidden_dim * dim)


def estimated_fwd_hbm_bytes(batch: int, tokens: int, dim: int, hidden_dim: int, dtype: torch.dtype) -> int:
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    elems = batch * tokens
    input_bytes = elems * dim * elem_bytes
    hidden_bytes = elems * hidden_dim * elem_bytes
    output_bytes = elems * dim * elem_bytes
    fc1_weight_bytes = dim * hidden_dim * elem_bytes
    fc2_weight_bytes = hidden_dim * dim * elem_bytes

    return input_bytes + fc1_weight_bytes + hidden_bytes + hidden_bytes + fc2_weight_bytes + output_bytes


def linear_fwd_part(name: str, elems: int, in_dim: int, out_dim: int, elem_bytes: int) -> WorkloadPart:
    flops = float(2 * elems * in_dim * out_dim)
    bytes_moved = (elems * in_dim + in_dim * out_dim + elems * out_dim) * elem_bytes
    return WorkloadPart(name=name, flops=flops, bytes=bytes_moved)


def linear_bwd_part(name: str, elems: int, in_dim: int, out_dim: int, elem_bytes: int) -> WorkloadPart:
    flops = float(4 * elems * in_dim * out_dim + elems * out_dim)
    bytes_moved = (
        elems * in_dim
        + elems * out_dim
        + in_dim * out_dim
        + elems * in_dim
        + in_dim * out_dim
        + out_dim
    ) * elem_bytes
    return WorkloadPart(name=name, flops=flops, bytes=bytes_moved)


def gelu_fwd_part(elems: int, hidden_dim: int, elem_bytes: int) -> WorkloadPart:
    hidden_elems = elems * hidden_dim
    return WorkloadPart(name="gelu fwd", flops=float(8 * hidden_elems), bytes=2 * hidden_elems * elem_bytes)


def gelu_bwd_part(elems: int, hidden_dim: int, elem_bytes: int) -> WorkloadPart:
    hidden_elems = elems * hidden_dim
    return WorkloadPart(name="gelu bwd", flops=float(10 * hidden_elems), bytes=3 * hidden_elems * elem_bytes)


def architecture_workload_parts(
    mode: str,
    *,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[WorkloadPart]:
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    elems = batch * tokens

    fwd_parts = [
        linear_fwd_part("fc1 fwd", elems, dim, hidden_dim, elem_bytes),
        gelu_fwd_part(elems, hidden_dim, elem_bytes),
        linear_fwd_part("fc2 fwd", elems, hidden_dim, dim, elem_bytes),
    ]
    bwd_parts = [
        linear_bwd_part("fc2 bwd", elems, hidden_dim, dim, elem_bytes),
        gelu_bwd_part(elems, hidden_dim, elem_bytes),
        linear_bwd_part("fc1 bwd", elems, dim, hidden_dim, elem_bytes),
    ]

    if mode == "fwd" or mode == "deploy":
        return fwd_parts
    if mode == "bwd":
        return bwd_parts
    if mode == "fwd-bwd":
        return fwd_parts + bwd_parts
    raise ValueError(f"unsupported mode: {mode}")


def timed_component_us(fn, groups: list[torch.Tensor], *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for i in range(warmup):
        fn(groups[i % len(groups)])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(groups[i % len(groups)])
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def measure_fwd_architecture_parts(
    mlp: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    warmup: int,
    iters: int,
) -> list[MeasuredPart]:
    mlp = getattr(mlp, "_orig_mod", mlp)
    context = torch.inference_mode if mode == "deploy" else torch.no_grad
    drop1 = getattr(mlp, "drop1", nn.Identity())
    drop2 = getattr(mlp, "drop2", nn.Identity())
    norm = getattr(mlp, "norm", nn.Identity())

    with context():
        fc1_inputs = groups
        act_inputs = [mlp.fc1(group) for group in groups]
        fc2_inputs = [norm(drop1(mlp.act(x))) for x in act_inputs]

    def fc1(x: torch.Tensor) -> torch.Tensor:
        with context():
            return mlp.fc1(x)

    def act_drop(x: torch.Tensor) -> torch.Tensor:
        with context():
            return norm(drop1(mlp.act(x)))

    def fc2(x: torch.Tensor) -> torch.Tensor:
        with context():
            return drop2(mlp.fc2(x))

    return [
        MeasuredPart("fc1 fwd", timed_component_us(fc1, fc1_inputs, warmup=warmup, iters=iters)),
        MeasuredPart("gelu/drop/norm", timed_component_us(act_drop, act_inputs, warmup=warmup, iters=iters)),
        MeasuredPart("fc2 fwd", timed_component_us(fc2, fc2_inputs, warmup=warmup, iters=iters)),
    ]


def print_architecture_breakdown(
    mlp: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    shape: str,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
) -> None:
    print_architecture_workload(
        op_name="timm_mlp",
        mode=mode,
        shape=shape,
        dim=dim,
        dtype=dtype,
        parts=architecture_workload_parts(
            mode,
            batch=batch,
            tokens=tokens,
            dim=dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
        ),
    )
    if mode == "fwd" or mode == "deploy":
        print_architecture_measurements(
            op_name="timm_mlp",
            mode=mode,
            shape=shape,
            dim=dim,
            dtype=dtype,
            parts=measure_fwd_architecture_parts(mlp, groups, mode=mode, warmup=warmup, iters=iters),
        )
    else:
        print("  measured architecture component timing: unavailable for backward modes")


def build_profile_case(
    mode: str,
    mlp: nn.Module,
    groups: list[torch.Tensor],
    *,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    eps: float,
    check_correctness: bool,
):
    fwd_flops = matmul_flops(batch, tokens, dim, hidden_dim)
    fwd_hbm_bytes = estimated_fwd_hbm_bytes(batch, tokens, dim, hidden_dim, dtype)

    if mode == "fwd":
        return make_fwd_case(
            mlp,
            groups,
            mode=mode,
            flops=fwd_flops,
            hbm_bytes=fwd_hbm_bytes,
            eps=eps,
            check_correctness=check_correctness,
        )
    if mode == "deploy":
        return make_deploy_case(
            mlp,
            groups,
            mode=mode,
            flops=fwd_flops,
            hbm_bytes=fwd_hbm_bytes,
            eps=eps,
            check_correctness=check_correctness,
        )
    if mode == "bwd":
        return make_bwd_case(
            mlp,
            groups,
            mode=mode,
            flops=2.0 * fwd_flops,
            hbm_bytes=2 * fwd_hbm_bytes,
            dtype=dtype,
            eps=eps,
            check_correctness=check_correctness,
        )
    if mode == "fwd-bwd":
        return make_fwd_bwd_case(
            mlp,
            groups,
            mode=mode,
            flops=3.0 * fwd_flops,
            hbm_bytes=3 * fwd_hbm_bytes,
            dtype=dtype,
            eps=eps,
            check_correctness=check_correctness,
        )
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="bf16")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="fwd")
    parser.add_argument("--model-state", choices=["eval", "train"], default="eval")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--compare-baseline", choices=["none", "eager", "compile"], default="none")
    parser.add_argument("--dynamo-explain", action="store_true")
    parser.add_argument("--profiler-trace", default=None)
    parser.add_argument("--profiler-warmup", type=int, default=5)
    parser.add_argument("--profiler-active", type=int, default=10)
    parser.add_argument("--architecture-breakdown", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6, help=argparse.SUPPRESS)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if should_prompt():
        print("Torch eager MLP benchmark setup. Press Enter to accept defaults.\n")
        args.mode = prompt_choice("mode", [*MODES, "all"], args.mode)
        args.model_state = prompt_choice("model state", ["eval", "train"], args.model_state)
        prompt_compile_options(args)
        args.compare_baseline = prompt_choice("compare baseline", ["none", "eager", "compile"], args.compare_baseline)
        prompt_diagnostic_options(args)
        args.architecture_breakdown = prompt_choice("architecture workload breakdown", ["off", "on"], "off") == "on"
        args.dtype = prompt_choice("dtype", ["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], args.dtype)
        args.shapes = prompt_shapes(args.shapes)
        args.dim = prompt_int("dim", args.dim)
        args.warmup = prompt_int("warmup iterations", args.warmup)
        args.iters = prompt_int("profile iterations", args.iters)
        correctness = prompt_choice("correctness", ["run", "skip"], "run")
        args.skip_correctness = correctness == "skip"

    if not torch.cuda.is_available():
        raise RuntimeError("torch_eager.py requires a CUDA device")

    dtype = parse_dtype(args.dtype)
    modes = MODES if args.mode == "all" else (args.mode,)
    current_variant = "compile" if args.compile else "eager"
    if args.compare_baseline == current_variant:
        print(f"warning: compare baseline is also {current_variant}; speedup should be near 1x")
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        device = torch.device("cuda")
        hidden_dim = int(args.dim * MLP_RATIO)
        torch.manual_seed(args.seed)

        hbm_used_before, _, total_bytes = device_hbm_used_bytes(device)
        mlp = make_mlp(args.dim, hidden_dim, dtype, device, train=args.model_state == "train")
        if args.compile:
            mlp = compile_module(
                mlp,
                backend=args.compile_backend,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
                dynamic=args.compile_dynamic,
            )
        baseline_mlp = None
        if args.compare_baseline != "none":
            baseline_mlp = make_mlp(args.dim, hidden_dim, dtype, device, train=args.model_state == "train")
            baseline_mlp.load_state_dict(getattr(mlp, "_orig_mod", mlp).state_dict())
            if args.compare_baseline == "compile":
                baseline_mlp = compile_module(
                    baseline_mlp,
                    backend=args.compile_backend,
                    mode=args.compile_mode,
                    fullgraph=args.compile_fullgraph,
                    dynamic=args.compile_dynamic,
                )
        groups, l2_bytes, num_groups = make_input_groups((batch, tokens, args.dim), dtype, device, args.seed)

        for mode in modes:
            if args.architecture_breakdown:
                print_architecture_breakdown(
                    mlp,
                    groups,
                    mode=mode,
                    shape=shape,
                    batch=batch,
                    tokens=tokens,
                    dim=args.dim,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            case = build_profile_case(
                mode,
                mlp,
                groups,
                batch=batch,
                tokens=tokens,
                dim=args.dim,
                hidden_dim=hidden_dim,
                dtype=dtype,
                eps=args.eps,
                check_correctness=not args.skip_correctness,
            )
            baseline_case = None
            if baseline_mlp is not None:
                baseline_case = build_profile_case(
                    mode,
                    baseline_mlp,
                    groups,
                    batch=batch,
                    tokens=tokens,
                    dim=args.dim,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    eps=args.eps,
                    check_correctness=False,
                )
            if args.dynamo_explain:
                run_dynamo_explain(case)
            if args.profiler_trace:
                export_profiler_trace(
                    case,
                    trace_path(args.profiler_trace, shape=shape, mode=mode, multi=len(args.shapes) * len(modes) > 1),
                    warmup=args.profiler_warmup,
                    active=args.profiler_active,
                )
            run_profile(
                case,
                baseline_case=baseline_case,
                baseline_name=args.compare_baseline if baseline_case is not None else None,
                op_name=f"timm_mlp_{current_variant}",
                shape=shape,
                dim=args.dim,
                dtype=dtype,
                model_state=args.model_state,
                l2_bytes=l2_bytes,
                num_groups=num_groups,
                warmup=args.warmup,
                iters=args.iters,
                hbm_used_before=hbm_used_before,
                total_bytes=total_bytes,
            )


if __name__ == "__main__":
    main()
