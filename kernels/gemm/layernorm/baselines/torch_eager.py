import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable, cast

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
    parse_optional_dtype,
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
    uniform_tensor,
)


MLP_RATIO = 4
MODES = PROFILE_MODES


ModelBuilder = Callable[[int, int, torch.dtype, torch.device, bool], nn.Module]
GroupBuilder = Callable[[int, int, int, torch.dtype, torch.device, int], tuple[list[object], int, int]]
WorkloadBuilder = Callable[[str, int, int, int, int, torch.dtype], list[WorkloadPart]]
FwdCostBuilder = Callable[[int, int, int, int, torch.dtype], tuple[float, int]]


@dataclass(frozen=True)
class ModelVariant:
    name: str
    make_model: ModelBuilder
    make_groups: GroupBuilder
    workload_parts: WorkloadBuilder
    fwd_cost: FwdCostBuilder

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            act_layer=cast(type[nn.GELU], partial(nn.GELU, approximate="tanh")),
            drop=0,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _, _, _, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        return x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))


def make_mlp(dim: int, hidden_dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> nn.Module:
    model = AdaLNMLPBlock(dim, hidden_dim).to(device=device, dtype=dtype)
    model.train(train)
    return model


def make_layernorm(dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> nn.Module:
    model = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).to(device=device, dtype=dtype)
    model.train(train)
    return model


def build_adaln_mlp(dim: int, hidden_dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> nn.Module:
    return make_mlp(dim, hidden_dim, dtype, device, train)


def build_layernorm(dim: int, hidden_dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> nn.Module:
    del hidden_dim
    return make_layernorm(dim, dtype, device, train)


def make_adaln_input_groups(
    batch: int,
    tokens: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int]:
    x_groups, l2_bytes, num_groups = make_input_groups((batch, tokens, dim), dtype, device, seed)
    c_groups = [
        uniform_tensor((batch, dim), dtype=dtype, device=device, seed=seed + 10_000 + group_idx)
        for group_idx in range(num_groups)
    ]
    return list(zip(x_groups, c_groups)), l2_bytes, num_groups


def make_model_input_groups(
    batch: int,
    tokens: int,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[list[object], int, int]:
    groups, l2_bytes, num_groups = make_input_groups((batch, tokens, dim), dtype, device, seed)
    return list(groups), l2_bytes, num_groups


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


def layernorm_flops(batch: int, tokens: int, dim: int) -> float:
    return float(batch * tokens * dim * 5)


def estimated_layernorm_hbm_bytes(batch: int, tokens: int, dim: int, dtype: torch.dtype) -> int:
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    return 2 * batch * tokens * dim * elem_bytes


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


def layernorm_workload_parts(
    mode: str,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> list[WorkloadPart]:
    del hidden_dim
    fwd_parts = [WorkloadPart("layernorm fwd", layernorm_flops(batch, tokens, dim), estimated_layernorm_hbm_bytes(batch, tokens, dim, dtype))]
    bwd_parts = [WorkloadPart("layernorm bwd", 2.0 * layernorm_flops(batch, tokens, dim), 2 * estimated_layernorm_hbm_bytes(batch, tokens, dim, dtype))]
    if mode == "fwd" or mode == "deploy":
        return fwd_parts
    if mode == "bwd":
        return bwd_parts
    if mode == "fwd-bwd":
        return fwd_parts + bwd_parts
    raise ValueError(f"unsupported mode: {mode}")


def adaln_mlp_fwd_cost(batch: int, tokens: int, dim: int, hidden_dim: int, dtype: torch.dtype) -> tuple[float, int]:
    return matmul_flops(batch, tokens, dim, hidden_dim), estimated_fwd_hbm_bytes(batch, tokens, dim, hidden_dim, dtype)


def layernorm_fwd_cost(batch: int, tokens: int, dim: int, hidden_dim: int, dtype: torch.dtype) -> tuple[float, int]:
    del hidden_dim
    return layernorm_flops(batch, tokens, dim), estimated_layernorm_hbm_bytes(batch, tokens, dim, dtype)


MODEL_REGISTRY: dict[str, ModelVariant] = {
    "adaln-mlp": ModelVariant(
        name="adaln-mlp",
        make_model=build_adaln_mlp,
        make_groups=make_adaln_input_groups,
        workload_parts=architecture_workload_parts,
        fwd_cost=adaln_mlp_fwd_cost,
    ),
    "layernorm": ModelVariant(
        name="layernorm",
        make_model=build_layernorm,
        make_groups=make_model_input_groups,
        workload_parts=layernorm_workload_parts,
        fwd_cost=layernorm_fwd_cost,
    ),
}
MODEL_VARIANTS = tuple(MODEL_REGISTRY)


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
    groups: list[object],
    *,
    mode: str,
    warmup: int,
    iters: int,
    autocast_dtype: torch.dtype | None,
) -> list[MeasuredPart]:
    mlp = getattr(mlp, "_orig_mod", mlp)
    context = torch.inference_mode if mode == "deploy" else torch.no_grad

    def maybe_autocast(x: torch.Tensor):
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=x.device.type, dtype=autocast_dtype)

    if isinstance(mlp, nn.LayerNorm):
        def layernorm(x: torch.Tensor) -> torch.Tensor:
            with context(), maybe_autocast(x):
                return mlp(x)

        return [MeasuredPart("layernorm fwd", timed_component_us(layernorm, groups, warmup=warmup, iters=iters))]

    timm_mlp = mlp.mlp
    drop1 = getattr(timm_mlp, "drop1", nn.Identity())
    drop2 = getattr(timm_mlp, "drop2", nn.Identity())
    norm = getattr(timm_mlp, "norm", nn.Identity())

    with context(), maybe_autocast(groups[0][0]):
        norm_inputs = groups
        fc1_inputs = []
        for x, c in groups:
            _, _, _, shift_mlp, scale_mlp, _ = mlp.adaLN_modulation(c).chunk(6, dim=1)
            fc1_inputs.append(modulate(mlp.norm2(x), shift_mlp, scale_mlp))
        act_inputs = [timm_mlp.fc1(x) for x in fc1_inputs]
        fc2_inputs = [norm(drop1(timm_mlp.act(x))) for x in act_inputs]

    def norm_mod(group: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, c = group
        with context(), maybe_autocast(x):
            _, _, _, shift_mlp, scale_mlp, _ = mlp.adaLN_modulation(c).chunk(6, dim=1)
            return modulate(mlp.norm2(x), shift_mlp, scale_mlp)

    def fc1(x: torch.Tensor) -> torch.Tensor:
        with context(), maybe_autocast(x):
            return timm_mlp.fc1(x)

    def act_drop(x: torch.Tensor) -> torch.Tensor:
        with context(), maybe_autocast(x):
            return norm(drop1(timm_mlp.act(x)))

    def fc2(x: torch.Tensor) -> torch.Tensor:
        with context(), maybe_autocast(x):
            return drop2(timm_mlp.fc2(x))

    return [
        MeasuredPart("norm/adLN/mod", timed_component_us(norm_mod, norm_inputs, warmup=warmup, iters=iters)),
        MeasuredPart("fc1 fwd", timed_component_us(fc1, fc1_inputs, warmup=warmup, iters=iters)),
        MeasuredPart("gelu/drop/norm", timed_component_us(act_drop, act_inputs, warmup=warmup, iters=iters)),
        MeasuredPart("fc2 fwd", timed_component_us(fc2, fc2_inputs, warmup=warmup, iters=iters)),
    ]


def print_architecture_breakdown(
    mlp: nn.Module,
    groups: list[object],
    *,
    model_variant: ModelVariant,
    mode: str,
    shape: str,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
    warmup: int,
    iters: int,
) -> None:
    traffic_dtype = autocast_dtype or dtype
    print_architecture_workload(
        op_name="timm_mlp",
        mode=mode,
        shape=shape,
        dim=dim,
        dtype=dtype,
        autocast_dtype=autocast_dtype,
        parts=model_variant.workload_parts(
            mode,
            batch=batch,
            tokens=tokens,
            dim=dim,
            hidden_dim=hidden_dim,
            dtype=traffic_dtype,
        ),
    )
    if mode == "fwd" or mode == "deploy":
        print_architecture_measurements(
            op_name="timm_mlp",
            mode=mode,
            shape=shape,
            dim=dim,
            dtype=dtype,
            autocast_dtype=autocast_dtype,
            parts=measure_fwd_architecture_parts(
                mlp,
                groups,
                mode=mode,
                warmup=warmup,
                iters=iters,
                autocast_dtype=autocast_dtype,
            ),
        )
    else:
        print("  measured architecture component timing: unavailable for backward modes")


def build_profile_case(
    model_variant: ModelVariant,
    mode: str,
    mlp: nn.Module,
    groups: list[torch.Tensor],
    *,
    batch: int,
    tokens: int,
    dim: int,
    hidden_dim: int,
    dtype: torch.dtype,
    traffic_dtype: torch.dtype,
    autocast_dtype: torch.dtype | None,
    eps: float,
    check_correctness: bool,
):
    fwd_flops, fwd_hbm_bytes = model_variant.fwd_cost(batch, tokens, dim, hidden_dim, traffic_dtype)

    if mode == "fwd":
        return make_fwd_case(
            mlp,
            groups,
            mode=mode,
            flops=fwd_flops,
            hbm_bytes=fwd_hbm_bytes,
            eps=eps,
            check_correctness=check_correctness,
            autocast_dtype=autocast_dtype,
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
            autocast_dtype=autocast_dtype,
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
            autocast_dtype=autocast_dtype,
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
            autocast_dtype=autocast_dtype,
        )
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="adaln-mlp")
    parser.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], default="fp32")
    parser.add_argument("--autocast", choices=["off", "none", "bf16", "bfloat16", "fp16", "float16"], default="bf16")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="fwd")
    parser.add_argument("--model-state", choices=["eval", "train"], default="eval")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--compile-fixed-shapes", action="store_true")
    parser.add_argument("--compare-baseline", choices=["none", "eager", "compile"], default="none")
    parser.add_argument("--baseline-compile-backend", default="inductor")
    parser.add_argument("--baseline-compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--baseline-compile-fullgraph", action="store_true")
    parser.add_argument("--baseline-compile-dynamic", choices=["auto", "true", "false"], default="auto")
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
        args.model_variant = prompt_choice("model variant", list(MODEL_VARIANTS), args.model_variant)
        args.mode = prompt_choice("mode", [*MODES, "all"], args.mode)
        args.model_state = prompt_choice("model state", ["eval", "train"], args.model_state)
        prompt_compile_options(args)
        args.compare_baseline = prompt_choice("compare baseline", ["none", "eager", "compile"], args.compare_baseline)
        prompt_diagnostic_options(args)
        args.architecture_breakdown = prompt_choice("architecture workload breakdown", ["off", "on"], "off") == "on"
        args.dtype = prompt_choice("dtype", ["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"], args.dtype)
        args.autocast = prompt_choice("autocast", ["bf16", "bfloat16", "fp16", "float16", "off", "none"], args.autocast)
        args.compile_fixed_shapes = prompt_choice("compile fixed shapes", ["off", "on"], "off") == "on"
        args.shapes = prompt_shapes(args.shapes)
        args.dim = prompt_int("dim", args.dim)
        args.warmup = prompt_int("warmup iterations", args.warmup)
        args.iters = prompt_int("profile iterations", args.iters)
        correctness = prompt_choice("correctness", ["run", "skip"], "run")
        args.skip_correctness = correctness == "skip"

    if not torch.cuda.is_available():
        raise RuntimeError("torch_eager.py requires a CUDA device")

    dtype = parse_dtype(args.dtype)
    autocast_dtype = parse_optional_dtype(args.autocast)
    traffic_dtype = autocast_dtype or dtype
    compile_dynamic = "false" if args.compile_fixed_shapes else args.compile_dynamic
    baseline_compile_dynamic = "false" if args.compile_fixed_shapes else args.baseline_compile_dynamic
    modes = MODES if args.mode == "all" else (args.mode,)
    model_variant = MODEL_REGISTRY[args.model_variant]
    current_variant = f"compile_{args.compile_mode}" if args.compile else "eager"
    if args.compare_baseline == current_variant:
        print(f"warning: compare baseline is also {current_variant}; speedup should be near 1x")
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        device = torch.device("cuda")
        hidden_dim = int(args.dim * MLP_RATIO)
        torch.manual_seed(args.seed)

        hbm_used_before, _, total_bytes = device_hbm_used_bytes(device)
        mlp = model_variant.make_model(args.dim, hidden_dim, dtype, device, args.model_state == "train")
        if args.compile:
            if args.compile_fixed_shapes:
                torch._dynamo.reset()
            mlp = compile_module(
                mlp,
                backend=args.compile_backend,
                mode=args.compile_mode,
                fullgraph=args.compile_fullgraph,
                dynamic=compile_dynamic,
            )
        baseline_mlp = None
        baseline_name = None
        if args.compare_baseline != "none":
            baseline_mlp = model_variant.make_model(args.dim, hidden_dim, dtype, device, args.model_state == "train")
            baseline_mlp.load_state_dict(getattr(mlp, "_orig_mod", mlp).state_dict())
            if args.compare_baseline == "compile":
                if args.compile_fixed_shapes:
                    torch._dynamo.reset()
                baseline_mlp = compile_module(
                    baseline_mlp,
                    backend=args.baseline_compile_backend,
                    mode=args.baseline_compile_mode,
                    fullgraph=args.baseline_compile_fullgraph,
                    dynamic=baseline_compile_dynamic,
                )
                baseline_name = f"compile_{args.baseline_compile_mode}"
            else:
                baseline_name = args.compare_baseline
        groups, l2_bytes, num_groups = model_variant.make_groups(batch, tokens, args.dim, dtype, device, args.seed)

        for mode in modes:
            if args.architecture_breakdown:
                print_architecture_breakdown(
                    mlp,
                    groups,
                    model_variant=model_variant,
                    mode=mode,
                    shape=shape,
                    batch=batch,
                    tokens=tokens,
                    dim=args.dim,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    autocast_dtype=autocast_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            case = build_profile_case(
                model_variant,
                mode,
                mlp,
                groups,
                batch=batch,
                tokens=tokens,
                dim=args.dim,
                hidden_dim=hidden_dim,
                dtype=dtype,
                traffic_dtype=traffic_dtype,
                autocast_dtype=autocast_dtype,
                eps=args.eps,
                check_correctness=not args.skip_correctness,
            )
            baseline_case = None
            if baseline_mlp is not None:
                baseline_case = build_profile_case(
                    model_variant,
                    mode,
                    baseline_mlp,
                    groups,
                    batch=batch,
                    tokens=tokens,
                    dim=args.dim,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    traffic_dtype=traffic_dtype,
                    autocast_dtype=autocast_dtype,
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
                baseline_name=baseline_name if baseline_case is not None else None,
                op_name=f"{model_variant.name}_{current_variant}",
                shape=shape,
                dim=args.dim,
                dtype=dtype,
                autocast_dtype=autocast_dtype,
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
