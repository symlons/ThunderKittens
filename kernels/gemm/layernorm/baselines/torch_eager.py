import argparse
from functools import partial
from typing import cast

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp

from bench_common import (
    DEFAULT_SHAPES,
    device_hbm_used_bytes,
    make_bwd_case,
    make_deploy_case,
    make_fwd_bwd_case,
    make_fwd_case,
    make_input_groups,
    parse_dtype,
    parse_shape,
    prompt_choice,
    prompt_int,
    prompt_shapes,
    run_profile,
    should_prompt,
)


MLP_RATIO = 4
MODES = ("fwd", "deploy", "bwd", "fwd-bwd")


def make_mlp(dim: int, hidden_dim: int, dtype: torch.dtype, device: torch.device, train: bool) -> Mlp:
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


def build_profile_case(
    mode: str,
    mlp: Mlp,
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6, help=argparse.SUPPRESS)
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if should_prompt():
        print("Torch eager MLP benchmark setup. Press Enter to accept defaults.\n")
        args.mode = prompt_choice("mode", [*MODES, "all"], args.mode)
        args.model_state = prompt_choice("model state", ["eval", "train"], args.model_state)
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
    for shape in args.shapes:
        batch, tokens = parse_shape(shape)
        device = torch.device("cuda")
        hidden_dim = int(args.dim * MLP_RATIO)
        torch.manual_seed(args.seed)

        hbm_used_before, _, total_bytes = device_hbm_used_bytes(device)
        mlp = make_mlp(args.dim, hidden_dim, dtype, device, train=args.model_state == "train")
        groups, l2_bytes, num_groups = make_input_groups((batch, tokens, args.dim), dtype, device, args.seed)

        for mode in modes:
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
            run_profile(
                case,
                op_name="timm_mlp",
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
