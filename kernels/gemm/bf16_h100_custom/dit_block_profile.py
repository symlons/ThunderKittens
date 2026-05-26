from __future__ import annotations

import argparse
import gc

import torch

from dit3d_e2e_bench import DiTBlock, dit_config
from tk_bench import input_group_count, print_bench, profile_groups, uniform_bf16


def parse_shape_pair(value: str) -> tuple[int, int]:
    for sep in ("x", ":", ","):
        if sep not in value:
            continue
        batch_s, tokens_s = value.split(sep, 1)
        return int(batch_s), int(tokens_s)
    raise argparse.ArgumentTypeError(f"shape must be BxT, B:T, or B,T; got {value!r}")


def zero_grads(items: tuple[torch.Tensor, ...], module: torch.nn.Module) -> None:
    for tensor in items:
        if tensor.grad is not None:
            tensor.grad = None
    for param in module.parameters():
        if param.grad is not None:
            param.grad = None


def block_step(block: torch.nn.Module, group: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    x, c, grad = group
    out = block(x, c)
    out.backward(grad)
    zero_grads((x, c), block)


def make_block(
    model_name: str,
    *,
    fused: bool,
    fused_residual: bool,
    tk_mlp: bool,
    fused_input_projection: bool,
    fused_output_projection: bool,
    attention_backend: str,
) -> DiTBlock:
    cfg = dit_config(model_name)
    block = DiTBlock(
        cfg["hidden_size"],
        cfg["num_heads"],
        fused_adaln_enabled=fused,
        fused_residual_enabled=fused_residual,
        tk_mlp_enabled=tk_mlp,
        fused_input_projection_enabled=fused_input_projection,
        fused_output_projection_enabled=fused_output_projection,
        attention_backend=attention_backend,
    )
    return block.cuda().to(torch.bfloat16).train()


def clone_group(group: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, c, grad = group
    return x.detach().clone().requires_grad_(True), c.detach().clone().requires_grad_(True), grad.detach().clone()


def make_groups(batch: int, tokens: int, dim: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    bytes_per_group = (batch * tokens * dim * 2 * 2) + (batch * dim * 2)
    groups_n = min(input_group_count(bytes_per_group), 4)
    groups = []
    for idx in range(groups_n):
        group_seed = seed + idx * 10
        x = uniform_bf16((batch, tokens, dim), group_seed, -1.0, 1.0).requires_grad_(True)
        c = uniform_bf16((batch, dim), group_seed + 1, -1.0, 1.0).requires_grad_(True)
        grad = uniform_bf16((batch, tokens, dim), group_seed + 2, -1.0, 1.0)
        groups.append((x, c, grad))
    return groups


def check_variant(
    label: str,
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    group: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    ref_group = clone_group(group)
    cand_group = clone_group(group)
    with torch.no_grad():
        ref = reference(ref_group[0], ref_group[1])
        out = candidate(cand_group[0], cand_group[1])
    diff = (out.float() - ref.float()).abs()
    print(
        f"  correctness {label}: max={diff.max().item():.6g} mean={diff.mean().item():.6g}",
        flush=True,
    )


def profile_shape(
    model_name: str,
    batch: int,
    tokens: int,
    warmup: int,
    iters: int,
    variants: list[str],
) -> None:
    cfg = dit_config(model_name)
    dim = cfg["hidden_size"]
    groups = make_groups(batch, tokens, dim, 98000 + batch + tokens)
    print(f"\n=== DiTBlock-{model_name} B={batch} T={tokens} D={dim} groups={len(groups)} ===", flush=True)

    base = make_block(
        model_name,
        fused=False,
        fused_residual=False,
        tk_mlp=False,
        fused_input_projection=False,
        fused_output_projection=False,
        attention_backend="timm",
    )
    variant_specs = {
        "eager": dict(fused=False, fused_residual=False, tk_mlp=False, fused_input_projection=False, fused_output_projection=False),
        "compile": dict(fused=False, fused_residual=False, tk_mlp=False, fused_input_projection=False, fused_output_projection=False),
        "custom_adaln_residual_compile": dict(fused=True, fused_residual=True, tk_mlp=False, fused_input_projection=False, fused_output_projection=False),
        "custom_full_compile": dict(fused=True, fused_residual=True, tk_mlp=True, fused_input_projection=True, fused_output_projection=True),
    }
    results: dict[str, float] = {}
    for variant in variants:
        if variant not in variant_specs:
            raise ValueError(f"unknown variant: {variant}")
        block = make_block(model_name, attention_backend="timm", **variant_specs[variant])
        block.load_state_dict(base.state_dict(), strict=False)
        if variant != "eager":
            check_variant(variant, base, block, groups[0])
        run_block: torch.nn.Module = torch.compile(block) if variant != "eager" else block
        try:
            result = profile_groups(
                f"DiTBlock-{model_name} B{batch} T{tokens} {variant} train",
                groups,
                lambda g, current=run_block: block_step(current, g),
                warmup=max(1, min(2, warmup)) if variant != "eager" else warmup,
                iters=iters,
                cooldown_s=0.0,
            )
            print_bench(result)
            results[variant] = result.us
        except torch.cuda.OutOfMemoryError as exc:
            print(f"DiTBlock-{model_name} B{batch} T{tokens} {variant}: SKIP OOM ({exc})", flush=True)
        finally:
            del run_block, block
            gc.collect()
            torch.cuda.empty_cache()
    if "compile" in results:
        compile_us = results["compile"]
        for variant, us in results.items():
            if variant == "compile":
                continue
            print(
                f"RESULT DiTBlock-{model_name} B{batch} T{tokens} {variant}: "
                f"compile={compile_us:.2f}us variant={us:.2f}us speedup={compile_us / us:.2f}x",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile one DiTBlock train step without full DiT E2E training.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="L")
    parser.add_argument("--shapes", type=parse_shape_pair, nargs="+", default=[(64, 1024), (80, 1024), (16, 4096), (20, 4096)])
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["eager", "compile", "custom_adaln_residual_compile", "custom_full_compile"],
        choices=["eager", "compile", "custom_adaln_residual_compile", "custom_full_compile"],
    )
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    print(f"DiTBlock profile gpu={torch.cuda.get_device_name()} model={args.model}", flush=True)
    for batch, tokens in args.shapes:
        profile_shape(args.model, batch, tokens, args.warmup, args.iters, args.variants)


if __name__ == "__main__":
    main()
