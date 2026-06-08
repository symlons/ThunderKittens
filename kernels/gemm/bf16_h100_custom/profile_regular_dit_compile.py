from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

from dit_profile_utils import (
    REGULAR_TIMM_VARIANT,
    configure_torchinductor_cache,
    early_torchinductor_cache_dir,
    parse_spatial,
    resolve_spatial,
)


_COMPILE_CACHE_DIR, _COMPILE_CACHE_DIR_EXPLICIT = early_torchinductor_cache_dir()
configure_torchinductor_cache(_COMPILE_CACHE_DIR, explicit=_COMPILE_CACHE_DIR_EXPLICIT)

import torch
import torch.nn as nn

from dit3d_e2e_bench import dit_config, make_group, make_model


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def zero_grads(model: nn.Module, x: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None


def train_step(model: nn.Module, group: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
    x, t, grad = group
    out = model(x, t)
    out.backward(grad)
    zero_grads(model, x)


def warmup_model(
    label: str,
    model: nn.Module,
    group: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    warmup: int,
) -> None:
    print(f"\nWarmup {label}: {warmup} train steps", flush=True)
    for idx in range(warmup):
        train_step(model, group)
        cuda_sync()
        print(f"  {idx + 1}/{warmup}", flush=True)


def profile_forward(
    label: str,
    model: nn.Module,
    group: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iters: int,
    rows: int,
    trace_dir: Path | None,
) -> None:
    x, t, _ = group
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    cuda_sync()
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(iters):
            with torch.profiler.record_function(f"{label}_forward"):
                model(x, t)
    cuda_sync()
    print(f"\n=== {label} forward only ===", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows), flush=True)
    if trace_dir is not None:
        path = trace_dir / f"{label}_forward.json"
        prof.export_chrome_trace(str(path))
        print(f"trace: {path}", flush=True)


def profile_backward(
    label: str,
    model: nn.Module,
    group: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iters: int,
    rows: int,
    trace_dir: Path | None,
) -> None:
    x, t, grad = group
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for _ in range(max(0, iters - 1)):
        out = model(x, t)
        cuda_sync()
        out.backward(grad)
        cuda_sync()
        zero_grads(model, x)
    out = model(x, t)
    cuda_sync()
    with torch.profiler.profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with torch.profiler.record_function(f"{label}_backward"):
            out.backward(grad)
        cuda_sync()
    zero_grads(model, x)
    print(f"\n=== {label} backward only ===", flush=True)
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=rows), flush=True)
    if trace_dir is not None:
        path = trace_dir / f"{label}_backward.json"
        prof.export_chrome_trace(str(path))
        print(f"trace: {path}", flush=True)


def make_regular_timm_model(model_name: str) -> nn.Module:
    return make_model(model_name, **REGULAR_TIMM_VARIANT)


def summarize_model(model_name: str, batch: int, spatial: tuple[int, int, int]) -> None:
    cfg = dit_config(model_name)
    tokens = spatial[0] * spatial[1] * spatial[2]
    print(
        f"Regular timm DiT-{model_name}: batch={batch} spatial={spatial} "
        f"tokens={tokens} hidden={cfg['hidden_size']} depth={cfg['depth']} "
        f"channels={cfg['in_channels']}",
        flush=True,
    )
    print("Variants: eager torch, torch.compile on the same unfused timm Mlp/Attention model", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile regular timm DiT eager vs torch.compile with separate forward/backward tables."
    )
    parser.add_argument("--model", choices=["S", "L", "XL"], default="S")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--spatial", type=parse_spatial, default=None)
    parser.add_argument("--tokens", type=int, default=None, help="3D token count; 4096 resolves to spatial 16x16x16.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--compile-mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic=True for torch.compile.")
    parser.add_argument("--trace-dir", type=Path, default=None, help="Optional directory for Chrome trace JSON files.")
    parser.add_argument(
        "--compile-cache-dir",
        type=Path,
        default=Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]),
        help="Persistent torch.compile/Inductor cache directory reused across process runs.",
    )
    args = parser.parse_args()

    args.compile_cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"torch.compile cache: {args.compile_cache_dir}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler script expects CUDA.")
    args.spatial = resolve_spatial(args.spatial, args.tokens)
    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)

    summarize_model(args.model, args.batch, args.spatial)
    cfg = dit_config(args.model)
    group = make_group(args.batch, cfg["in_channels"], args.spatial, seed=123000)

    eager = make_regular_timm_model(args.model)
    eager.pos_embed(args.spatial, torch.bfloat16, torch.device("cuda"))
    warmup_model("eager", eager, group, args.warmup)
    profile_forward("eager", eager, group, args.iters, args.rows, args.trace_dir)
    profile_backward("eager", eager, group, args.iters, args.rows, args.trace_dir)

    del eager
    gc.collect()
    torch.cuda.empty_cache()

    compiled_base = make_regular_timm_model(args.model)
    compiled_base.pos_embed(args.spatial, torch.bfloat16, torch.device("cuda"))
    compile_kwargs = {"dynamic": args.dynamic}
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    compiled = torch.compile(compiled_base, **compile_kwargs)
    warmup_model("compile", compiled, group, args.warmup)
    profile_forward("compile", compiled, group, args.iters, args.rows, args.trace_dir)
    profile_backward("compile", compiled, group, args.iters, args.rows, args.trace_dir)


if __name__ == "__main__":
    main()
