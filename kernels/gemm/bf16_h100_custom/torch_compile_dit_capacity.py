from __future__ import annotations

import argparse
import gc
import time

import torch

from dit3d_e2e_bench import dit_config, make_group, make_model, spatial_for_tokens, train_step


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def memory_gib(value: int) -> float:
    return value / 2**30


def run_one(model_name: str, tokens: int, batch: int, seed: int, second_step: bool) -> tuple[bool, str]:
    spatial = spatial_for_tokens(tokens)
    cfg = dit_config(model_name)
    clear_cuda()
    started = time.perf_counter()
    label = f"DiT-{model_name} torch.compile no_custom B={batch} T={tokens} spatial={spatial}"
    try:
        group = make_group(batch, cfg["in_channels"], spatial, seed)
        model = make_model(
            model_name,
            fused=False,
            fused_residual=False,
            tk_mlp=False,
            fused_input_projection=False,
            fused_output_projection=False,
            attention_backend="timm",
        )
        model = torch.compile(model)

        train_step(model, group)
        torch.cuda.synchronize()
        first_peak = torch.cuda.max_memory_allocated()
        first_reserved = torch.cuda.max_memory_reserved()

        steady_peak = first_peak
        steady_reserved = first_reserved
        if second_step:
            torch.cuda.reset_peak_memory_stats()
            train_step(model, group)
            torch.cuda.synchronize()
            steady_peak = torch.cuda.max_memory_allocated()
            steady_reserved = torch.cuda.max_memory_reserved()

        elapsed = time.perf_counter() - started
        message = (
            f"{label}: PASS compile_first_peak={memory_gib(first_peak):.2f}GiB "
            f"compile_first_reserved={memory_gib(first_reserved):.2f}GiB "
            f"steady_peak={memory_gib(steady_peak):.2f}GiB "
            f"steady_reserved={memory_gib(steady_reserved):.2f}GiB elapsed={elapsed:.1f}s"
        )
        del model, group
        clear_cuda()
        return True, message
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        clear_cuda()
        return (
            False,
            f"{label}: OOM peak={memory_gib(peak):.2f}GiB reserved={memory_gib(reserved):.2f}GiB error={str(exc).splitlines()[0]}",
        )
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        clear_cuda()
        return (
            False,
            f"{label}: OOM peak={memory_gib(peak):.2f}GiB reserved={memory_gib(reserved):.2f}GiB error={str(exc).splitlines()[0]}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Find torch.compile DiT training batch capacity without TK/custom kernels.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="L")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--batches", nargs="+", type=int, default=[16, 32, 48, 64, 80, 96, 128])
    parser.add_argument("--stop-after-oom", action="store_true")
    parser.add_argument("--second-step", action="store_true", help="Run a second train step after compile to report steady-state peak.")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.init()
    device_name = torch.cuda.get_device_name()
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(
        f"torch.compile DiT capacity no_custom model={args.model} gpu={device_name} total_mem={memory_gib(total_mem):.2f}GiB",
        flush=True,
    )
    print("Batches are tested independently with a freshly compiled model per case.", flush=True)

    overall: list[tuple[int, int | None]] = []
    for tokens in args.tokens:
        print(f"\n=== tokens={tokens} spatial={spatial_for_tokens(tokens)} ===", flush=True)
        max_pass: int | None = None
        for batch in args.batches:
            ok, message = run_one(args.model, tokens, batch, seed=88000 + tokens + batch, second_step=args.second_step)
            print(message, flush=True)
            if ok:
                max_pass = batch
            elif args.stop_after_oom:
                break
        overall.append((tokens, max_pass))
        print(f"SUMMARY tokens={tokens}: max_passing_batch={max_pass}", flush=True)

    print("\n=== capacity summary ===", flush=True)
    for tokens, max_pass in overall:
        print(f"tokens={tokens} max_passing_batch={max_pass}", flush=True)


if __name__ == "__main__":
    main()
