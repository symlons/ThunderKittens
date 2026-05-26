import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


COOLDOWN_S = 0.5
DEFAULT_SHAPES = ["64x1024", "80x1024", "16x4096", "20x4096"]
GRAD_SEED_BWD = 17
GRAD_SEED_FWD_BWD = 31


Correctness = tuple[bool, float, float, float, float, float]


@dataclass(frozen=True)
class ProfileCase:
    mode: str
    groups: list[object]
    run: Callable[[object], object]
    output_shape: tuple[int, ...]
    flops: float
    hbm_bytes: int
    correctness: Correctness | None


def parse_shape(text: str) -> tuple[int, int]:
    batch, tokens = text.lower().replace("b", "").replace("t", "").split("x")
    return int(batch), int(tokens)


def parse_dtype(text: str) -> torch.dtype:
    if text == "bf16" or text == "bfloat16":
        return torch.bfloat16
    if text == "fp16" or text == "float16":
        return torch.float16
    if text == "fp32" or text == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {text}")


def prompt_choice(label: str, choices: list[str], default: str) -> str:
    options = ", ".join(choices)
    while True:
        value = input(f"{label} [{default}] ({options}): ").strip()
        if not value:
            return default
        if value in choices:
            return value
        print(f"Please choose one of: {options}")


def prompt_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def prompt_shapes(default: list[str]) -> list[str]:
    default_text = " ".join(default)
    while True:
        value = input(f"shapes [{default_text}]: ").strip()
        shapes = value.split() if value else default
        try:
            for shape in shapes:
                parse_shape(shape)
            return shapes
        except ValueError:
            print("Please enter shapes like: 64x1024 16x4096")


def should_prompt() -> bool:
    return len(sys.argv) == 1


def l2_cache_size_bytes(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).L2_cache_size


def device_hbm_used_bytes(device: torch.device) -> tuple[int, int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes, free_bytes, total_bytes


def input_group_count(input_bytes: int, l2_bytes: int) -> int:
    if input_bytes <= 0:
        raise ValueError("input size must be positive")
    return 1 if input_bytes >= 3 * l2_bytes else int(3 * l2_bytes / input_bytes) + 1


def uniform_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    x = torch.rand(shape, device=device, dtype=torch.float32, generator=gen)
    x = x * (high - low) + low
    return x.to(dtype)


def make_input_groups(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[list[torch.Tensor], int, int]:
    input_bytes = torch.empty((), dtype=dtype).element_size()
    for size in shape:
        input_bytes *= size

    l2_bytes = l2_cache_size_bytes(device)
    num_groups = input_group_count(input_bytes, l2_bytes)
    groups = [uniform_tensor(shape, dtype=dtype, device=device, seed=seed + group_idx) for group_idx in range(num_groups)]
    return groups, l2_bytes, num_groups


def format_bytes(num_bytes: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(num_bytes)
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def print_kv(label: str, value: str, *, indent: int = 2, width: int = 12) -> None:
    print(f"{' ' * indent}{label:<{width}} {value}")


def format_hbm_state(used: int, *, delta: int | None = None, free: int | None = None) -> str:
    details = []
    if delta is not None:
        details.append(f"delta {format_bytes(delta)}")
    if free is not None:
        details.append(f"free {format_bytes(free)}")

    value = format_bytes(used)
    return value if not details else f"{value} ({', '.join(details)})"


def print_hbm_report(
    *,
    total_bytes: int,
    hbm_used_before: int,
    hbm_used_setup: int,
    measured_setup_hbm: int,
    hbm_free_setup: int,
    hbm_used_after_profile: int,
    measured_profile_hbm_delta: int,
    hbm_free_after_profile: int,
) -> None:
    print_kv("HBM device", "")
    print_kv("total", format_bytes(total_bytes), indent=4)
    print_kv("used before", format_hbm_state(hbm_used_before), indent=4)
    print_kv(
        "after setup",
        format_hbm_state(hbm_used_setup, delta=measured_setup_hbm, free=hbm_free_setup),
        indent=4,
    )
    print_kv(
        "after profile",
        format_hbm_state(hbm_used_after_profile, delta=measured_profile_hbm_delta, free=hbm_free_after_profile),
        indent=4,
    )


def print_allocator_report(
    *,
    static_allocated: int,
    static_reserved: int,
    peak_allocated: int,
    peak_reserved: int,
) -> None:
    print_kv("Torch alloc", "")
    print_kv("allocated", format_bytes(static_allocated), indent=4)
    print_kv("reserved", format_bytes(static_reserved), indent=4)
    print_kv("peak alloc", format_bytes(peak_allocated), indent=4)
    print_kv("peak reserv", format_bytes(peak_reserved), indent=4)


def print_benchmark_report(
    *,
    op_name: str,
    mode: str,
    model_state: str,
    shape: str,
    dim: int,
    dtype: torch.dtype,
    l2_bytes: int,
    num_groups: int,
    output_shape: tuple[int, ...],
    us: float,
    tflops: float,
    hbm_tb_s: float,
    hbm_bytes: int,
    total_bytes: int,
    hbm_used_before: int,
    hbm_used_setup: int,
    measured_setup_hbm: int,
    hbm_free_setup: int,
    hbm_used_after_profile: int,
    measured_profile_hbm_delta: int,
    hbm_free_after_profile: int,
    static_allocated: int,
    static_reserved: int,
    peak_allocated: int,
    peak_reserved: int,
    correctness: Correctness | None,
) -> None:
    print(f"\nop={op_name} mode={mode} model={model_state} shape={shape} dim={dim} dtype={dtype}")
    print_kv("l2", f"{l2_bytes / 1024 / 1024:.1f} MiB")
    print_kv("groups", str(num_groups))
    print_kv("output", str(output_shape))
    print_kv("time", f"{us:.2f} us")
    print_kv("TFLOPS", f"{tflops:.2f}")
    print_kv("HBM BW", f"{hbm_tb_s:.2f} TB/s estimated traffic ({format_bytes(hbm_bytes)} / iter)")
    print_hbm_report(
        total_bytes=total_bytes,
        hbm_used_before=hbm_used_before,
        hbm_used_setup=hbm_used_setup,
        measured_setup_hbm=measured_setup_hbm,
        hbm_free_setup=hbm_free_setup,
        hbm_used_after_profile=hbm_used_after_profile,
        measured_profile_hbm_delta=measured_profile_hbm_delta,
        hbm_free_after_profile=hbm_free_after_profile,
    )
    print_allocator_report(
        static_allocated=static_allocated,
        static_reserved=static_reserved,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )
    if correctness is None:
        print_kv("correctness", "SKIPPED")
    else:
        correct, max_diff, mean_diff, rel_diff, atol, rtol = correctness
        print_kv(
            "correctness",
            (
                f"{'PASS' if correct else 'FAIL'} "
                f"max={max_diff:.6g} mean={mean_diff:.6g} rel={rel_diff:.6g} "
                f"atol={atol:g} rtol={rtol:g}"
            ),
        )


def clear_grads(module: nn.Module, inputs: list[torch.Tensor]) -> None:
    for x in inputs:
        x.grad = None
    for param in module.parameters():
        param.grad = None


def grad_snapshot(x: torch.Tensor, module: nn.Module) -> tuple[torch.Tensor, ...]:
    grads = [x.grad.detach().float()]
    grads.extend(param.grad.detach().float() for param in module.parameters() if param.grad is not None)
    return tuple(grads)


def compare_tensors(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], *, eps: float) -> Correctness:
    diffs = [(a - e).abs() for a, e in zip(actual, expected)]
    max_diff = max(diff.max().item() for diff in diffs)
    mean_diff = sum(diff.mean().item() for diff in diffs) / len(diffs)
    ref_max = max(e.abs().max().item() for e in expected)
    rel_diff = max_diff / max(ref_max, 1e-12)
    ok = len(actual) == len(expected) and all(torch.allclose(a, e, atol=eps, rtol=0.0) for a, e in zip(actual, expected))
    return ok, max_diff, mean_diff, rel_diff, eps, 0.0


def self_consistency(module: nn.Module, group: torch.Tensor, *, eps: float) -> Correctness:
    with torch.inference_mode():
        actual = module(group).float()
        expected = module(group).float()
        correctness = compare_tensors((actual,), (expected,), eps=eps)
    del actual, expected
    torch.cuda.synchronize()
    return correctness


def make_fwd_case(
    module: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    flops: float,
    hbm_bytes: int,
    eps: float,
    check_correctness: bool,
) -> ProfileCase:
    correctness = self_consistency(module, groups[0], eps=eps) if check_correctness else None

    def run(group: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return module(group)

    output = run(groups[0])
    return ProfileCase(mode=mode, groups=groups, run=run, output_shape=tuple(output.shape), flops=flops, hbm_bytes=hbm_bytes, correctness=correctness)


def make_deploy_case(
    module: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    flops: float,
    hbm_bytes: int,
    eps: float,
    check_correctness: bool,
) -> ProfileCase:
    correctness = self_consistency(module, groups[0], eps=eps) if check_correctness else None

    def run(group: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return module(group)

    output = run(groups[0])
    return ProfileCase(mode=mode, groups=groups, run=run, output_shape=tuple(output.shape), flops=flops, hbm_bytes=hbm_bytes, correctness=correctness)


def make_bwd_case(
    module: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    flops: float,
    hbm_bytes: int,
    dtype: torch.dtype,
    eps: float,
    check_correctness: bool,
) -> ProfileCase:
    grad_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for group in groups:
        x = group.detach().clone().requires_grad_(True)
        y = module(x)
        grad = uniform_tensor(tuple(y.shape), dtype=dtype, device=group.device, seed=GRAD_SEED_BWD + len(grad_groups))
        grad_groups.append((x, y, grad))

    def run(group: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        x, y, grad = group
        clear_grads(module, [x])
        torch.autograd.backward(y, grad, retain_graph=True, create_graph=False)
        return grad_snapshot(x, module)

    correctness = None
    if check_correctness:
        actual = run(grad_groups[0])
        expected = run(grad_groups[0])
        correctness = compare_tensors(actual, expected, eps=eps)
        clear_grads(module, [grad_groups[0][0]])

    return ProfileCase(
        mode=mode,
        groups=grad_groups,
        run=run,
        output_shape=tuple(grad_groups[0][0].shape),
        flops=flops,
        hbm_bytes=hbm_bytes,
        correctness=correctness,
    )


def make_fwd_bwd_case(
    module: nn.Module,
    groups: list[torch.Tensor],
    *,
    mode: str,
    flops: float,
    hbm_bytes: int,
    dtype: torch.dtype,
    eps: float,
    check_correctness: bool,
) -> ProfileCase:
    grad_groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for group in groups:
        x = group.detach().clone().requires_grad_(True)
        with torch.inference_mode():
            y = module(group)
        grad = uniform_tensor(tuple(y.shape), dtype=dtype, device=group.device, seed=GRAD_SEED_FWD_BWD + len(grad_groups))
        grad_groups.append((x, grad))

    def run(group: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        x, grad = group
        clear_grads(module, [x])
        y = module(x)
        torch.autograd.backward(y, grad, create_graph=False)
        return grad_snapshot(x, module)

    correctness = None
    if check_correctness:
        actual = run(grad_groups[0])
        expected = run(grad_groups[0])
        correctness = compare_tensors(actual, expected, eps=eps)
        clear_grads(module, [grad_groups[0][0]])

    return ProfileCase(
        mode=mode,
        groups=grad_groups,
        run=run,
        output_shape=tuple(grad_groups[0][0].shape),
        flops=flops,
        hbm_bytes=hbm_bytes,
        correctness=correctness,
    )


def run_profile(
    case: ProfileCase,
    *,
    op_name: str,
    shape: str,
    dim: int,
    dtype: torch.dtype,
    model_state: str,
    l2_bytes: int,
    num_groups: int,
    warmup: int,
    iters: int,
    hbm_used_before: int,
    total_bytes: int,
) -> None:
    device = torch.device("cuda")
    static_allocated = torch.cuda.memory_allocated(device)
    static_reserved = torch.cuda.memory_reserved(device)
    hbm_used_setup, hbm_free_setup, _ = device_hbm_used_bytes(device)

    torch.cuda.reset_peak_memory_stats(device)
    hbm_used_before_profile, _, _ = device_hbm_used_bytes(device)

    torch.cuda.synchronize()
    for i in range(warmup):
        case.run(case.groups[i % num_groups])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(iters):
        case.run(case.groups[i % num_groups])
    end.record()
    end.synchronize()

    us = start.elapsed_time(end) * 1000.0 / iters
    seconds = us * 1e-6
    tflops = case.flops / seconds / 1e12
    hbm_tb_s = case.hbm_bytes / seconds / 1e12
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    hbm_used_after_profile, hbm_free_after_profile, _ = device_hbm_used_bytes(device)
    measured_setup_hbm = hbm_used_setup - hbm_used_before
    measured_profile_hbm_delta = hbm_used_after_profile - hbm_used_before_profile

    print_benchmark_report(
        op_name=op_name,
        mode=case.mode,
        model_state=model_state,
        shape=shape,
        dim=dim,
        dtype=dtype,
        l2_bytes=l2_bytes,
        num_groups=num_groups,
        output_shape=case.output_shape,
        us=us,
        tflops=tflops,
        hbm_tb_s=hbm_tb_s,
        hbm_bytes=case.hbm_bytes,
        total_bytes=total_bytes,
        hbm_used_before=hbm_used_before,
        hbm_used_setup=hbm_used_setup,
        measured_setup_hbm=measured_setup_hbm,
        hbm_free_setup=hbm_free_setup,
        hbm_used_after_profile=hbm_used_after_profile,
        measured_profile_hbm_delta=measured_profile_hbm_delta,
        hbm_free_after_profile=hbm_free_after_profile,
        static_allocated=static_allocated,
        static_reserved=static_reserved,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
        correctness=case.correctness,
    )

    torch.cuda.synchronize()
    time.sleep(COOLDOWN_S)
