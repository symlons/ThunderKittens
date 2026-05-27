from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "tk-layernorm-torch-eager"
TK_ROOT = Path("/ThunderKittens")
LOCAL_TK_ROOT = Path(os.environ.get("THUNDERKITTENS_ROOT", "/Users/sfkost/research/ThunderKittens"))
PYTORCH_IMAGE = os.environ.get("PYTORCH_IMAGE", "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
MODAL_ADD_PYTHON = os.environ.get("MODAL_ADD_PYTHON") or None
BENCH_DIR = TK_ROOT / "kernels/gemm/layernorm/baselines"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(PYTORCH_IMAGE, add_python=MODAL_ADD_PYTHON)
    .env(
        {
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            # Reuse compiled autograd graphs across repeated benchmark invocations.
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
            # Reuse Inductor FX graph compilation artifacts when shapes/options match.
            "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        }
    )
    .pip_install("timm")
    .add_local_dir(str(LOCAL_TK_ROOT), str(TK_ROOT), copy=True)
)


def run(command: list[str]) -> None:
    print(f"$ cd {BENCH_DIR} && {' '.join(command)}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Reuse compiled autograd graphs across repeated benchmark invocations.
    env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    # Reuse Inductor FX graph compilation artifacts when shapes/options match.
    env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    completed = subprocess.run(command, cwd=BENCH_DIR, env=env, text=True)
    completed.check_returncode()


def benchmark_command(
    *,
    model_variant: str,
    mode: str,
    model_state: str,
    compile_model: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: str,
    compile_fixed_shapes: bool,
    compare_baseline: str,
    baseline_compile_backend: str,
    baseline_compile_mode: str,
    baseline_compile_fullgraph: bool,
    baseline_compile_dynamic: str,
    shapes: str,
    dim: int,
    warmup: int,
    iters: int,
    dtype: str,
    autocast: str,
    skip_correctness: bool,
    architecture_breakdown: bool,
    dynamo_explain: bool,
    profiler_trace: str,
) -> list[str]:
    command = [
        "python3",
        "torch_eager.py",
        "--model-variant",
        model_variant,
        "--mode",
        mode,
        "--model-state",
        model_state,
        "--dim",
        str(dim),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--dtype",
        dtype,
        "--autocast",
        autocast,
        "--compare-baseline",
        compare_baseline,
        "--baseline-compile-backend",
        baseline_compile_backend,
        "--baseline-compile-mode",
        baseline_compile_mode,
        "--baseline-compile-dynamic",
        baseline_compile_dynamic,
    ]
    if baseline_compile_fullgraph:
        command.append("--baseline-compile-fullgraph")
    if shapes:
        command += ["--shapes", *shapes.replace(",", " ").split()]
    if compile_model:
        command.append("--compile")
        command += ["--compile-backend", compile_backend]
        command += ["--compile-mode", compile_mode]
        command += ["--compile-dynamic", compile_dynamic]
        if compile_fullgraph:
            command.append("--compile-fullgraph")
        if compile_fixed_shapes:
            command.append("--compile-fixed-shapes")
    if skip_correctness:
        command.append("--skip-correctness")
    if architecture_breakdown:
        command.append("--architecture-breakdown")
    if dynamo_explain:
        command.append("--dynamo-explain")
    if profiler_trace:
        command += ["--profiler-trace", profiler_trace]
    return command


@app.function(gpu="H100", image=image, timeout=60 * 30)
def run_h100(command: list[str]) -> None:
    run(["nvidia-smi"])
    run(["python3", "-c", "import torch; print('torch', torch.__version__)"])
    run(command)


@app.function(gpu="H200", image=image, timeout=60 * 30)
def run_h200(command: list[str]) -> None:
    run(["nvidia-smi"])
    run(["python3", "-c", "import torch; print('torch', torch.__version__)"])
    run(command)


@app.local_entrypoint()
def main(
    gpu: str = "H100",
    model_variant: str = "adaln-mlp",
    mode: str = "fwd",
    model_state: str = "eval",
    compile_model: bool = True,
    compile_backend: str = "inductor",
    compile_mode: str = "default",
    compile_fullgraph: bool = False,
    compile_dynamic: str = "auto",
    compile_fixed_shapes: bool = False,
    compare_baseline: str = "eager",
    baseline_compile_backend: str = "inductor",
    baseline_compile_mode: str = "default",
    baseline_compile_fullgraph: bool = False,
    baseline_compile_dynamic: str = "auto",
    shapes: str = "",
    dim: int = 1024,
    warmup: int = 500,
    iters: int = 100,
    dtype: str = "fp32",
    autocast: str = "bf16",
    skip_correctness: bool = False,
    architecture_breakdown: bool = False,
    dynamo_explain: bool = False,
    profiler_trace: str = "",
) -> None:
    command = benchmark_command(
        model_variant=model_variant,
        mode=mode,
        model_state=model_state,
        compile_model=compile_model,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        compile_fixed_shapes=compile_fixed_shapes,
        compare_baseline=compare_baseline,
        baseline_compile_backend=baseline_compile_backend,
        baseline_compile_mode=baseline_compile_mode,
        baseline_compile_fullgraph=baseline_compile_fullgraph,
        baseline_compile_dynamic=baseline_compile_dynamic,
        shapes=shapes,
        dim=dim,
        warmup=warmup,
        iters=iters,
        dtype=dtype,
        autocast=autocast,
        skip_correctness=skip_correctness,
        architecture_breakdown=architecture_breakdown,
        dynamo_explain=dynamo_explain,
        profiler_trace=profiler_trace,
    )
    if gpu.upper() == "H200":
        run_h200.remote(command)
    else:
        run_h100.remote(command)
