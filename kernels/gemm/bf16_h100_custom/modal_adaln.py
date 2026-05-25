from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "thunderkittens-adaln-layernorm"
THUNDERKITTENS_ROOT = "/ThunderKittens"
KERNEL_DIR = f"{THUNDERKITTENS_ROOT}/kernels/gemm/bf16_h100_custom"
DEFAULT_LOCAL_TK_ROOT = Path(
    os.environ.get("THUNDERKITTENS_ROOT", "/Users/sfkost/research/ThunderKittens")
).expanduser()

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("make", "ninja-build")
    .pip_install("pybind11", "kernels", "timm")
    .add_local_dir(local_path=str(DEFAULT_LOCAL_TK_ROOT), remote_path=THUNDERKITTENS_ROOT, copy=True)
    .run_commands(
        "cd /ThunderKittens/kernels/gemm/bf16_h100_custom && "
        "make clean && "
        "make -j4 _C$(python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') && "
        "make -j4 _gelu_bwd$(python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))') && "
        "make -j4 _linear_bwd_fused$(python3 -c 'import sysconfig; print(sysconfig.get_config_var(\"EXT_SUFFIX\"))')"
    )
)


def run_checked(command: list[str], *, cwd: str | None = None) -> str:
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["PATH"] = f"{env['CUDA_HOME']}/bin:{env['PATH']}"
    torch_lib = subprocess.check_output(
        ["python3", "-c", "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"],
        text=True,
    ).strip()
    env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PYTHONPATH"] = f"{KERNEL_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, env=env, text=True)
    completed.check_returncode()
    return ""


def run_profile(mode: str) -> str:
    run_checked(["nvidia-smi"])
    if mode.startswith("dit3d"):
        parts = mode.split("_")
        model = parts[1].upper() if len(parts) > 1 else "S"
        command = ["python3", "dit3d_e2e_bench.py", "--model", model]
        token_shapes = {
            "t256": ("4", "8", "8"),
            "t512": ("8", "8", "8"),
            "t1024": ("8", "8", "16"),
            "t2048": ("8", "16", "16"),
            "t4096": ("16", "16", "16"),
            "t8192": ("16", "16", "32"),
            "t16384": ("16", "32", "32"),
            "t32768": ("32", "32", "32"),
            "t60000": ("30", "40", "50"),
        }
        for part in parts:
            if part in token_shapes:
                command.extend(["--spatial", *token_shapes[part], "--batches", "1", "--warmup", "5", "--iters", "5"])
                break
        if "fitb" in parts and "--batches" in command:
            batch_i = command.index("--batches")
            del command[batch_i:batch_i + 2]
            command.extend(["--batches", "1", "2", "4", "8", "16", "32", "64", "128"])
        if "b256" in parts:
            if "--batches" in command:
                command[command.index("--batches") + 1] = "256"
            else:
                command.extend(["--batches", "256"])
        if "b512" in parts:
            if "--batches" in command:
                command[command.index("--batches") + 1] = "512"
            else:
                command.extend(["--batches", "512"])
        if "b1024" in parts:
            if "--batches" in command:
                command[command.index("--batches") + 1] = "1024"
            else:
                command.extend(["--batches", "1024"])
        if "mem" in parts:
            command.append("--probe-memory")
        if "sweep" in parts or "tokens" in parts:
            batches = ["1"]
            warmup = "1"
            iters = "3"
            if "smallb" in parts:
                batches = ["1", "2", "4", "8"]
            elif "fitb" in parts:
                batches = ["1", "2", "4", "8", "16", "32", "64", "128"]
                warmup = "5"
                iters = "5"
            elif "highb" in parts:
                batches = ["256", "512", "1024"]
                warmup = "5"
                iters = "10"
            elif "b256" in parts:
                batches = ["256"]
            elif "b512" in parts:
                batches = ["512"]
            elif "b1024" in parts:
                batches = ["1024"]
            command.extend([
                "--sweep",
                "--tokens", "256", "512", "1024", "2048", "4096", "8192", "16384",
                "--batches", *batches,
                "--warmup", warmup,
                "--iters", iters,
            ])
        if "compile" in parts:
            command.append("--compile")
        return run_checked(command, cwd=KERNEL_DIR)
    return run_checked(["python3", "harness.py", mode, "--report", "KERNEL_REPORT.md"], cwd=KERNEL_DIR)


@app.function(gpu="H100", image=image, timeout=60 * 8)
def test_and_profile_h100(mode: str = "block") -> str:
    return run_profile(mode)


@app.function(gpu="H200", image=image, timeout=60 * 8)
def test_and_profile_h200(mode: str = "block") -> str:
    return run_profile(mode)


@app.local_entrypoint()
def main(mode: str = "block", gpu: str = "H100") -> None:
    if gpu.upper() == "H200":
        test_and_profile_h200.remote(mode)
    else:
        test_and_profile_h100.remote(mode)
