from __future__ import annotations

import os
import subprocess
import sysconfig
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
    .pip_install("pybind11", "kernels")
    .add_local_dir(local_path=str(DEFAULT_LOCAL_TK_ROOT), remote_path=THUNDERKITTENS_ROOT)
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
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())
    completed.check_returncode()
    return completed.stdout


@app.function(gpu="H100", image=image, timeout=60 * 20)
def test_and_profile(mode: str = "block") -> str:
    run_checked(["nvidia-smi"])
    run_checked(["make", "clean"], cwd=KERNEL_DIR)
    ext_target = f"_C{sysconfig.get_config_var('EXT_SUFFIX')}"
    gelu_ext_target = f"_gelu_bwd{sysconfig.get_config_var('EXT_SUFFIX')}"
    run_checked(["make", "-j4", ext_target], cwd=KERNEL_DIR)
    run_checked(["make", "-j4", gelu_ext_target], cwd=KERNEL_DIR)
    return run_checked(["python3", "harness.py", mode, "--report", "KERNEL_REPORT.md"], cwd=KERNEL_DIR)


@app.local_entrypoint()
def main(mode: str = "block") -> None:
    test_and_profile.remote(mode)
