from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

from adaln_profile_modes import command_for_mode


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
    return run_checked(command_for_mode(mode), cwd=KERNEL_DIR)


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
