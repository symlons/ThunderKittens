from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal


APP_NAME = "tk-quack-layernorm-compare"
TK_ROOT = Path("/ThunderKittens")
LOCAL_TK_ROOT = Path(os.environ.get("THUNDERKITTENS_ROOT", "/Users/sfkost/research/ThunderKittens"))
KERNEL_DIR = TK_ROOT / "kernels/gemm/bf16_h100_custom"
PYTORCH_IMAGE = os.environ.get("PYTORCH_IMAGE", "pytorch/pytorch:2.12.0-cuda13.2-cudnn9-devel")
MODAL_ADD_PYTHON = os.environ.get("MODAL_ADD_PYTHON", "3.12")

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(PYTORCH_IMAGE, add_python=MODAL_ADD_PYTHON)
    .env({"PIP_BREAK_SYSTEM_PACKAGES": "1"})
    .apt_install("make", "ninja-build")
    .pip_install("packaging", "pybind11")
    .pip_install("quack-kernels[cu13]", extra_index_url="https://download.pytorch.org/whl/cu130")
    .add_local_dir(str(LOCAL_TK_ROOT), str(TK_ROOT), copy=True)
)


def run(command: list[str]) -> None:
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
    print(f"$ cd {KERNEL_DIR} && {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=KERNEL_DIR, env=env, text=True, check=True)


def extension_suffix() -> str:
    return subprocess.check_output(
        ["python3", "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        text=True,
    ).strip()


@app.function(gpu="H100", cpu=8, image=image, timeout=60 * 30)
def run_h100(command: list[str]) -> None:
    run(["nvidia-smi"])
    run(["python3", "-c", "import torch; print('torch', torch.__version__)"])
    run(["make", "-j4", f"_C{extension_suffix()}"])
    run(command)


@app.local_entrypoint()
def main(
    shapes: str = "",
    dim: int = 1024,
    warmup: int = 500,
    iters: int = 100,
) -> None:
    command = [
        "python3",
        "profile_tk_vs_quack_layernorm.py",
        "--dim",
        str(dim),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if shapes:
        command.extend(["--shapes", *shapes.replace(",", " ").split()])
    run_h100.remote(command)
