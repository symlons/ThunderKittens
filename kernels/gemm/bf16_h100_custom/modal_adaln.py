from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, "/ThunderKittens/kernels/gemm/bf16_h100_custom")

from adaln_profile_modes import command_for_mode, dit_command


APP_NAME = "thunderkittens-adaln-layernorm"
THUNDERKITTENS_ROOT = "/ThunderKittens"
KERNEL_DIR = f"{THUNDERKITTENS_ROOT}/kernels/gemm/bf16_h100_custom"
DEFAULT_LOCAL_TK_ROOT = Path(
    os.environ.get("THUNDERKITTENS_ROOT", "/Users/sfkost/research/ThunderKittens")
).expanduser()

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel")
    .apt_install("git", "make", "ninja-build")
    .pip_install("packaging", "psutil", "pybind11", "kernels", "timm")
    .run_commands(
        "git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
        "cd /tmp/flash-attention/hopper && "
        "FLASH_ATTENTION_DISABLE_SM80=TRUE "
        "FLASH_ATTENTION_DISABLE_FP16=TRUE "
        "FLASH_ATTENTION_DISABLE_FP8=TRUE "
        "FLASH_ATTENTION_DISABLE_VARLEN=TRUE "
        "FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE "
        "FLASH_ATTENTION_DISABLE_APPENDKV=TRUE "
        "FLASH_ATTENTION_DISABLE_LOCAL=TRUE "
        "FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE "
        "FLASH_ATTENTION_DISABLE_PACKGQA=TRUE "
        "FLASH_ATTENTION_DISABLE_SPLIT=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIM96=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIM128=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIM192=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIM256=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE "
        "FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE "
        "MAX_JOBS=2 NVCC_THREADS=1 python setup.py install"
    )
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


def run_profile(mode: str, command: list[str] | None = None) -> str:
    run_checked(["nvidia-smi"])
    return run_checked(command or command_for_mode(mode), cwd=KERNEL_DIR)


@app.function(gpu="H100", image=image, timeout=60 * 15)
def test_and_profile_h100(mode: str = "block", command: list[str] | None = None) -> str:
    return run_profile(mode, command)


@app.function(gpu="H200", image=image, timeout=60 * 15)
def test_and_profile_h200(mode: str = "block", command: list[str] | None = None) -> str:
    return run_profile(mode, command)


@app.local_entrypoint()
def main(
    mode: str = "block",
    gpu: str = "H100",
    dit: bool = False,
    model: str = "L",
    tokens: str = "",
    batches: str = "1",
    spatial: str = "",
    sweep: bool = False,
    compile: bool = False,
    fa3: bool = False,
    probe_memory: bool = False,
    warmup: int = 5,
    iters: int = 5,
    variants: str = "",
    profile_variant: str = "",
    profile_rows: int = 30,
) -> None:
    def parse_ints(value: str) -> list[int]:
        return [int(part) for part in value.replace(",", " ").split() if part]

    command = None
    if dit or tokens or spatial:
        spatial_values = parse_ints(spatial)
        command = dit_command(
            model=model,
            tokens=parse_ints(tokens) or None,
            batches=parse_ints(batches) or [1],
            spatial=tuple(spatial_values) if spatial_values else None,
            sweep=sweep,
            compile_model=compile,
            fa3=fa3,
            probe_memory=probe_memory,
            warmup=warmup,
            iters=iters,
            variants=[part for part in variants.replace(",", " ").split() if part] or None,
            profile_variant=profile_variant,
            profile_rows=profile_rows,
        )
    if gpu.upper() == "H200":
        test_and_profile_h200.remote(mode, command)
    else:
        test_and_profile_h100.remote(mode, command)
