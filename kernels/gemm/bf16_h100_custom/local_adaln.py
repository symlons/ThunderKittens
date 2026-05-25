from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from adaln_profile_modes import command_for_mode


KERNEL_DIR = Path(__file__).resolve().parent


def run_checked(command: list[str], *, cwd: Path = KERNEL_DIR, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=cwd, env=env, text=True)
    completed.check_returncode()


def local_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    try:
        import torch

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    except Exception:
        pass
    env["PYTHONPATH"] = f"{KERNEL_DIR}:{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_extensions(env: dict[str, str]) -> None:
    suffix = subprocess.check_output(
        [sys.executable, "-c", "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"],
        text=True,
    ).strip()
    targets = [f"_C{suffix}", f"_gelu_bwd{suffix}", f"_linear_bwd_fused{suffix}"]
    for target in targets:
        run_checked(["make", "-j4", target], env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaLN/DiT benchmarks locally without Modal.")
    parser.add_argument("--mode", default="block", help="Same mode string accepted by modal_adaln.py.")
    parser.add_argument("--build", action="store_true", help="Build local PyTorch CUDA extensions before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved local command without running it.")
    args = parser.parse_args()

    env = local_env()
    command = command_for_mode(args.mode)
    if args.dry_run:
        print(f"cwd: {KERNEL_DIR}")
        print(f"$ {' '.join(command)}")
        return
    if args.build:
        build_extensions(env)
    run_checked(command, env=env)


if __name__ == "__main__":
    main()
