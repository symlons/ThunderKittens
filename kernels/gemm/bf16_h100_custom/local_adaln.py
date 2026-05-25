from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from adaln_profile_modes import command_for_mode, dit_command


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


def command_from_args(args: argparse.Namespace) -> list[str]:
    guided_dit = args.dit or args.tokens is not None or args.spatial is not None
    if not guided_dit:
        return command_for_mode(args.mode)
    return dit_command(
        model=args.model,
        tokens=args.tokens,
        batches=args.batches,
        spatial=tuple(args.spatial) if args.spatial is not None else None,
        sweep=args.sweep,
        compile_model=args.compile,
        probe_memory=args.probe_memory,
        warmup=args.warmup,
        iters=args.iters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AdaLN/DiT benchmarks locally without Modal.")
    parser.add_argument("--mode", default="block", help="Shorthand mode accepted by modal_adaln.py, e.g. mlp_t1024 or dit3d_L_t1024_b128.")
    parser.add_argument("--dit", action="store_true", help="Use guided full-DiT arguments instead of a shorthand mode.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="L", help="DiT model for guided full-DiT runs.")
    parser.add_argument("--tokens", nargs="+", type=int, default=None, help="Custom token counts. Arbitrary counts are mapped to exact 3D shapes.")
    parser.add_argument("--batches", nargs="+", type=int, default=[1], help="Batch sizes for guided full-DiT runs.")
    parser.add_argument("--spatial", nargs=3, type=int, default=None, metavar=("D", "H", "W"), help="Explicit 3D spatial shape for a single full-DiT case.")
    parser.add_argument("--sweep", action="store_true", help="Run all token/batch combinations in guided full-DiT mode.")
    parser.add_argument("--compile", action="store_true", help="Include plain torch.compile(model) in guided full-DiT runs.")
    parser.add_argument("--probe-memory", action="store_true", help="Run memory probe instead of timing in guided full-DiT mode.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for guided full-DiT runs.")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations for guided full-DiT runs.")
    parser.add_argument("--build", action="store_true", help="Build local PyTorch CUDA extensions before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved local command without running it.")
    args = parser.parse_args()

    env = local_env()
    command = command_from_args(args)
    if args.dry_run:
        print(f"cwd: {KERNEL_DIR}")
        print(f"$ {' '.join(command)}")
        return
    if args.build:
        build_extensions(env)
    run_checked(command, env=env)


if __name__ == "__main__":
    main()
