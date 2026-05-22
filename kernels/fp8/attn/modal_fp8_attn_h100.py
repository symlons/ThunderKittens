import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import modal


app = modal.App("tk-fp8-attn-h100")

cuda_version = "13.0.2"
tag = f"{cuda_version}-devel-ubuntu22.04"
artifact_volume = modal.Volume.from_name("tk-fp8-attn-build-artifacts", create_if_missing=True)

cuda_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-11", "g++-11", "clang-11", "make")
    .run_commands(
        "update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11",
        "apt update",
        "DEBIAN_FRONTEND=noninteractive apt install -y clang-11 python3.10-dev",
        "pip install torch tqdm einops numpy pybind11 matplotlib seaborn packaging ninja",
    )
    .add_local_dir("/Users/sfkost/research/ThunderKittens", "/ThunderKittens")
)

ATTN_DIR = Path("/ThunderKittens/kernels/fp8/attn")
ARTIFACT_DIR = Path("/data/fp8_attn_cuda13")


def named_command(label: str, command: str) -> tuple[str, str]:
    return (label, command)


def shape_label(shape: tuple[int, int, int, int]) -> str:
    b, h, n, d = shape
    return f"B={b} H={h} N={n} D={d}"


def run(command: str, cwd: str | Path = ATTN_DIR) -> None:
    print(f"\n$ cd {cwd} && {command}", flush=True)
    with subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        if proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, command)


def print_env() -> None:
    print("Python:", sys.version, flush=True)
    print("CPU cores (os):", os.cpu_count(), flush=True)
    print("CPU cores (multiprocessing):", multiprocessing.cpu_count(), flush=True)
    print("CPU architecture:", platform.machine(), flush=True)
    run("nvcc --version", cwd="/ThunderKittens")
    run("gcc --version && g++ --version && /usr/bin/g++-11 --version", cwd="/ThunderKittens")
    run(
        "python3 - <<'PY'\n"
        "import torch, pybind11\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
        "print('pybind11', pybind11.get_include())\n"
        "print('cuda available', torch.cuda.is_available())\n"
        "if torch.cuda.is_available(): print('device', torch.cuda.get_device_name(0))\n"
        "PY",
        cwd="/ThunderKittens",
    )


@app.function(cpu=4, image=cuda_image, serialized=False, timeout=60 * 20, volumes={"/data": artifact_volume})
def build():
    os.environ["THUNDERKITTENS_ROOT"] = "/ThunderKittens"
    os.environ["TORCH_PYBIND11"] = "/usr/local/lib/python3.10/site-packages/pybind11/include"
    os.environ["PYTHONPATH"] = "/ThunderKittens/kernels/fp8:/ThunderKittens/kernels/fp8/attn"
    print_env()

    for old in ATTN_DIR.glob("_C*.so"):
        old.unlink()

    n_cores = os.cpu_count() or 4
    run(f"make clean && make BUILD_MODE=torch KERNEL=fp8 -B -j{n_cores}")

    artifacts = sorted(ATTN_DIR.glob("_C*.so"))
    if not artifacts:
        raise RuntimeError("Build completed but produced no _C*.so artifact")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in ARTIFACT_DIR.glob("_C*.so"):
        old.unlink()
    for artifact in artifacts:
        dst = ARTIFACT_DIR / artifact.name
        shutil.copy2(artifact, dst)
        print(f"saved artifact: {dst} ({dst.stat().st_size} bytes)", flush=True)
    artifact_volume.commit()
    return [artifact.name for artifact in artifacts]


DEFAULT_COMMANDS = (
    named_command("forward smoke", "python3 test_fp8.py"),
    named_command("forward sweep", "python3 test_fp8_extensive.py"),
    named_command("backward reference", "python3 test_fp8_bwd.py"),
    named_command("backward kernel", "python3 test_fp8_bwd_kernel.py"),
    named_command("quant correctness", "python3 correctness_quant.py --quick"),
    named_command("attn correctness", "python3 correctness_attn.py --quick"),
    named_command("moderate quant profile", "python3 profile_quant.py --B 4 --H 16 --N 4096 --D 128 --quick-profile"),
    named_command("full short+long profile", "bash profile_fp8_runs.sh"),
)

QUICK_COMMANDS = (
    named_command("forward smoke", "python3 test_fp8.py"),
    named_command("forward quick sweep", "python3 test_fp8_extensive.py --quick"),
    named_command("backward quick reference", "python3 test_fp8_bwd.py --quick"),
    named_command("backward kernel quick", "python3 test_fp8_bwd_kernel.py --quick"),
    named_command("quant correctness quick", "python3 correctness_quant.py --quick"),
    named_command("attn correctness quick", "python3 correctness_attn.py --quick"),
    named_command("quant profile smoke", "python3 profile_quant.py --B 1 --H 8 --N 1536 --D 128 --quick-profile"),
    named_command("fp8 profile smoke", "python3 profile_fp8.py --B 1 --H 8 --N 1536 --D 128 --bench-iters 20 --bench-warmup 20 --bench-groups 1 --bench-cooldown 0.02 --bwd-sdp-descale-mode constant"),
)

LARGE_PROFILE_COMMANDS = (
    named_command(f"quant profile larger batch {shape_label((2, 16, 8192, 128))}", "python3 profile_quant.py --B 2 --H 16 --N 8192 --D 128 --quick-profile"),
    named_command(f"fp8 profile {shape_label((1, 8, 4224, 128))}", "python3 profile_fp8.py --B 1 --H 8 --N 4224 --D 128 --bench-iters 20 --bench-warmup 20 --bench-groups 1 --bench-cooldown 0.02 --bwd-sdp-descale-mode constant"),
    named_command(f"fp8 profile {shape_label((2, 16, 8064, 128))}", "python3 profile_fp8.py --B 2 --H 16 --N 8064 --D 128 --bench-iters 10 --bench-warmup 10 --bench-groups 1 --bench-cooldown 0.02 --bwd-sdp-descale-mode constant"),
    named_command(f"fp8 profile {shape_label((4, 16, 16128, 128))}", "python3 profile_fp8.py --B 4 --H 16 --N 16128 --D 128 --bench-iters 5 --bench-warmup 5 --bench-groups 1 --bench-cooldown 0.02 --bwd-sdp-descale-mode constant --no-sdpa"),
)

LONG_QUICK_COMMANDS = (
    named_command(f"long smoke {shape_label((1, 4, 57600, 128))}", "python3 profile_long_context.py --quick-profile --case 0"),
    named_command(f"long smoke {shape_label((8, 2, 57600, 128))}", "python3 profile_long_context.py --quick-profile --case 3"),
    named_command(f"long smoke {shape_label((1, 2, 144000, 128))}", "python3 profile_long_context.py --quick-profile --case 8"),
)

LONG_300K_COMMANDS = (
    named_command(
        "long 300K smoke",
        "python3 profile_long_context.py --mode long --quick-profile --shapes '1,4,299904,128;2,2,299904,128;4,1,299904,128;8,1,288000,128' --skip-sdpa-bwd",
    ),
)

QUANT_SMOKE_COMMANDS = (
    named_command("quant correctness quick sweep", "python3 correctness_quant.py --quick"),
    named_command(f"quant profile {shape_label((1, 8, 1536, 128))}", "python3 profile_quant.py --B 1 --H 8 --N 1536 --D 128 --quick-profile"),
    named_command(f"quant profile {shape_label((2, 16, 3072, 128))}", "python3 profile_quant.py --B 2 --H 16 --N 3072 --D 128 --quick-profile"),
    named_command(f"quant profile {shape_label((4, 16, 8192, 128))}", "python3 profile_quant.py --B 4 --H 16 --N 8192 --D 128 --quick-profile"),
    named_command(f"quant profile {shape_label((8, 16, 16384, 128))}", "python3 profile_quant.py --B 8 --H 16 --N 16384 --D 128 --quick-profile"),
)

DIAG_BWD_COMMAND = r"""python3 - <<'PY'
import torch
from fp8_suite.metrics import tensor_metrics
from fp8_suite.profiling import uniform_tensor
from fp8_suite.references import sdpa_backward
from fp8_suite.test_backward_kernel import run_kernel

shape = (1, 8, 1536, 128)
generator = torch.Generator(device="cuda").manual_seed(0)
Q = uniform_tensor(shape, generator=generator)
K = uniform_tensor(shape, generator=generator)
V = uniform_tensor(shape, generator=generator)
dO = uniform_tensor(shape, generator=generator)
_, dQ_ref, dK_ref, dV_ref = sdpa_backward(Q, K, V, dO, causal=False)
ref = (dQ_ref, dK_ref, dV_ref)

for label, kwargs in (
    ("SR-dO+SR-dS", {"sr_dO": True, "fp8_dS_mode": 2}),
    ("SR-dO+RTNE-dS", {"sr_dO": True, "fp8_dS_mode": 1}),
    ("SR-dO+bf16-dS", {"sr_dO": True, "fp8_dS_mode": 0}),
    ("RTNE-dO+bf16-dS", {"sr_dO": False, "fp8_dS_mode": 0}),
):
    _, _, _, got = run_kernel(Q, K, V, dO, **kwargs)
    print("\n" + label, flush=True)
    for name, actual, expected in zip(("dQ", "dK", "dV"), got, ref):
        m = tensor_metrics(actual, expected)
        actual_f = actual.float()
        expected_f = expected.float()
        print(
            f"  {name}: QSNR={m['qsnr_dB']:.2f} relL1={m['rel_L1']:.3e} "
            f"RMSE={m['rmse']:.3e} cos={m['cos']:.6f} "
            f"nonzero={int((actual_f != 0).sum().item())}/{actual.numel()} "
            f"absmax={actual_f.abs().max().item():.3e} ref_absmax={expected_f.abs().max().item():.3e} "
            f"mean={actual_f.mean().item():.3e}",
            flush=True,
        )
PY"""


@app.function(gpu="H100", image=cuda_image, serialized=False, timeout=60 * 60 * 4, volumes={"/data": artifact_volume})
def run_tests_and_profiles(artifact_names: list[str], commands: list[tuple[str, str]], continue_on_error: bool):
    artifact_volume.reload()
    os.environ["THUNDERKITTENS_ROOT"] = "/ThunderKittens"
    os.environ["TORCH_PYBIND11"] = "/usr/local/lib/python3.10/site-packages/pybind11/include"
    os.environ["PYTHONPATH"] = "/ThunderKittens/kernels/fp8:/ThunderKittens/kernels/fp8/attn"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print_env()
    run("nvidia-smi", cwd="/ThunderKittens")

    if not artifact_names:
        raise RuntimeError("No CPU build artifacts were provided")
    for name in artifact_names:
        src = ARTIFACT_DIR / name
        dst = ATTN_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing persisted build artifact: {src}")
        shutil.copy2(src, dst)
        print(f"loaded artifact: {src} -> {dst}", flush=True)

    run("python3 - <<'PY'\nimport _C\nprint('loaded _C from', _C.__file__)\nPY")

    failures = []
    total = len(commands)
    for idx, (label, command) in enumerate(commands, start=1):
        print(f"\n[{idx}/{total}] {label}", flush=True)
        try:
            run(command)
        except subprocess.CalledProcessError as exc:
            failures.append((command, exc.returncode))
            print(f"\nFAILED: {command} exited with {exc.returncode}", flush=True)
            if not continue_on_error:
                raise

    if failures:
        print("\nCommand failures:", flush=True)
        for command, code in failures:
            print(f"  [{code}] {command}", flush=True)
        raise RuntimeError(f"{len(failures)} command(s) failed")


@app.local_entrypoint()
def main(
    only_bwd_kernel: bool = False,
    diagnose_bwd: bool = False,
    quick: bool = False,
    profile_large: bool = False,
    long_quick: bool = False,
    long_300k: bool = False,
    quant_smoke: bool = False,
    continue_on_error: bool = False,
):
    if diagnose_bwd:
        commands = [named_command("bwd diagnosis", DIAG_BWD_COMMAND)]
    elif only_bwd_kernel:
        commands = [named_command("backward kernel quick", "python3 test_fp8_bwd_kernel.py --quick")]
    elif profile_large:
        commands = list(LARGE_PROFILE_COMMANDS)
    elif long_quick:
        commands = list(LONG_QUICK_COMMANDS)
    elif long_300k:
        commands = list(LONG_300K_COMMANDS)
    elif quant_smoke:
        commands = list(QUANT_SMOKE_COMMANDS)
    elif quick:
        commands = list(QUICK_COMMANDS)
    else:
        commands = list(DEFAULT_COMMANDS)
    artifact_names = build.remote()
    print("CPU build artifacts:", artifact_names, flush=True)
    run_tests_and_profiles.remote(artifact_names, commands, continue_on_error)
