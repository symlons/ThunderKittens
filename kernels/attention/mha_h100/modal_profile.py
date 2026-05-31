import math
import os
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "tk-mha-h100-profile"
TK_ROOT = Path("/ThunderKittens")
KERNEL_DIR = TK_ROOT / "kernels/attention/mha_h100"
ARTIFACT_DIR = Path("/data/mha_h100")
LOCAL_TK_ROOT = Path(os.environ.get("THUNDERKITTENS_ROOT", "/Users/sfkost/research/ThunderKittens"))

app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name("tk-mha-h100-build-artifacts", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "gcc", "g++", "make")
    .pip_install("torch", "numpy", "pandas", "pybind11", "ninja")
    .add_local_dir(str(LOCAL_TK_ROOT), str(TK_ROOT), copy=True)
)


def run(command: str, cwd: Path = KERNEL_DIR) -> None:
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


@app.function(cpu=8, image=image, timeout=60 * 20, volumes={"/data": artifact_volume})
def build() -> list[str]:
    os.environ["THUNDERKITTENS_ROOT"] = str(TK_ROOT)
    for artifact in KERNEL_DIR.glob("_C*.so"):
        artifact.unlink()

    run(f"make clean && make -B -j{os.cpu_count() or 8}")

    artifacts = sorted(KERNEL_DIR.glob("_C*.so"))
    if not artifacts:
        raise RuntimeError("Build completed but produced no _C*.so artifact")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for old in ARTIFACT_DIR.glob("_C*.so"):
        old.unlink()
    names = []
    for artifact in artifacts:
        dst = ARTIFACT_DIR / artifact.name
        shutil.copy2(artifact, dst)
        names.append(artifact.name)
        print(f"saved artifact: {dst} ({dst.stat().st_size} bytes)", flush=True)

    artifact_volume.commit()
    return names


def _load_artifacts(artifact_names: list[str]) -> None:
    artifact_volume.reload()
    if not artifact_names:
        raise RuntimeError("No build artifacts were provided")
    for name in artifact_names:
        src = ARTIFACT_DIR / name
        dst = KERNEL_DIR / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        print(f"loaded artifact: {src} -> {dst}", flush=True)


def _make_inputs(torch, b: int, h: int, n: int, d: int):
    q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    do = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda")
    return q, k, v, do


def _eager_attention(torch, q, k, v, causal: bool):
    qk = torch.matmul(q, k.transpose(-2, -1))
    qk = qk / math.sqrt(q.shape[-1])
    if causal:
        mask = torch.ones(qk.size(-2), qk.size(-1), device=qk.device, dtype=torch.bool).triu(1)
        qk = qk.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(qk, dim=-1), v)


def _time_cuda(torch, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


@app.function(gpu="H100", image=image, timeout=60 * 2, volumes={"/data": artifact_volume})
def profile_case(artifact_names: list[str], b: int, n: int, mode: str, method: str) -> dict:
    os.environ["THUNDERKITTENS_ROOT"] = str(TK_ROOT)
    sys.path.insert(0, str(KERNEL_DIR))
    _load_artifacts(artifact_names)

    if method == "tk" and n < 192:
        return {"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": "unsupported_tk_min_seq_192"}
    if method == "eager" and (n > 3072 or b * n > 32768):
        return {"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": "skipped_large_eager"}
    script = f"""
import json
import math
import statistics
import sys

sys.path.insert(0, {str(KERNEL_DIR)!r})

import torch
import benchmark as bench

b = {b!r}
n = {n!r}
mode = {mode!r}
method = {method!r}
h = 16
d = 128
causal = False
warmup = 2
iters = 7

def make_inputs():
    q = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    do = torch.randn(b, h, n, d, dtype=torch.bfloat16, device="cuda")
    return q, k, v, do

def eager_attention(q, k, v):
    qk = torch.matmul(q, k.transpose(-2, -1))
    qk = qk / math.sqrt(q.shape[-1])
    return torch.matmul(torch.softmax(qk, dim=-1), v)

def fn():
    q, k, v, do = make_inputs()
    if method == "tk":
        y, l_vec = bench.tk.mha_forward(q, k, v, causal)
        if mode == "bwd":
            bench.tk.mha_backward(q, k, v, y, l_vec, do, causal)
    elif method == "sdpa":
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        if mode == "bwd":
            y.backward(do)
    elif method == "eager":
        y = eager_attention(q, k, v)
        if mode == "bwd":
            y.backward(do)
    else:
        raise ValueError(method)

try:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    payload = {{"B": b, "N": n, "mode": mode, "method": method, "ms": statistics.median(times), "status": "ok"}}
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    payload = {{"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": "oom"}}
except Exception as exc:
    torch.cuda.empty_cache()
    payload = {{"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": "error: " + str(exc)}}

print("JSON_RESULT " + json.dumps(payload), flush=True)
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(KERNEL_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=110,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode:
        result = {"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": f"crash_{proc.returncode}"}
    else:
        result = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("JSON_RESULT "):
                result = json.loads(line.removeprefix("JSON_RESULT "))
                break
        if result is None:
            result = {"B": b, "N": n, "mode": mode, "method": method, "ms": None, "status": "missing_json_result"}

    print(
        f"RESULT B={b} N={n} mode={mode} method={method} ms={result['ms']} status={result['status']}",
        flush=True,
    )
    return result


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _speedup(base, tk):
    if base is None or tk is None:
        return None
    return base / tk


@app.local_entrypoint()
def main():
    artifact_names = build.remote()
    print("CPU build artifacts:", artifact_names, flush=True)

    batches = [1, 2, 4, 8, 16]
    seqs = [64, 128, 768, 1024, 1536, 3072, 6144, 12288]
    modes = ["fwd", "bwd"]
    methods = ["tk", "sdpa", "eager"]

    results = {}
    for b in batches:
        for n in seqs:
            if b * n > 65536:
                continue
            for mode in modes:
                for method in methods:
                    result = profile_case.remote(artifact_names, b, n, mode, method)
                    results[(b, n, mode, method)] = result

    print("\nSUMMARY")
    header = [
        "B",
        "N",
        "fwd_tk_ms",
        "fwd_sdpa_ms",
        "fwd_eager_ms",
        "fwd_vs_sdpa",
        "fwd_vs_eager",
        "bwd_tk_ms",
        "bwd_sdpa_ms",
        "bwd_eager_ms",
        "bwd_vs_sdpa",
        "bwd_vs_eager",
    ]
    print(",".join(header), flush=True)
    for b in batches:
        for n in seqs:
            if b * n > 65536:
                continue
            fwd_tk = results[(b, n, "fwd", "tk")]["ms"]
            fwd_sdpa = results[(b, n, "fwd", "sdpa")]["ms"]
            fwd_eager = results[(b, n, "fwd", "eager")]["ms"]
            bwd_tk = results[(b, n, "bwd", "tk")]["ms"]
            bwd_sdpa = results[(b, n, "bwd", "sdpa")]["ms"]
            bwd_eager = results[(b, n, "bwd", "eager")]["ms"]
            row = [
                str(b),
                str(n),
                _fmt(fwd_tk),
                _fmt(fwd_sdpa),
                _fmt(fwd_eager),
                _fmt(_speedup(fwd_sdpa, fwd_tk)),
                _fmt(_speedup(fwd_eager, fwd_tk)),
                _fmt(bwd_tk),
                _fmt(bwd_sdpa),
                _fmt(bwd_eager),
                _fmt(_speedup(bwd_sdpa, bwd_tk)),
                _fmt(_speedup(bwd_eager, bwd_tk)),
            ]
            print(",".join(row), flush=True)
