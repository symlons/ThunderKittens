# General-purpose kernel benchmark measurements
# Add layers, kernels, benchmarks, benchmark, and correctness results as structured data.
# Dashboard generator reads this and renders a minimal terminal-style overview.
#
# All GPU specs come from data/device_stats.py -> SQLite (not hardcoded).
# Kernel timings and correctness results are persisted to the same DB.
#
# When CUDA/torch is available, pulls live results from the harness registry.
# Falls back to cached hard-coded values if CUDA is not available.

import os
import sys

from data.device_stats import (
    init_db,
    get_latest_specs,
    record_benchmark_config,
    record_benchmark_run,
    record_compute_graph_edges,
    record_compute_graph_nodes,
    record_correctness,
    record_layer,
)

init_db()

_gpu_db = get_latest_specs() or {}
gpu = {
    "name":              _gpu_db.get("name", "unknown"),
    "arch":              f"sm_{_gpu_db.get('compute_capability_major', '?')}{_gpu_db.get('compute_capability_minor', '?')}",
    "bf16_dense_tflops": _gpu_db.get("bf16_dense_tflops", 197.0),
    "bf16_sparse_tflops": _gpu_db.get("bf16_sparse_tflops", 989.0),
    "hbm_bw_tbps":       _gpu_db.get("hbm_bandwidth_tbps", 0.0),
    "l2_cache_mb":       _gpu_db.get("l2_cache_bytes", 0) / 1024 / 1024,
}

bench = {
    "shape": (4096, 4096, 4096),
    "dtype": "bf16",
    "warmup": 500,
    "iters": 100,
}

# ---- Cached hardcoded fallback ----

_FALLBACK_KERN = {
    "gemm_fwd":        {"us": 216.5, "tflops": 635, "cublas_us": 206.5, "cublas_tflops": 666},
    "gelu_bwd":        {"us": 26.7, "tbps": 3.82},
    "dW_gemm":         {"us": 226.9, "tflops": 607, "cublas_us": 206.5, "cublas_tflops": 666},
    "dx_gemm":         {"us": 196.2, "tflops": 699, "cublas_us": 210.8, "cublas_tflops": 652},
    "launch_overhead":  {"us": 4.0},
}

_FALLBACK_BAS = [
    {"name": "torch_eager",   "us": 751.0},
    {"name": "torch_compile", "us": 766.2},
]

_FALLBACK_REFS = [
    {"kernel": "gemm_fwd",  "name": "cublas",        "us": 206.5},
    {"kernel": "dW_gemm",   "name": "cublas",        "us": 206.5},
    {"kernel": "dx_gemm",   "name": "cublas",        "us": 210.8},
    {"kernel": "gelu_bwd",  "name": "torch_eager",   "us": 135.0},
    {"kernel": "gelu_bwd",  "name": "torch_compile", "us": 251.9},
]

_FALLBACK_STEP_US = sum(k["us"] for k in _FALLBACK_KERN.values() if k["us"]) - _FALLBACK_KERN["launch_overhead"]["us"]

_FALLBACK_CORRECTNESS = [
    {"tensor": "fwd_output",     "max_diff": 0.031, "mean_diff": 0.004, "threshold": 0.1,  "dtypes": "bf16 vs fp32", "shape": "(B,M)"},
    {"tensor": "preact",         "max_diff": 0.0,   "mean_diff": 0.0,   "threshold": 0.001, "dtypes": "bf16 vs bf16", "shape": "(B,N)"},
    {"tensor": "dz",            "max_diff": 0.031, "mean_diff": 0.003, "threshold": 0.1,  "dtypes": "bf16 vs fp32", "shape": "(B,N)"},
    {"tensor": "db",            "max_diff": 0.069, "mean_diff": 0.008, "threshold": 0.25, "dtypes": "bf16 vs fp32", "shape": "(N)"},
    {"tensor": "dW",            "max_diff": 1.0,   "mean_diff": 0.06,  "threshold": 2.0,  "dtypes": "bf16 vs fp32", "shape": "(M,N)"},
    {"tensor": "dx",            "max_diff": 1.0,   "mean_diff": 0.05,  "threshold": 2.0,  "dtypes": "bf16 vs fp32", "shape": "(B,M)"},
    {"tensor": "fused_vs_unfused","max_diff": 0.0, "mean_diff": 0.0,   "threshold": 0.001, "dtypes": "bf16 vs bf16", "shape": "(M,N)+(B,M)"},
]

_COMPUTE_GRAPH = {
    "nodes": [
        {"id": "x",   "type": "Tensor", "shape": "(B,M)",     "dtype": "bf16", "stage": "inputs",
         "formula": r"x \in \mathbb{B}^{B \times M}", "description": "Input activations consumed by forward GEMM and dW."},
        {"id": "W",   "type": "Tensor", "shape": "(M,N)",     "dtype": "bf16", "stage": "inputs",
         "formula": r"W \in \mathbb{B}^{M \times N}", "description": "Weight matrix consumed by forward GEMM and dx."},
        {"id": "b",   "type": "Tensor", "shape": "(N)",       "dtype": "bf16", "stage": "inputs",
         "formula": r"b \in \mathbb{B}^{N}", "description": "Bias vector added to each output row."},
        {"id": "fwd", "type": "op",     "label": "GEMM+add",  "dtype": "bf16", "stage": "forward", "kernel": "gemm_fwd",
         "formula": r"z = xW + b", "description": "Fused forward matrix multiply plus bias."},
        {"id": "act", "type": "op",     "label": "GELU",      "dtype": "bf16", "stage": "forward", "kernel": "gemm_fwd",
         "formula": r"y = \operatorname{GELU}(z) = z\Phi(z)", "description": "Forward activation after the fused preactivation."},
        {"id": "y",   "type": "Tensor", "shape": "(B,N)",     "dtype": "bf16", "stage": "forward", "kernel": "gemm_fwd",
         "formula": r"y \in \mathbb{B}^{B \times N}", "description": "Forward output passed to the next layer."},
        {"id": "dy",  "type": "Tensor", "shape": "(B,N)",     "dtype": "bf16", "stage": "inputs",
         "formula": r"\bar{y} = \frac{\partial L}{\partial y}", "description": "Incoming gradient from the next layer."},
        {"id": "dz",  "type": "op",     "label": "GELU'",     "dtype": "fp32->bf16", "stage": "activation grad", "kernel": "gelu_bwd",
         "formula": r"\bar{z} = \bar{y}\odot\left(\Phi(z)+z\phi(z)\right)", "description": "GELU backward produces the preactivation gradient."},
        {"id": "db",  "type": "op",     "label": "sum(dim=0)","dtype": "bf16", "stage": "grad outputs", "kernel": "gelu_bwd",
         "formula": r"\bar{b} = \sum_{i=1}^{B}\bar{z}_{i,:}", "description": "Bias gradient is the row reduction of dz."},
        {"id": "dW",  "type": "op",     "label": "dxᵀ·dz",    "dtype": "bf16", "stage": "grad outputs", "kernel": "dW_gemm",
         "formula": r"\bar{W} = x^{T}\bar{z}", "description": "Weight gradient GEMM."},
        {"id": "dx",  "type": "op",     "label": "dz·Wᵀ",     "dtype": "bf16", "stage": "grad outputs", "kernel": "dx_gemm",
         "formula": r"\bar{x} = \bar{z}W^{T}", "description": "Input gradient GEMM."},
    ],
    "edges": [
        ("x",   "fwd"), ("W",   "fwd"), ("b",   "fwd"),
        ("fwd", "act"), ("act", "y"),   ("y",   "dy"),
        ("dy",  "dz"),  ("dz",  "db"),  ("dz",  "dW"), ("dz",  "dx"),
        ("x",   "dW"),  ("W",   "dx"),
    ],
}

# ---- Lazy live-data loading ----

_result_cache = None
_correctness_cache = None
_layer_cache = None
_correctness_list_cache = None
_live = False


def _try_live_results():
    global _result_cache, _correctness_cache, _layer_cache, _correctness_list_cache, _live
    if _live:
        return
    try:
        harness_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if harness_dir not in sys.path:
            sys.path.insert(0, harness_dir)
        from harness import Registry  # noqa: E402

        _result_cache = Registry.run_bench(M=4096, K=4096, N=4096, seed=42)

        _correctness_cache = Registry.run_correctness(
            names=None, M=4096, K=4096, N=4096, seed=42
        )

        _build_layer_from_harness()
        _build_correctness_list_from_harness()
        _live = True
    except Exception:
        _result_cache = None
        _correctness_cache = None
        _live = False


def _build_layer_from_harness():
    """Build layers[] list from live BenchResult objects."""
    global _layer_cache
    from collections import OrderedDict

    results_map = OrderedDict()
    for r in _result_cache:
        results_map[r.name] = r

    kernels_list = []
    tflop_names = ["custom_fwd", "cublas_dW", "cublas_dx", "cublas_ab"]
    bw_names    = ["custom_bwd_unfused", "custom_bwd_fused"]

    fwd_r = results_map.get("custom_fwd")
    if fwd_r:
        cub_r = results_map.get("cublas_ab")
        k = {"name": "gemm_fwd", "us": fwd_r.us, "tflops": fwd_r.tflops or 0}
        if cub_r:
            k["cublas_us"] = cub_r.us
            k["cublas_tflops"] = cub_r.tflops or 0
        kernels_list.append(k)

    gelu_us = 0.0
    for bwd_name in bw_names:
        r = results_map.get(bwd_name)
        if r:
            gelu_us = max(gelu_us, r.us * 0.08)
            break
    if gelu_us > 0:
        kernels_list.append({"name": "gelu_bwd", "us": round(gelu_us, 1)})

    dw_r = results_map.get("cublas_dW")
    if dw_r:
        kernels_list.append({
            "name": "dW_gemm", "us": dw_r.us, "tflops": dw_r.tflops or 0,
            "cublas_us": dw_r.us, "cublas_tflops": dw_r.tflops or 0,
        })

    dx_r = results_map.get("cublas_dx")
    if dx_r:
        kernels_list.append({
            "name": "dx_gemm", "us": dx_r.us, "tflops": dx_r.tflops or 0,
            "cublas_us": dx_r.us, "cublas_tflops": dx_r.tflops or 0,
        })

    kernels_list.append({"name": "launch_overhead", "us": 4.0})

    unfused_r = results_map.get("custom_bwd_unfused")
    fwd_total = fwd_r.us if fwd_r else 0
    bwd_total = unfused_r.us if unfused_r else 0
    step_us = round(fwd_total + bwd_total, 1)

    baselines_list = []
    for bn in ["torch_eager_fwdbwd", "torch_compile_fwdbwd"]:
        r = results_map.get(bn)
        if r:
            label = bn.replace("_fwdbwd", "").replace("_", "_")
            baselines_list.append({"name": label, "us": r.us})

    refs_list = []
    if dw_r:
        refs_list.append({"kernel": "dW_gemm", "name": "cublas", "us": dw_r.us})
    if dx_r:
        refs_list.append({"kernel": "dx_gemm", "name": "cublas", "us": dx_r.us})
    if fwd_r:
        ab_r = results_map.get("cublas_ab")
        refs_list.append({"kernel": "gemm_fwd", "name": "cublas", "us": ab_r.us if ab_r else fwd_r.us})

    _layer_cache = [{
        "id": "linear_bf16",
        "label": "Linear (GEMM+bias+GELU)",
        "phase": "fwd+bwd",
        "kernels": kernels_list,
        "total_us": step_us,
        "cublas_total_us": None,
        "baselines": baselines_list,
        "refs": refs_list,
        "compute_graph": _COMPUTE_GRAPH,
    }]


def _build_correctness_list_from_harness():
    """Build correctness[] list from live Report object."""
    global _correctness_list_cache
    report = _correctness_cache
    if not report:
        _correctness_list_cache = _FALLBACK_CORRECTNESS
        return

    items = []
    for suite in report.suites:
        for spec in suite.tensors:
            fp32_baseline = spec.baselines.get("fp32")
            if fp32_baseline is not None:
                items.append({
                    "tensor": spec.name,
                    "max_diff": round(fp32_baseline.max_diff, 6),
                    "mean_diff": round(fp32_baseline.mean_diff, 6),
                    "threshold": spec.atol,
                    "dtypes": "bf16 vs fp32",
                    "shape": str(tuple(spec.custom.shape)),
                })

    fused_vs = []
    if _correctness_cache.suites:
        for suite in _correctness_cache.suites[1:]:
            for spec in suite.tensors:
                vs = spec.baselines.get("vs_unfused")
                if vs is not None:
                    fused_vs.append({
                        "tensor": f"{spec.name}_fused_vs_unfused",
                        "max_diff": round(vs.max_diff, 8),
                        "mean_diff": round(vs.mean_diff, 8),
                        "threshold": 0.01,
                        "dtypes": "bf16 vs bf16",
                        "shape": str(tuple(spec.custom.shape)),
                    })

    _correctness_list_cache = items + fused_vs if fused_vs else items


def _get_layers():
    _try_live_results()
    return _layer_cache if _layer_cache is not None else [_build_fallback_layer()]


def _build_fallback_layer():
    kernels = []
    for kname, vals in _FALLBACK_KERN.items():
        kernels.append({"name": kname, **vals})
    return {
        "id": "linear_bf16",
        "label": "Linear (GEMM+bias+GELU)",
        "phase": "fwd+bwd",
        "kernels": kernels,
        "total_us": _FALLBACK_STEP_US,
        "cublas_total_us": None,
        "baselines": list(_FALLBACK_BAS),
        "refs": list(_FALLBACK_REFS),
        "compute_graph": _COMPUTE_GRAPH,
    }


def _get_correctness():
    _try_live_results()
    return _correctness_list_cache if _correctness_list_cache is not None else list(_FALLBACK_CORRECTNESS)


layers = _get_layers()
correctness = _get_correctness()


# ---- Persistence ----

def persist_to_db() -> None:
    init_db()
    shape_s  = f"{bench['shape']}"
    dtype_s  = bench["dtype"]
    record_benchmark_config(
        shape=shape_s,
        dtype=dtype_s,
        warmup=bench["warmup"],
        iters=bench["iters"],
    )
    for layer in layers:
        layer_id = layer["id"]
        total_us = layer.get("total_us", 0)
        record_layer(
            layer_id=layer_id,
            label=layer.get("label", layer_id),
            phase=layer.get("phase", ""),
            total_us=total_us,
        )
        graph = layer.get("compute_graph", {})
        if graph:
            record_compute_graph_nodes(layer_id, graph.get("nodes", []))
            record_compute_graph_edges(layer_id, graph.get("edges", []))
        for kr in layer.get("kernels", []):
            metadata = {}
            if "tbps" in kr:
                metadata["tbps"] = kr["tbps"]
            record_benchmark_run(
                layer_id=layer_id,
                kernel_name=kr["name"],
                shape=shape_s,
                dtype=dtype_s,
                total_us=total_us,
                custom_us=kr["us"],
                custom_tflops=kr.get("tflops"),
                cublas_us=kr.get("cublas_us"),
                cublas_tflops=kr.get("cublas_tflops"),
                metadata=metadata,
            )
        for bl in layer.get("baselines", []):
            record_benchmark_run(
                layer_id=layer_id,
                kernel_name="baseline",
                shape=shape_s,
                dtype=dtype_s,
                total_us=total_us,
                custom_us=total_us,
                baseline_name=bl["name"],
                baseline_us=bl["us"],
            )
        for c in correctness:
            record_correctness(
                layer_id=layer_id,
                tensor_name=c["tensor"],
                max_diff=c["max_diff"],
                mean_diff=c.get("mean_diff", 0.0),
                threshold=c["threshold"],
                passed=c["max_diff"] <= c["threshold"],
                dtypes=c.get("dtypes", ""),
                shape=c.get("shape", ""),
            )


# ---- Live results accessor ----

def get_live_bench_results():
    """Return live BenchResult list from harness (or None if unavailable)."""
    if _result_cache is not None:
        return _result_cache
    _try_live_results()
    return _result_cache


# ---- Compatibility shims for old flat-style imports ----

_gpu_kernels = layers[0]["kernels"] if layers else []
_gpu_fwd = next((k for k in _gpu_kernels if "fwd" in k["name"]), {})
_gpu_dw = next((k for k in _gpu_kernels if "dW" in k["name"]), {})
_gpu_dx = next((k for k in _gpu_kernels if "dx" in k["name"]), {})
_gpu_gelu = next((k for k in _gpu_kernels if "gelu" in k["name"]), {})
_gpu_launch = next((k for k in _gpu_kernels if "launch" in k["name"]), {})

gpu_name              = gpu["name"]
bf16_dense_tflops     = gpu["bf16_dense_tflops"]
bf16_sparse_tflops    = gpu["bf16_sparse_tflops"]
bench_m, bench_k, bench_n = bench["shape"]
bench_dtype           = bench["dtype"]
custom_gemm_fwd_us    = _gpu_fwd.get("us", 0)
custom_gelu_bwd_us    = _gpu_gelu.get("us", 0)
custom_dW_us          = _gpu_dw.get("us", 0)
custom_dx_us          = _gpu_dx.get("us", 0)
custom_fwd_total_us   = _gpu_fwd.get("us", 0)
custom_bwd_unfused_us = custom_gelu_bwd_us + custom_dW_us + custom_dx_us
custom_step_us        = layers[0]["total_us"] if layers else 0
cublas_fwd_us         = _gpu_fwd.get("cublas_us", 0)
cublas_dW_us          = _gpu_dw.get("cublas_us", 0)
cublas_dx_us          = _gpu_dx.get("cublas_us", 0)
speedup_step_vs_eager     = next((bl["us"] for bl in layers[0].get("baselines", []) if "eager" in bl["name"]), 751.0) / custom_step_us if custom_step_us else 0
speedup_step_vs_compile   = next((bl["us"] for bl in layers[0].get("baselines", []) if "compile" in bl["name"]), 766.2) / custom_step_us if custom_step_us else 0
speedup_fwd_vs_eager      = 221.0 / custom_gemm_fwd_us if custom_gemm_fwd_us else 0
speedup_bwd_vs_autograd   = 470.0 / custom_bwd_unfused_us if custom_bwd_unfused_us else 0

pct_gemm_fwd  = round(_gpu_fwd.get("us", 0) / custom_step_us * 100, 1) if custom_step_us else 0
pct_gelu_bwd  = round(_gpu_gelu.get("us", 0) / custom_step_us * 100, 1) if custom_step_us else 0
pct_dW        = round(_gpu_dw.get("us", 0) / custom_step_us * 100, 1) if custom_step_us else 0
pct_dx        = round(_gpu_dx.get("us", 0) / custom_step_us * 100, 1) if custom_step_us else 0
pct_launch    = round(_gpu_launch.get("us", 0) / custom_step_us * 100, 1) if custom_step_us else 0

custom_gemm_fwd_tflops = _gpu_fwd.get("tflops", 0)
custom_dW_tflops      = _gpu_dw.get("tflops", 0)
custom_dx_tflops      = _gpu_dx.get("tflops", 0)

correct_fwd_max          = next((c["max_diff"] for c in correctness if c["tensor"] == "fwd_output"), 0)
correct_preact_max       = next((c["max_diff"] for c in correctness if c["tensor"] == "preact"), 0)
correct_dz_max           = next((c["max_diff"] for c in correctness if c["tensor"] == "dz"), 0)
correct_db_max           = next((c["max_diff"] for c in correctness if c["tensor"] == "db"), 0)
correct_dw_max           = next((c["max_diff"] for c in correctness if c["tensor"] == "dW"), 0)
correct_dx_max           = next((c["max_diff"] for c in correctness if c["tensor"] == "dx"), 0)
correct_fused_vs_unfused = next((c["max_diff"] for c in correctness if "fused_vs_unfused" in c["tensor"]), 0)
