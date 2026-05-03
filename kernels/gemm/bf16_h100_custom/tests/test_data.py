#!/usr/bin/env python3
"""Unit tests for measurement data module — no GPU required."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.measurements import *

EXPECTED_CORRECTNESS = {
    "fwd_output",
    "preact",
    "dz",
    "db",
    "dW",
    "dx",
    "fused_vs_unfused",
}

def test_gpu_specs():
    assert gpu_name in ("NVIDIA H100", "NVIDIA H200")
    assert bf16_dense_tflops > 0
    assert bf16_sparse_tflops > bf16_dense_tflops

def test_bench_config():
    assert bench_m == 4096
    assert bench_k == 4096
    assert bench_n == 4096
    assert bench_dtype == "bf16"

def test_cublas_positive():
    assert cublas_fwd_us > 0
    assert cublas_dW_us  > 0
    assert cublas_dx_us  > 0

def test_custom_times_positive():
    assert custom_gemm_fwd_us > 0
    assert custom_gelu_bwd_us > 0
    assert custom_dW_us > 0
    assert custom_dx_us > 0

def test_step_time_sums():
    """Verify derived totals are consistent."""
    assert abs(custom_fwd_total_us - custom_gemm_fwd_us) < 1
    assert abs(custom_bwd_unfused_us - (custom_gelu_bwd_us + custom_dW_us + custom_dx_us)) < 1
    assert abs(custom_step_us - (custom_fwd_total_us + custom_bwd_unfused_us)) < 1

def test_speedups_gt_one():
    """Custom should beat baselines."""
    assert speedup_step_vs_eager > 1.0
    assert speedup_step_vs_compile > 1.0
    assert speedup_fwd_vs_eager > 1.0
    assert speedup_bwd_vs_autograd > 1.0

def test_percentages_sum_100():
    total = pct_gemm_fwd + pct_gelu_bwd + pct_dW + pct_dx + pct_launch
    assert 99 <= total <= 101, f"percentages sum to {total}"

def test_correctness_pass():
    """All measured max diffs should be within tolerance."""
    assert {c["tensor"] for c in correctness} == EXPECTED_CORRECTNESS
    for c in correctness:
        assert c["threshold"] > 0
        assert c["max_diff"] >= 0
        assert c.get("mean_diff", 0) >= 0
        assert c["max_diff"] <= c["threshold"]

def test_correctness_thresholds_are_meaningful():
    """Thresholds should be tensor-specific, not a single loose catch-all."""
    threshold_by_tensor = {c["tensor"]: c["threshold"] for c in correctness}
    assert threshold_by_tensor["preact"] <= 0.1
    assert threshold_by_tensor["fwd_output"] <= 0.1
    assert threshold_by_tensor["dz"] <= 0.1
    assert threshold_by_tensor["db"] <= 0.25
    assert threshold_by_tensor["dW"] <= 2.0
    assert threshold_by_tensor["dx"] <= 2.0
    assert threshold_by_tensor["fused_vs_unfused"] <= 0.01

def test_graph_nodes_have_meaningful_metadata():
    graph = layers[0]["compute_graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    edge_nodes = {x for edge in graph["edges"] for x in edge}
    assert edge_nodes <= node_ids
    for node in graph["nodes"]:
        assert node.get("dtype")
        assert node.get("formula")
        assert node.get("description")
    op_nodes = [n for n in graph["nodes"] if n.get("type") == "op"]
    assert op_nodes
    assert all(n.get("kernel") for n in op_nodes)

def test_tflops_reasonable():
    """TFLOPs should be below peak + reasonable margin (sparse can exceed dense)."""
    assert custom_gemm_fwd_tflops < bf16_dense_tflops + 600
    assert custom_dW_tflops < bf16_dense_tflops + 600
    assert custom_dx_tflops < bf16_dense_tflops + 600


if __name__ == "__main__":
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed+failed}")
