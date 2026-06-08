from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from dit_profile_utils import CATEGORY_DESCRIPTIONS, CUDA_BREAKDOWN_ORDER


FUSED_KERNEL_MAP = {
    "triton_poi_fused_convolution_0": (
        "patch_embed_bias",
        "PatchEmbed3D.forward: Conv3d output bias/add/layout cleanup after proj.",
    ),
    "triton_poi_fused__to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze_1": (
        "timestep_embedding",
        "TimestepEmbedder.timestep_embedding: arange/exp/div/mul/sin/cos/cat/to/unsqueeze.",
    ),
    "triton_poi_fused_silu_2": (
        "adaln_modulation_silu",
        "AdaLN modulation MLP: SiLU before the 6*hidden linear.",
    ),
    "triton_red_fused_add_native_layer_norm_transpose_view_3": (
        "final_norm_layout",
        "FinalLayer/patch layout: residual add plus layernorm reduction and transpose/view.",
    ),
    "triton_per_fused_add_native_layer_norm_transpose_view_4": (
        "final_norm_layout",
        "FinalLayer/patch layout: layernorm rstd/writeback and transpose/view.",
    ),
    "triton_poi_fused_add_mul_native_layer_norm_split_transpose_unsqueeze_view_5": (
        "adaln_modulation",
        "DiTBlock AdaLN: layernorm normalize, split shift/scale/gate, unsqueeze, transpose/view.",
    ),
    "triton_poi_fused_add_mul_split_transpose_unsqueeze_view_6": (
        "gated_residual",
        "DiTBlock gated residual: residual + gate.unsqueeze(1) * branch output, transpose/view.",
    ),
    "triton_red_fused_native_layer_norm_7": (
        "layer_norm",
        "DiTBlock norm1/norm2: layernorm mean/variance reduction.",
    ),
    "triton_per_fused_native_layer_norm_8": (
        "layer_norm",
        "DiTBlock norm1/norm2: layernorm rstd/writeback.",
    ),
    "triton_poi_fused_add_mul_native_layer_norm_split_unsqueeze_9": (
        "adaln_modulation",
        "DiTBlock AdaLN: ((x - mean) * rstd) * (1 + scale) + shift for attention/MLP input.",
    ),
    "triton_poi_fused_gelu_view_10": (
        "mlp_gelu",
        "timm Mlp: GELU(approximate='tanh') after fc1, plus view/layout.",
    ),
    "triton_poi_fused_add_mul_split_unsqueeze_view_11": (
        "gated_residual",
        "DiTBlock gated residual: x + gate_msa/gate_mlp.unsqueeze(1) * branch output.",
    ),
    "triton_poi_fused_add_mul_native_layer_norm_split_unsqueeze_12": (
        "adaln_modulation",
        "DiTBlock AdaLN: ((x - mean) * rstd) * (1 + scale) + shift for attention/MLP input.",
    ),
    "triton_poi_fused_add_mul_split_unsqueeze_view_13": (
        "gated_residual",
        "DiTBlock gated residual: x + gate_msa/gate_mlp.unsqueeze(1) * branch output.",
    ),
    "triton_poi_fused_add_mul_split_unsqueeze_view_14": (
        "final_modulation",
        "FinalLayer AdaLN/output prep: add/mul/split/unsqueeze/view pointwise work.",
    ),
    "triton_poi_fused_add_split_unsqueeze_15": (
        "final_modulation",
        "FinalLayer AdaLN modulation split/unsqueeze for shift and scale.",
    ),
    "triton_poi_fused_add_mul_native_layer_norm_split_unsqueeze_16": (
        "final_modulation",
        "FinalLayer AdaLN: normalize then apply scale/shift, split/unsqueeze.",
    ),
}


EAGER_OP_MAP = {
    "aten::gelu": "mlp_gelu",
    "aten::silu": "adaln_modulation_silu",
    "aten::native_layer_norm": "layer_norm",
    "aten::layer_norm": "layer_norm_wrapper",
    "aten::add": "pointwise_add",
    "aten::mul": "pointwise_mul",
    "aten::split": "layout_split_chunk",
    "aten::chunk": "layout_split_chunk",
    "aten::unsqueeze": "layout_view",
    "aten::view": "layout_view",
    "aten::reshape": "layout_view",
    "aten::transpose": "layout_transpose",
    "aten::permute": "layout_transpose",
    "aten::clone": "layout_copy_contiguous",
    "aten::copy_": "layout_copy_contiguous",
    "aten::contiguous": "layout_copy_contiguous",
    "aten::linear": "linear_wrapper",
    "aten::addmm": "linear_gemm",
    "aten::_cudnn_attention_forward": "attention",
    "aten::_scaled_dot_product_cudnn_attention": "attention_wrapper",
    "aten::scaled_dot_product_attention": "attention_wrapper",
    "aten::cudnn_convolution": "patch_embed_conv",
    "aten::convolution": "patch_embed_conv_wrapper",
    "aten::_convolution": "patch_embed_conv_wrapper",
    "aten::conv3d": "patch_embed_conv_wrapper",
}


def load_events(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)["traceEvents"]


def normalize_triton_name(name: str) -> str | None:
    match = re.search(r"(triton_[A-Za-z0-9_]+)", name)
    return match.group(1) if match else None


def addmm_category(args: dict) -> str:
    dims = args.get("Input Dims") or []
    if len(dims) < 3 or not all(isinstance(dim, list) for dim in dims[:3]):
        return "linear_gemm"
    bias, mat1, mat2 = dims[:3]
    if len(mat1) != 2 or len(mat2) != 2:
        return "linear_gemm"
    in_features = mat2[0]
    out_features = mat2[1]
    rows = mat1[0]
    if rows <= 4096:
        if in_features == 256:
            return "linear_timestep_fc1"
        if out_features == 6 * in_features:
            return "linear_block_adaln_modulation"
        if out_features == 2 * in_features:
            return "linear_final_adaln_modulation"
        if in_features == out_features:
            return "linear_timestep_fc2"
        return "linear_gemm"
    if out_features == 3 * in_features:
        return "linear_qkv"
    if out_features == 4 * in_features:
        return "linear_dit_mlp_fc1"
    if in_features == 4 * out_features:
        return "linear_dit_mlp_fc2"
    if in_features == out_features:
        return "linear_attention_out"
    if bias and out_features == bias[0]:
        return "linear_final"
    return "linear_gemm"


def build_external_linear_map(events: list[dict]) -> dict[int, str]:
    external_to_category: dict[int, str] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op" or event.get("name") != "aten::addmm":
            continue
        args = event.get("args") or {}
        external_id = args.get("External id")
        if isinstance(external_id, int):
            external_to_category[external_id] = addmm_category(args)
    return external_to_category



def build_external_category_map(events: list[dict]) -> dict[int, str]:
    external_to_category: dict[int, str] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        args = event.get("args") or {}
        external_id = args.get("External id")
        if not isinstance(external_id, int):
            continue
        name = event.get("name", "")
        if name == "aten::addmm":
            external_to_category[external_id] = addmm_category(args)
        elif name == "aten::gelu":
            external_to_category[external_id] = "mlp_gelu"
        elif name == "aten::_cudnn_attention_forward":
            external_to_category[external_id] = "unfused_attention"
        elif name == "aten::native_layer_norm":
            external_to_category[external_id] = "adaln_modulation"
        elif name in {"aten::add", "aten::mul"}:
            external_to_category[external_id] = "adaln_modulation"
        elif name == "aten::silu":
            external_to_category[external_id] = "adaln_modulation_silu"
        elif name == "aten::cudnn_convolution":
            external_to_category[external_id] = "patch_embed_conv"
    return external_to_category


def semantic_cuda_category(name: str, external_id: int | None, external_category_map: dict[int, str]) -> str:
    triton_name = normalize_triton_name(name)
    if triton_name is not None:
        return map_fused_kernel(triton_name)[0]
    lower = name.lower()
    if "memset" in lower:
        return "memset"
    if external_id is not None and external_id in external_category_map:
        return external_category_map[external_id]
    if "sdpa" in lower or "attention" in lower:
        return "unfused_attention"
    if "nvjet" in lower or "cublas" in lower or "cutlass" in lower or "gemm" in lower:
        return "linear_gemm"
    if "convolution" in lower or "conv" in lower:
        return "patch_embed_conv"
    return "other_cuda"


def semantic_cuda_summary(events: list[dict], external_category_map: dict[int, str]) -> tuple[dict[str, dict[str, float]], float]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    total_us = 0.0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
            continue
        dur_us = event.get("dur", 0.0)
        total_us += dur_us
        category = semantic_cuda_category(
            event.get("name", ""),
            (event.get("args") or {}).get("External id"),
            external_category_map,
        )
        grouped[category]["count"] += 1
        grouped[category]["dur_us"] += dur_us
    return grouped, total_us


def print_side_by_side_cuda(
    eager_events: list[dict],
    compile_events: list[dict],
    eager_external_map: dict[int, str],
    compile_external_map: dict[int, str],
) -> None:
    eager_grouped, eager_total_us = semantic_cuda_summary(eager_events, eager_external_map)
    compile_grouped, compile_total_us = semantic_cuda_summary(compile_events, compile_external_map)
    keys = [key for key in CUDA_BREAKDOWN_ORDER if key in eager_grouped or key in compile_grouped]
    extras = sorted((set(eager_grouped) | set(compile_grouped)) - set(keys))
    print("\n== Eager vs Compile CUDA Semantic Breakdown ==")
    print(
        f"{'category':30s} {'eager_ms':>10s} {'eager_%':>8s} {'compile_ms':>11s} "
        f"{'compile_%':>9s} {'delta_ms':>10s} {'local_%':>9s} {'total_%':>9s}"
    )
    for category in keys + extras:
        eager_us = eager_grouped.get(category, {}).get("dur_us", 0.0)
        compile_us = compile_grouped.get(category, {}).get("dur_us", 0.0)
        eager_pct = 100.0 * eager_us / eager_total_us if eager_total_us else 0.0
        compile_pct = 100.0 * compile_us / compile_total_us if compile_total_us else 0.0
        delta_ms = (compile_us - eager_us) / 1000.0
        change = 100.0 * (compile_us - eager_us) / eager_us if eager_us else None
        total_impact = 100.0 * (compile_us - eager_us) / eager_total_us if eager_total_us else 0.0
        change_text = f"{change:+.2f}%" if change is not None else "n/a"
        total_impact_text = f"{total_impact:+.2f}%"
        print(
            f"{category:30s} {eager_us / 1000.0:10.3f} {eager_pct:7.2f}% {compile_us / 1000.0:11.3f} "
            f"{compile_pct:8.2f}% {delta_ms:10.3f} {change_text:>9s} {total_impact_text:>9s}"
        )
    total_change = 100.0 * (compile_total_us - eager_total_us) / eager_total_us if eager_total_us else 0.0
    print(
        f"{'TOTAL':30s} {eager_total_us / 1000.0:10.3f} {100.0:7.2f}% {compile_total_us / 1000.0:11.3f} "
        f"{100.0:8.2f}% {(compile_total_us - eager_total_us) / 1000.0:10.3f} {total_change:+8.2f}% {total_change:+8.2f}%"
    )

def map_fused_kernel(triton_name: str) -> tuple[str, str]:
    exact = FUSED_KERNEL_MAP.get(triton_name)
    if exact is not None:
        return exact
    if "_to_copy_arange_cat_cos_div_exp_mul_sin_unsqueeze" in triton_name:
        return ("timestep_embedding", "TimestepEmbedder.timestep_embedding: arange/exp/div/mul/sin/cos/cat/to/unsqueeze.")
    if "convolution" in triton_name:
        return ("patch_embed_bias", "PatchEmbed3D.forward: Conv3d output bias/add/layout cleanup after proj.")
    if "silu" in triton_name:
        return ("adaln_modulation_silu", "AdaLN modulation MLP: SiLU before the 6*hidden linear.")
    if "gelu" in triton_name:
        return ("mlp_gelu", "timm Mlp: GELU(approximate='tanh') after fc1, plus view/layout.")
    if "native_layer_norm_backward" in triton_name:
        if "add_mul" in triton_name:
            return ("adaln_modulation_backward", "Backward for AdaLN/layernorm modulation: layernorm backward plus scale/gate pointwise terms.")
        if "mul" in triton_name:
            return ("layer_norm_backward", "Backward for DiTBlock norm1/norm2 layernorm with scale/mul terms.")
        return ("layer_norm_backward", "Backward for DiTBlock norm1/norm2 layernorm.")
    if re.search(r"triton_(red|per)_fused_native_layer_norm", triton_name):
        return ("layer_norm", "DiTBlock norm1/norm2: layernorm reduction/rstd work.")
    if "cat_mul_native_layer_norm_squeeze_sum" in triton_name:
        return ("adaln_modulation_backward", "Backward reduction for AdaLN modulation and layernorm scale/shift/gate gradients.")
    if "cat_mul_squeeze_sum" in triton_name:
        return ("gated_residual_backward", "Backward reduction for gated residual gate gradients.")
    if "unsafe_view_clone_permute_stack_sum" in triton_name:
        return ("layout_reduction", "Compiled layout/copy/permute/stack plus sum reduction work.")
    if "mul_sum_view" in triton_name or "per_fused_sum" in triton_name:
        return ("gated_residual_backward", "Backward reduction/sum for gated residual branch or gate gradients.")
    if "add_addmm_mul_native_layer_norm_split" in triton_name:
        return ("linear_adaln_modulation_epilogue", "AdaLN modulation linear addmm epilogue fused with layernorm/scale/shift split pointwise work.")
    if "add_mul_native_layer_norm_split" in triton_name:
        return ("adaln_modulation", "DiTBlock AdaLN: layernorm normalize plus scale/shift split/unsqueeze/view work.")
    if "add_addmm_mul_split" in triton_name:
        return ("linear_adaln_modulation_epilogue", "AdaLN modulation linear addmm epilogue plus split/unsqueeze/view pointwise work.")
    if "add_mul_split" in triton_name:
        return ("gated_residual", "DiTBlock gated residual: x + gate.unsqueeze(1) * branch output, plus layout work.")
    if "add_split_unsqueeze" in triton_name:
        return ("final_modulation", "FinalLayer AdaLN modulation split/unsqueeze work.")
    if "add_native_layer_norm_transpose" in triton_name:
        return ("final_norm_layout", "FinalLayer/patch layout: residual add plus layernorm and transpose/view.")
    return ("unknown_triton", "Unmapped Triton kernel.")


def collect_cpu_ops(events: list[dict]) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "cpu_op":
            continue
        name = event.get("name", "")
        counts[name] += 1
        durations[name] += event.get("dur", 0.0)
    return counts, durations


def collect_cuda_kernels(events: list[dict]) -> tuple[Counter[str], Counter[str]]:
    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
            continue
        name = event.get("name", "")
        counts[name] += 1
        durations[name] += event.get("dur", 0.0)
    return counts, durations


def group_eager(counts: Counter[str], durations: Counter[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for name, count in counts.items():
        category = EAGER_OP_MAP.get(name)
        if category is None:
            continue
        grouped[category]["count"] += count
        grouped[category]["dur_us"] += durations[name]
    return grouped


def group_compile(counts: Counter[str], durations: Counter[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for name, count in counts.items():
        triton_name = normalize_triton_name(name)
        if triton_name is None:
            category = EAGER_OP_MAP.get(name)
        else:
            category = map_fused_kernel(triton_name)[0]
        if category is None:
            continue
        grouped[category]["count"] += count
        grouped[category]["dur_us"] += durations[name]
    return grouped


def print_grouped(title: str, grouped: dict[str, dict[str, float]], total_us: float, rows: int) -> None:
    print(f"\n== {title} ==")
    print(f"{'category':30s} {'count':>8s} {'time_ms':>10s} {'pct':>8s}  description")
    ordered = sorted(grouped.items(), key=lambda item: item[1]["dur_us"], reverse=True)
    for category, stat in ordered[:rows]:
        pct = (100.0 * stat["dur_us"] / total_us) if total_us else 0.0
        desc = CATEGORY_DESCRIPTIONS.get(category, "")
        print(f"{category:30s} {int(stat['count']):8d} {stat['dur_us'] / 1000.0:10.3f} {pct:7.2f}%  {desc}")


def print_fused_kernels(counts: Counter[str], durations: Counter[str], total_us: float) -> None:
    print("\n== Compiled Fused Kernels Mapped To Original Code ==")
    print(f"{'kernel':68s} {'count':>8s} {'time_ms':>10s} {'pct':>8s}  original implementation")
    rows = []
    for name, count in counts.items():
        triton_name = normalize_triton_name(name)
        if triton_name is None:
            continue
        category, desc = map_fused_kernel(triton_name)
        rows.append((durations[name], triton_name, count, category, desc))
    for dur_us, triton_name, count, _category, desc in sorted(rows, reverse=True):
        pct = (100.0 * dur_us / total_us) if total_us else 0.0
        print(f"{triton_name:68s} {count:8d} {dur_us / 1000.0:10.3f} {pct:7.2f}%  {desc}")


def print_fused_cuda_kernels(counts: Counter[str], durations: Counter[str], total_us: float) -> None:
    print("\n== Compiled Fused Kernels By CUDA Kernel Time ==")
    print(f"{'kernel':68s} {'count':>8s} {'cuda_ms':>10s} {'pct':>8s}  original implementation")
    rows = []
    for name, count in counts.items():
        triton_name = normalize_triton_name(name)
        if triton_name is None:
            continue
        category, desc = map_fused_kernel(triton_name)
        rows.append((durations[name], triton_name, count, category, desc))
    for dur_us, triton_name, count, _category, desc in sorted(rows, reverse=True):
        pct = (100.0 * dur_us / total_us) if total_us else 0.0
        print(f"{triton_name:68s} {count:8d} {dur_us / 1000.0:10.3f} {pct:7.2f}%  {desc}")


def print_cuda_category_summary(counts: Counter[str], durations: Counter[str], total_us: float) -> None:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for name, count in counts.items():
        triton_name = normalize_triton_name(name)
        if triton_name is None:
            continue
        category = map_fused_kernel(triton_name)[0]
        grouped[category]["count"] += count
        grouped[category]["dur_us"] += durations[name]
    print("\n== Compiled Fused Categories By CUDA Kernel Time ==")
    print(f"{'category':30s} {'count':>8s} {'cuda_ms':>10s} {'pct':>8s}  description")
    for category, stat in sorted(grouped.items(), key=lambda item: item[1]["dur_us"], reverse=True):
        pct = (100.0 * stat["dur_us"] / total_us) if total_us else 0.0
        desc = CATEGORY_DESCRIPTIONS.get(category, "")
        print(f"{category:30s} {int(stat['count']):8d} {stat['dur_us'] / 1000.0:10.3f} {pct:7.2f}%  {desc}")


def cuda_kernel_category(
    name: str,
    external_id: int | None = None,
    external_linear_map: dict[int, str] | None = None,
) -> tuple[str, str]:
    triton_name = normalize_triton_name(name)
    if triton_name is not None:
        category, desc = map_fused_kernel(triton_name)
        return (f"fused_{category}", desc)
    lower = name.lower()
    if "memset" in lower:
        return ("memset", "CUDA memset/zero-fill kernels.")
    if external_id is not None and external_linear_map is not None and external_id in external_linear_map:
        category = external_linear_map[external_id]
        return (category, CATEGORY_DESCRIPTIONS.get(category, "Linear/matmul library kernel."))
    if "sdpa" in lower or "attention" in lower:
        return ("unfused_attention", "cuDNN SDPA/flash attention kernels; not fused by torch.compile.")
    if "nvjet" in lower or "cublas" in lower or "cutlass" in lower or "gemm" in lower:
        return ("linear_gemm", "Linear/matmul library kernels backing aten::addmm; not fused by torch.compile.")
    if "convolution" in lower or "conv" in lower:
        return ("unfused_convolution", "Patch embedding convolution library kernels; not fused by torch.compile.")
    return ("other_cuda", "Other CUDA kernels not matched by the analyzer.")


def print_all_cuda_category_summary(
    events: list[dict],
    external_linear_map: dict[int, str],
) -> None:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    total_us = 0.0
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
            continue
        dur_us = event.get("dur", 0.0)
        total_us += dur_us
        category, _desc = cuda_kernel_category(
            event.get("name", ""),
            (event.get("args") or {}).get("External id"),
            external_linear_map,
        )
        grouped[category]["count"] += 1
        grouped[category]["dur_us"] += dur_us
    print("\n== Whole Compile Trace By CUDA Kernel Time ==")
    print(f"{'category':30s} {'count':>8s} {'cuda_ms':>10s} {'pct':>8s}  description")
    for category, stat in sorted(grouped.items(), key=lambda item: item[1]["dur_us"], reverse=True):
        pct = (100.0 * stat["dur_us"] / total_us) if total_us else 0.0
        desc = CATEGORY_DESCRIPTIONS.get(category.removeprefix("fused_"), "") or "CUDA kernels in this category."
        print(f"{category:30s} {int(stat['count']):8d} {stat['dur_us'] / 1000.0:10.3f} {pct:7.2f}%  {desc}")


def print_all_cuda_kernels(
    events: list[dict],
    external_linear_map: dict[int, str],
    rows: int,
) -> None:
    total_us = 0.0
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0, "dur_us": 0.0})
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
            continue
        name = event.get("name", "")
        dur_us = event.get("dur", 0.0)
        total_us += dur_us
        category, _desc = cuda_kernel_category(
            name,
            (event.get("args") or {}).get("External id"),
            external_linear_map,
        )
        grouped[(category, name)]["count"] += 1
        grouped[(category, name)]["dur_us"] += dur_us
    print("\n== Top Compile CUDA Kernels ==")
    print(f"{'category':24s} {'count':>8s} {'cuda_ms':>10s} {'pct':>8s}  kernel")
    ordered = sorted(grouped.items(), key=lambda item: item[1]["dur_us"], reverse=True)
    for (category, name), stat in ordered[:rows]:
        pct = (100.0 * stat["dur_us"] / total_us) if total_us else 0.0
        print(f"{category:24s} {int(stat['count']):8d} {stat['dur_us'] / 1000.0:10.3f} {pct:7.2f}%  {name[:120]}")


def print_unfused_compile(counts: Counter[str], durations: Counter[str], total_us: float) -> None:
    print("\n== Compile Ops That Remained Unfused Library Calls ==")
    print(f"{'op':45s} {'count':>8s} {'time_ms':>10s} {'pct':>8s}")
    keep = [
        "aten::addmm",
        "aten::_cudnn_attention_forward",
        "aten::_scaled_dot_product_cudnn_attention",
        "aten::cudnn_convolution",
        "aten::convolution",
        "aten::_convolution",
    ]
    for name in keep:
        dur_us = durations[name]
        pct = (100.0 * dur_us / total_us) if total_us else 0.0
        print(f"{name:45s} {counts[name]:8d} {dur_us / 1000.0:10.3f} {pct:7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Map torch.compile fused DiT kernels back to the original implementation.")
    parser.add_argument("--eager", type=Path, required=True, help="Chrome trace JSON from eager forward.")
    parser.add_argument("--compile", type=Path, required=True, help="Chrome trace JSON from compiled forward.")
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()

    eager_events = load_events(args.eager)
    eager_cpu_counts, eager_cpu_durations = collect_cpu_ops(eager_events)
    compile_events = load_events(args.compile)
    compile_cpu_counts, compile_cpu_durations = collect_cpu_ops(compile_events)
    compile_kernel_counts, compile_kernel_durations = collect_cuda_kernels(compile_events)
    external_linear_map = build_external_linear_map(compile_events)
    eager_external_category_map = build_external_category_map(eager_events)
    compile_external_category_map = build_external_category_map(compile_events)

    eager_total_us = sum(eager_cpu_durations.values())
    compile_total_us = sum(compile_cpu_durations.values())
    compile_cuda_total_us = sum(compile_kernel_durations.values())

    print(f"eager cpu-op total:   {eager_total_us / 1000.0:.3f} ms")
    print(f"compile cpu-op total: {compile_total_us / 1000.0:.3f} ms")
    print("CPU-op percentages are relative to each trace's CPU-op event total, not wall-clock time.")
    print(f"compile CUDA kernel total: {compile_cuda_total_us / 1000.0:.3f} ms")

    print_grouped("Eager Original Ops By Source Category", group_eager(eager_cpu_counts, eager_cpu_durations), eager_total_us, args.rows)
    print_grouped("Compile Ops By Source Category", group_compile(compile_cpu_counts, compile_cpu_durations), compile_total_us, args.rows)
    print_side_by_side_cuda(eager_events, compile_events, eager_external_category_map, compile_external_category_map)
    print_all_cuda_category_summary(compile_events, external_linear_map)
    print_all_cuda_kernels(compile_events, external_linear_map, args.rows)
    print_fused_kernels(compile_cpu_counts, compile_cpu_durations, compile_total_us)
    print_cuda_category_summary(compile_kernel_counts, compile_kernel_durations, compile_cuda_total_us)
    print_fused_cuda_kernels(compile_kernel_counts, compile_kernel_durations, compile_cuda_total_us)
    print_unfused_compile(compile_cpu_counts, compile_cpu_durations, compile_total_us)


if __name__ == "__main__":
    main()
