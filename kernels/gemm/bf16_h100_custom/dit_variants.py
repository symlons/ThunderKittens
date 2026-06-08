from __future__ import annotations

import json
import os
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, cast


@dataclass(frozen=True)
class CompileFusionEvidence:
    source: Path
    lines: tuple[str, ...]
    categories: tuple[str, ...] = ()




@dataclass(frozen=True)
class TraceEvent:
    start_us: float
    dur_us: float
    category: str
    name: str

    @property
    def end_us(self) -> float:
        return self.start_us + self.dur_us


@dataclass(frozen=True)
class TraceLane:
    label: str
    events: tuple[TraceEvent, ...]
    total_cuda_us: float

    @property
    def span_us(self) -> float:
        if not self.events:
            return 1.0
        return max(event.end_us for event in self.events)


@dataclass(frozen=True)
class TraceCompareData:
    eager_trace: Path
    compile_trace: Path
    eager: TraceLane
    compile: TraceLane
    summary_lines: tuple[str, ...]


def trace_event_glyph(category: str) -> str:
    if category.startswith("fused_"):
        return "F"
    if "attention" in category:
        return "A"
    if "linear" in category or "gemm" in category:
        return "G"
    if "layer_norm" in category:
        return "L"
    if "adaln" in category:
        return "N"
    if "residual" in category:
        return "R"
    if "gelu" in category:
        return "U"
    if "conv" in category or "patch_embed" in category:
        return "P"
    if "memset" in category:
        return "0"
    return "."


def packed_call_timeline_cells(events: tuple[TraceEvent, ...], lane_width: int, call_offset: int, call_window_size: int) -> tuple[str, ...]:
    cells = [" " for _ in range(max(0, lane_width))]
    if not cells or not events:
        return tuple(cells)
    visible_calls = max(1, min(call_window_size, len(cells), len(events)))
    start = min(max(0, call_offset), max(0, len(events) - visible_calls))
    stop = min(len(events), start + visible_calls)
    for idx in range(start, stop):
        cells[idx - start] = trace_event_glyph(events[idx].category)
    return tuple(cells)


def time_marker_timeline_cells(events: tuple[TraceEvent, ...], lane_width: int, offset_us: float, window_us: float) -> tuple[str, ...]:
    cells = [" " for _ in range(max(0, lane_width))]
    if not cells or not events:
        return tuple(cells)
    window = max(1.0, window_us)
    for event in events:
        if event.end_us < offset_us or event.start_us > offset_us + window:
            continue
        left = max(0, min(len(cells) - 1, int((event.start_us - offset_us) / window * len(cells))))
        right = max(left + 1, min(len(cells), int((event.end_us - offset_us) / window * len(cells)) + 1))
        cells[(left + right - 1) // 2] = trace_event_glyph(event.category)
    return tuple(cells)


@dataclass(frozen=True)
class VariantConfig:
    fused: bool = False
    fused_residual: bool = False
    tk_mlp: bool = False
    fused_input_projection: bool = False
    fused_output_projection: bool = False
    fused_epilogue_only: bool = False
    attention_backend: str = "timm"
    compiled: bool = False


@dataclass(frozen=True)
class VariantSpec:
    config: VariantConfig
    description: str
    bench: bool = True
    probe_order: int | None = None

    @property
    def probe(self) -> bool:
        return self.probe_order is not None

    @property
    def requires_compile(self) -> bool:
        return self.config.compiled

    @property
    def requires_fa3(self) -> bool:
        return self.config.attention_backend == "fa3"


def variant(description: str, *, bench: bool = True, probe_order: int | None = None, **config_kwargs) -> VariantSpec:
    return VariantSpec(VariantConfig(**config_kwargs), description, bench=bench, probe_order=probe_order)


VARIANT_SPECS: dict[str, VariantSpec] = {
    # Baselines and core fused paths. These are also the memory-probe set.
    "eager": variant(
        "Baseline timm attention and PyTorch MLP without fused AdaLN paths.",
        probe_order=0,
    ),
    "fused_adaln": variant(
        "Fuse layer norm plus AdaLN modulation, without residual fusion.",
        fused=True,
        probe_order=1,
    ),
    "fused_adaln_residual": variant(
        "Fuse AdaLN and gated residual updates around the block.",
        fused=True,
        fused_residual=True,
        probe_order=2,
    ),
    "tk_mlp": variant(
        "Eager attention with the ThunderKittens MLP path enabled.",
        tk_mlp=True,
        probe_order=3,
    ),
    "fused_adaln_residual_tk_mlp": variant(
        "Combine fused AdaLN/residual with the ThunderKittens MLP path.",
        fused=True,
        fused_residual=True,
        tk_mlp=True,
        probe_order=4,
    ),
    "compile": variant(
        "torch.compile version of the eager baseline.",
        compiled=True,
        probe_order=5,
    ),

    # Projection fusion variants.
    "fused_output_proj": variant(
        "Also fuse the output projection epilogue into the residual path.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
    ),
    "fused_input_proj": variant(
        "Fuse AdaLN into input projections for attention and MLP.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
    ),
    "fused_input_output_proj": variant(
        "Fuse both input projections and output projection residual epilogues.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        fused_output_projection=True,
    ),
    "fused_input_proj_tk_mlp": variant(
        "Input projection fusion with the ThunderKittens MLP path.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        tk_mlp=True,
    ),
    "fused_input_output_proj_tk_mlp": variant(
        "Input and output projection fusion plus ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        fused_output_projection=True,
        tk_mlp=True,
    ),

    # torch.compile variants. The compiled flag is the only thing that gates them behind --compile.
    "compile_fused_adaln": variant(
        "torch.compile with fused AdaLN enabled.",
        fused=True,
        compiled=True,
    ),
    "compile_tk_adaln_only": variant(
        "Alias-style compile case for isolating the fused AdaLN custom op.",
        fused=True,
        compiled=True,
    ),
    "compile_fused_adaln_residual": variant(
        "torch.compile with fused AdaLN and residual paths.",
        fused=True,
        fused_residual=True,
        compiled=True,
    ),
    "compile_fused_output_proj": variant(
        "torch.compile with fused output projection residual epilogue.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        compiled=True,
    ),
    "compile_fused_output_proj_epilogue": variant(
        "torch.compile case that isolates the output projection epilogue fusion.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        fused_epilogue_only=True,
        compiled=True,
    ),
    "compile_tk_adaln_residual_only": variant(
        "Alias-style compile case for isolating fused AdaLN plus residual custom ops.",
        fused=True,
        fused_residual=True,
        compiled=True,
    ),
    "compile_fused_adaln_tk_mlp": variant(
        "torch.compile with fused AdaLN and ThunderKittens MLP.",
        fused=True,
        tk_mlp=True,
        compiled=True,
    ),
    "compile_fused_adaln_residual_tk_mlp": variant(
        "torch.compile with fused AdaLN/residual and ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        tk_mlp=True,
        compiled=True,
    ),

    # FlashAttention-3 variants. Setting attention_backend="fa3" gates these behind --fa3.
    "fa3_attn": variant(
        "FlashAttention-3 attention backend with otherwise eager baseline paths.",
        attention_backend="fa3",
        probe_order=6,
    ),
    "fused_adaln_residual_fa3": variant(
        "FlashAttention-3 plus fused AdaLN and residual paths.",
        fused=True,
        fused_residual=True,
        attention_backend="fa3",
        probe_order=7,
    ),
    "fused_adaln_residual_fa3_tk_mlp": variant(
        "FlashAttention-3 plus fused AdaLN/residual and ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        tk_mlp=True,
        attention_backend="fa3",
    ),
    "fused_output_proj_fa3": variant(
        "FlashAttention-3 with fused output projection residual epilogue.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        attention_backend="fa3",
    ),
    "fused_input_proj_fa3": variant(
        "FlashAttention-3 with fused AdaLN input projections.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        attention_backend="fa3",
    ),
    "fused_input_output_proj_fa3": variant(
        "FlashAttention-3 with input and output projection fusions.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        fused_output_projection=True,
        attention_backend="fa3",
    ),
    "fused_input_proj_fa3_tk_mlp": variant(
        "FlashAttention-3 input projection fusion plus ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        tk_mlp=True,
        attention_backend="fa3",
    ),
    "fused_input_output_proj_fa3_tk_mlp": variant(
        "FlashAttention-3 with input/output projection fusion and ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        fused_input_projection=True,
        fused_output_projection=True,
        tk_mlp=True,
        attention_backend="fa3",
    ),
    "compile_fa3_attn": variant(
        "torch.compile over the FlashAttention-3 baseline.",
        attention_backend="fa3",
        compiled=True,
    ),
    "compile_fused_adaln_residual_fa3": variant(
        "torch.compile with FlashAttention-3 and fused AdaLN/residual paths.",
        fused=True,
        fused_residual=True,
        attention_backend="fa3",
        compiled=True,
    ),
    "compile_fused_adaln_residual_fa3_tk_mlp": variant(
        "torch.compile with FlashAttention-3, fused AdaLN/residual, and ThunderKittens MLP.",
        fused=True,
        fused_residual=True,
        tk_mlp=True,
        attention_backend="fa3",
        compiled=True,
    ),

    # Extra profile-only isolation cases. They remain selectable by --profile-variant.
    "fused_adaln_residual_epilogue": variant(
        "Isolate residual epilogue fusion without projection fusion.",
        fused=True,
        fused_residual=True,
        fused_epilogue_only=True,
        bench=False,
    ),
    "compile_fused_adaln_residual_epilogue": variant(
        "Compiled residual epilogue isolation case without projection fusion.",
        fused=True,
        fused_residual=True,
        fused_epilogue_only=True,
        compiled=True,
        bench=False,
    ),
    "fused_output_proj_epilogue": variant(
        "Isolate the output projection epilogue fusion without torch.compile.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        fused_epilogue_only=True,
        bench=False,
    ),
    "fused_adaln_residual_epilogue_fa3": variant(
        "FlashAttention-3 residual epilogue isolation case.",
        fused=True,
        fused_residual=True,
        fused_epilogue_only=True,
        attention_backend="fa3",
        bench=False,
    ),
    "compile_fused_adaln_residual_epilogue_fa3": variant(
        "Compiled FlashAttention-3 residual epilogue isolation case.",
        fused=True,
        fused_residual=True,
        fused_epilogue_only=True,
        attention_backend="fa3",
        compiled=True,
        bench=False,
    ),
    "fused_output_proj_epilogue_fa3": variant(
        "FlashAttention-3 output projection epilogue isolation case.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        fused_epilogue_only=True,
        attention_backend="fa3",
        bench=False,
    ),
    "compile_fused_output_proj_epilogue_fa3": variant(
        "Compiled FlashAttention-3 output projection epilogue isolation case.",
        fused=True,
        fused_residual=True,
        fused_output_projection=True,
        fused_epilogue_only=True,
        attention_backend="fa3",
        compiled=True,
        bench=False,
    ),
}


def selected_variants(*, probe: bool, include_compile: bool, include_fa3: bool, only_variants: set[str] | None = None) -> list[str]:
    variants = [
        name for name, spec in VARIANT_SPECS.items()
        if (spec.probe if probe else spec.bench)
        and (include_compile or not spec.requires_compile)
        and (include_fa3 or not spec.requires_fa3)
    ]
    if probe:
        variants.sort(key=lambda name: cast(int, VARIANT_SPECS[name].probe_order))
    if only_variants:
        missing = only_variants.difference(VARIANT_SPECS)
        if missing:
            raise ValueError(f"unknown variants: {sorted(missing)}")
        unavailable = only_variants.difference(variants)
        if unavailable:
            raise ValueError(f"variants require disabled flags: {sorted(unavailable)}")
        variants = [variant for variant in variants if variant in only_variants]
    return variants


def variant_config(variant_name: str) -> VariantConfig:
    try:
        return VARIANT_SPECS[variant_name].config
    except KeyError as exc:
        raise ValueError(f"unknown profile variant: {variant_name}") from exc



def _ansi(text: str, code: str, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def _variant_tags(cfg: VariantConfig, *, color_enabled: bool) -> str:
    tags = []
    if cfg.compiled:
        tags.append(_ansi("compile", "1;33", color_enabled))
    if cfg.attention_backend == "fa3":
        tags.append(_ansi("fa3", "1;36", color_enabled))
    if cfg.tk_mlp:
        tags.append(_ansi("tk_mlp", "1;36", color_enabled))
    return f" [{' '.join(tags)}]" if tags else ""


def _stage(status: str, text: str, detail: str, boundary: str) -> tuple[str, str, str, str]:
    return status, text, detail, boundary


def ditblock_fusion_rows(cfg: VariantConfig) -> list[tuple[str, list[tuple[str, str, str, str]]]]:
    attn_impl = "attn_out = FlashAttention-3(qkv)" if cfg.attention_backend == "fa3" else "attn_out = SDPA(qkv)" if cfg.fused_input_projection or cfg.fused_output_projection else "attn_out = timm Attention(attn_in)"
    if cfg.fused_input_projection:
        attn_adaln = _stage(
            "fused",
            "qkv = fused_adaln_linear(x, shift_msa, scale_msa, qkv)",
            "norm1 + AdaLN modulation and qkv projection are computed by fused_adaln_linear.",
            "single fused kernel; no materialized AdaLN output before qkv.",
        )
        attn_qkv = None
        mlp_adaln = _stage(
            "fused",
            "mlp_h = fused_adaln_linear_gelu(x, shift_mlp, scale_mlp, fc1)",
            "Inside the block MLP, norm2 + AdaLN modulation and fc1 projection are computed by fused_adaln_linear_gelu.",
            "single fused kernel; no materialized AdaLN output before fc1.",
        )
        mlp_fc1 = None
    elif cfg.fused:
        attn_adaln = _stage(
            "fused",
            "attn_in = fused_adaln(x, shift_msa, scale_msa)",
            "AdaLN itself is fused.",
            "fused_adaln kernel ends, materializes output, then qkv linear reads it.",
        )
        attn_qkv = _stage("torch", "qkv = linear(attn_in)", "qkv is not fused with AdaLN in this variant.",
            "separate qkv linear kernel/path after AdaLN output.")
        mlp_adaln = _stage(
            "fused",
            "mlp_in = fused_adaln(x, shift_mlp, scale_mlp)",
            "Inside the block MLP, AdaLN itself is fused.",
            "fused_adaln kernel ends, materializes output, then fc1/GELU reads it.",
        )
        mlp_fc1 = _stage("torch", "mlp_h = block_mlp.fc1_gelu(mlp_in)", "This is the input half of the block MLP; it is not the timestep-conditioning MLP.",
            "separate block MLP fc1/GELU path.")
    else:
        attn_adaln = _stage("torch", "attn_in = modulate(norm1(x), shift_msa, scale_msa)", "Reference PyTorch/timm path; no custom AdaLN fusion.",
            "standard PyTorch/timm kernel boundaries.")
        attn_qkv = _stage("torch", "qkv = linear(attn_in)", "Regular qkv projection.",
            "separate qkv linear kernel/path.")
        mlp_adaln = _stage("torch", "mlp_in = modulate(norm2(x), shift_mlp, scale_mlp)", "Reference PyTorch/timm path; no custom AdaLN fusion.",
            "standard PyTorch/timm kernel boundaries.")
        mlp_fc1 = _stage("torch", "mlp_h = block_mlp.fc1_gelu(mlp_in)", "This is the input half of the block MLP; it is not the timestep-conditioning MLP.",
            "separate block MLP fc1/GELU path.")

    if cfg.fused_residual and cfg.fused_output_projection:
        attn_residual_rows = [_stage(
            "fused",
            "x = fused_linear_gated_residual(attn_out, x, gate_msa, proj)",
            "Attention output projection and gated residual epilogue are fused together.",
            "single fused output-projection + gate/residual kernel.",
        )]
        mlp_residual_rows = [_stage(
            "fused",
            "x = fused_linear_gated_residual(mlp_h, x, gate_mlp, fc2)",
            "This is the output half of the block MLP; fc2 projection and gated residual epilogue are fused together.",
            "single fused fc2 + gate/residual kernel.",
        )]
    elif cfg.fused_residual:
        attn_residual_rows = [_stage(
            "fused",
            "x = gated_residual(x, attn_out, gate_msa)",
            "Gating and residual add are fused, but the attention output projection is separate.",
            "attention/proj path ends, then gated_residual runs as its own kernel.",
        )]
        mlp_residual_rows = [_stage(
            "fused",
            "x = gated_residual(x, mlp_out, gate_mlp)",
            "Gating and residual add are fused, but block MLP fc2 is separate.",
            "fc2 path ends, then gated_residual runs as its own kernel.",
        )]
    else:
        attn_residual_rows = [
            _stage("torch", "gated_attn_out = gate_msa * attn_out", "Reference PyTorch gate multiply for attention residual.", "separate elementwise multiply kernel/path in eager PyTorch."),
            _stage("torch", "x = x + gated_attn_out", "Reference PyTorch residual add for attention.", "separate elementwise add kernel/path in eager PyTorch."),
        ]
        mlp_residual_rows = [
            _stage("torch", "gated_mlp_out = gate_mlp * mlp_out", "Reference PyTorch gate multiply after block MLP fc2 has produced mlp_out.", "separate elementwise multiply kernel/path in eager PyTorch."),
            _stage("torch", "x = x + gated_mlp_out", "Reference PyTorch residual add for block MLP.", "separate elementwise add kernel/path in eager PyTorch."),
        ]

    attn_rows = [attn_adaln]
    if attn_qkv is not None and (cfg.attention_backend == "fa3" or cfg.fused_input_projection or cfg.fused_output_projection):
        attn_rows.append(attn_qkv)
    attn_rows.append(_stage(
        "custom" if cfg.attention_backend == "fa3" else "torch",
        attn_impl,
        "Attention backend produces the attention/module output tensor.",
        "separate attention/module call after its input is available.",
    ))
    attn_rows.extend(attn_residual_rows)

    mlp_rows = [mlp_adaln]
    if mlp_fc1 is not None:
        mlp_rows.append(mlp_fc1)
    if not (cfg.fused_residual and cfg.fused_output_projection):
        mlp_rows.append(_stage(
            "torch",
            "mlp_out = block_mlp.fc2(mlp_h)",
            "Output projection of the block MLP before gate_mlp is applied.",
            "separate fc2 path before gated residual update.",
        ))
    mlp_rows.extend(mlp_residual_rows)

    return [("attn", attn_rows), ("block mlp", mlp_rows)]



def load_compile_fusion_evidence(trace_path: Path, *, rows: int = 8) -> CompileFusionEvidence:
    # This is intentionally derived from profiler CUDA kernel names, not from the static variant spec.
    from analyze_dit_compile_fusion import map_fused_kernel, normalize_triton_name

    if not trace_path.exists():
        return CompileFusionEvidence(
            trace_path,
            (
                "trace file not found; generate one with --profile-variant compile --profile-trace-out <path>",
                "or point --ditblock-compile-trace at an existing compile_forward.json artifact",
            ),
            (),
        )

    with trace_path.open() as handle:
        events = json.load(handle)["traceEvents"]

    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"kernel", "gpu_memset"}:
            continue
        name = event.get("name", "")
        triton_name = normalize_triton_name(name)
        if triton_name is None:
            continue
        counts[triton_name] += 1
        durations[triton_name] += event.get("dur", 0.0)

    total_us = sum(durations.values())
    evidence_lines: list[str] = []
    all_categories: set[str] = set()
    ordered_kernels = sorted(counts.items(), key=lambda item: durations[item[0]], reverse=True)
    for triton_name, _count in ordered_kernels:
        category, _desc = map_fused_kernel(triton_name)
        all_categories.add(category)
    for triton_name, count in ordered_kernels[:rows]:
        category, desc = map_fused_kernel(triton_name)
        dur_us = durations[triton_name]
        pct = 100.0 * dur_us / total_us if total_us else 0.0
        evidence_lines.append(f"{category}: {count}x {dur_us / 1000.0:.3f} ms {pct:.1f}%")
        evidence_lines.append(f"  kernel: {triton_name}")
        evidence_lines.append(f"  maps to: {desc}")

    if not evidence_lines:
        evidence_lines.append("no TorchInductor Triton kernels found in this trace")
    return CompileFusionEvidence(trace_path, tuple(evidence_lines), tuple(sorted(all_categories)))


def compile_evidence_categories(evidence: CompileFusionEvidence | None) -> set[str]:
    if evidence is None:
        return set()
    if evidence.categories:
        return set(evidence.categories)
    categories: set[str] = set()
    for line in evidence.lines:
        if line.startswith("  ") or ":" not in line:
            continue
        categories.add(line.split(":", 1)[0])
    return categories


def compile_row_hits(text: str, categories: set[str]) -> list[str]:
    hits: list[str] = []
    if "modulate(norm" in text:
        for category in ("layer_norm", "adaln_modulation"):
            if category in categories:
                hits.append(category)
    if "fc1_gelu" in text and "mlp_gelu" in categories:
        hits.append("mlp_gelu")
    if text.startswith("x = x + gated_") and "gated_residual" in categories:
        hits.append("gated_residual")
    return hits

def _format_stage(status: str, text: str, *, color_enabled: bool) -> str:
    if status == "fused":
        return _ansi("FUSED [" + text + "]", "1;32", color_enabled)
    if status == "custom":
        return _ansi("CUSTOM [" + text + "]", "1;36", color_enabled)
    return _ansi("torch [" + text + "]", "2", color_enabled)


def print_ditblock_fusion_plan(
    variant_names: list[str],
    *,
    color: bool | None = None,
    detail: bool = False,
    compile_evidence: CompileFusionEvidence | None = None,
) -> None:
    color_enabled = ("NO_COLOR" not in os.environ) if color is None else color
    compile_categories = compile_evidence_categories(compile_evidence)
    print("DiTBlock fusion plan")
    print("  " + _ansi("green", "1;32", color_enabled) + " = fused path, " + _ansi("cyan", "1;36", color_enabled) + " = custom/backend path, " + _ansi("yellow", "1;33", color_enabled) + " = torch.compile wrapper, not exact generated fusion")
    for variant_name in variant_names:
        spec = VARIANT_SPECS[variant_name]
        cfg = spec.config
        print(f"\n{_ansi(variant_name, '1', color_enabled)}{_variant_tags(cfg, color_enabled=color_enabled)}")
        print(f"  {spec.description}")
        if cfg.compiled:
            print("  " + _ansi("COMPILE WRAPPER [model = torch.compile(model)]", "1;33", color_enabled))
            if detail:
                print("    detail: torch.compile wraps the whole model/module execution for this variant; rows below remain the source-level DiTBlock tensor program.")
                if compile_evidence is None:
                    print("    actual fusion: not loaded; pass --ditblock-compile-trace with a Chrome trace from this shape/run to show real TorchInductor kernels.")
                else:
                    print(f"    trace evidence from: {compile_evidence.source}")
                    print("    kernel fusion: generated Triton kernels below; library calls and graph-level rewrites are not counted as generated kernels.")
                    for line in compile_evidence.lines:
                        if line.startswith("  "):
                            print("      " + line)
                        else:
                            print("      " + _ansi("INDUCTOR FUSED [" + line + "]", "1;32", color_enabled))
        for section, rows in ditblock_fusion_rows(cfg):
            print(f"  {section}:")
            for status, text, row_detail, boundary in rows:
                print("    " + _format_stage(status, text, color_enabled=color_enabled))
                if cfg.compiled:
                    hits = compile_row_hits(text, compile_categories)
                    if hits:
                        print("      " + _ansi("INDUCTOR FUSED [" + ", ".join(hits) + "]", "1;32", color_enabled))
                if detail:
                    print("      detail: " + row_detail)
                    boundary_label = "source boundary before compile/custom lowering" if cfg.compiled else "kernel boundary"
                    print("      " + boundary_label + ": " + boundary)


def run_ditblock_fusion_ui(
    variant_names: list[str],
    *,
    compile_evidence: CompileFusionEvidence | None = None,
    trace_runner: Callable[[str], tuple[CompileFusionEvidence | None, tuple[str, ...]]] | None = None,
    compare_runner: Callable[[str], tuple[CompileFusionEvidence | None, tuple[str, ...], TraceCompareData | None]] | None = None,
    trace_config: tuple[str, ...] = (),
    initial_compare_lines: tuple[str, ...] = (),
    initial_compare_data: TraceCompareData | None = None,
    initial_show_compare: bool = False,
) -> None:
    import curses

    def draw(
        stdscr,
        selected: int,
        show_detail: bool,
        show_compile_evidence: bool,
        show_compare: bool,
        status_lines: tuple[str, ...],
        compare_lines: tuple[str, ...],
        compare_data: TraceCompareData | None,
        compare_scroll: int,
        compare_scroll_step: int,
        timeline_offset_us: float,
        timeline_window_us: float,
        call_offset: int,
        call_window_size: int,
        timeline_mode: str,
        active_lane: str,
        event_indices: dict[str, int],
        inspect_event: bool,
    ) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        left_width = min(42, max(24, width // 3))
        sep_x = min(left_width, max(0, width - 1))
        x = min(left_width + 2, max(0, width - 1))
        right_width = max(1, width - x - 1)

        def put(row: int, col: int, text: str, attr: int = curses.A_NORMAL) -> None:
            if row < 0 or row >= height or col >= width:
                return
            stdscr.addstr(row, col, text[: max(0, width - col - 1)], attr)

        def put_wrapped(row: int, col: int, text: str, attr: int = curses.A_NORMAL) -> int:
            available = max(12, width - col - 1)
            chunks = textwrap.wrap(text, width=available, break_long_words=True, break_on_hyphens=False) or [""]
            for chunk in chunks:
                if row >= height:
                    return row
                put(row, col, chunk, attr)
                row += 1
            return row

        put(0, 0, "DiTBlock variants", curses.A_BOLD)
        left_help = "up/down variants  t timeline  e trace" if show_compare else "j/k or up/down variants  t timeline"
        put(1, 0, left_help[: max(0, left_width - 1)], curses.A_DIM)
        if sep_x < width:
            for row in range(height):
                put(row, sep_x, "|", curses.A_DIM)
        visible_variants = max(0, height - 3)
        variant_start = min(max(0, selected - visible_variants // 2), max(0, len(variant_names) - visible_variants)) if visible_variants else 0
        for row, name in enumerate(variant_names[variant_start:variant_start + visible_variants]):
            idx = variant_start + row
            attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
            put(row + 3, 0, name[: max(0, left_width - 1)], attr)

        name = variant_names[selected]
        spec = VARIANT_SPECS[name]
        cfg = spec.config
        compile_categories = compile_evidence_categories(compile_evidence)
        y = 0
        put(y, x, name, curses.A_BOLD)
        y += 1
        put(y, x, spec.description)
        y += 1
        put(y, x, f"variant backend: attention={cfg.attention_backend} torch_compile={'on' if cfg.compiled else 'off'}", curses.A_DIM)
        y += 1
        put(
            y,
            x,
            (
                "variant fusions: "
                f"adaln={'on' if cfg.fused else 'off'} "
                f"residual={'on' if cfg.fused_residual else 'off'} "
                f"tk_mlp={'on' if cfg.tk_mlp else 'off'} "
                f"in_proj={'on' if cfg.fused_input_projection else 'off'} "
                f"out_proj={'on' if cfg.fused_output_projection else 'off'} "
                f"epilogue_only={'on' if cfg.fused_epilogue_only else 'off'}"
            ),
            curses.A_DIM,
        )
        y += 1
        for line in trace_config:
            put(y, x, line, curses.A_DIM)
            y += 1
        if status_lines:
            for line in status_lines:
                put(y, x, line, curses.A_DIM)
                y += 1
        y += 1

        def category_color_attr(category: str, *, bold: bool = False, selected_event: bool = False) -> int:
            attr = curses.A_BOLD if bold or selected_event else curses.A_NORMAL
            if not curses.has_colors():
                return attr | (curses.A_REVERSE if selected_event else 0)
            if selected_event:
                return attr | curses.color_pair(6) | curses.A_REVERSE
            if category.startswith("fused_"):
                return attr | curses.color_pair(1)
            if "linear" in category or "gemm" in category:
                return attr | curses.color_pair(3)
            if "attention" in category or "adaln" in category:
                return attr | curses.color_pair(2)
            if "layer_norm" in category or "residual" in category:
                return attr | curses.color_pair(4)
            if "gelu" in category:
                return attr | curses.color_pair(5)
            if category in {"other_cuda", "unknown_triton"}:
                return attr | curses.color_pair(7)
            return attr | curses.A_DIM

        def delta_value_attr(delta_text: str) -> int:
            attr = curses.A_BOLD
            if not curses.has_colors():
                return attr
            stripped = delta_text.strip()
            if stripped.startswith("-") and not stripped.startswith("-0.000"):
                return attr | curses.color_pair(1)
            if stripped.startswith("+") and not stripped.startswith("+0.000"):
                return attr | curses.color_pair(6)
            return curses.A_DIM

        def put_delta_line(row: int, col: int, line: str) -> int:
            prefix, delta = line.split("delta=", 1)
            put(row, col, prefix + "delta=", curses.A_NORMAL)
            delta_col = col + len(prefix) + len("delta=")
            if delta_col < width - 1:
                put(row, delta_col, delta, delta_value_attr(delta))
            return row + 1

        def detail_line_attr(line: str) -> int:
            stripped = line.strip()
            if not stripped:
                return curses.A_DIM
            if stripped.startswith(("mode:", "eager trace:", "compile trace:")):
                return curses.A_DIM
            if stripped.startswith(("category deltas", "top compile", "top eager", "timeline")):
                return curses.A_BOLD | (curses.color_pair(2) if curses.has_colors() else 0)
            if stripped.startswith("category /"):
                return curses.A_DIM
            if stripped.startswith("total CUDA:"):
                return curses.A_BOLD
            if " delta=" in stripped:
                return curses.A_NORMAL
            first = stripped.split(None, 1)[0]
            if first and all(ch.isalnum() or ch == "_" for ch in first):
                return category_color_attr(first, bold=True)
            return curses.A_NORMAL

        if show_compare:
            put(y, x, "eager vs compile trace", curses.A_BOLD)
            y += 1
            put(y, x + 2, "m mode  tab lane  [/] kernel  h/l pan", curses.A_DIM)
            y += 1
            put(y, x + 2, "j/k scroll  up/down variants  a/f speed  t back  q", curses.A_DIM)
            y += 1
            if compare_data is not None:
                lane_width = max(24, min(76, width - x - 16))
                scale_us = max(compare_data.eager.span_us, compare_data.compile.span_us, 1.0)
                window_us = min(max(1.0, timeline_window_us), scale_us)
                offset_us = min(max(0.0, timeline_offset_us), max(0.0, scale_us - window_us))
                zoom = scale_us / max(window_us, 1.0)
                active_for_window = compare_data.eager if active_lane == "eager" else compare_data.compile
                call_window = max(1, min(call_window_size, lane_width, max(1, len(active_for_window.events))))
                call_start = min(max(0, call_offset), max(0, len(active_for_window.events) - call_window))
                call_end = min(len(active_for_window.events), call_start + call_window)
                if timeline_mode == "calls":
                    put(y, x + 2, f"mode=calls  lane={active_lane}  calls={call_start + 1}-{call_end}/{len(active_for_window.events)}  detail={'on' if inspect_event else 'off'}", curses.A_BOLD)
                else:
                    put(y, x + 2, f"mode=time  lane={active_lane}  window={offset_us / 1000.0:.3f}-{(offset_us + window_us) / 1000.0:.3f} ms / {scale_us / 1000.0:.3f} ms  zoom={zoom:.1f}x  detail={'on' if inspect_event else 'off'}", curses.A_BOLD)
                y += 1

                def event_char(category: str) -> str:
                    return trace_event_glyph(category)

                def lane_cells(lane: TraceLane, is_active: bool) -> tuple[list[tuple[str, int]], int | None]:
                    cells = [(" ", curses.A_DIM) for _ in range(lane_width)]
                    selected_col: int | None = None
                    selected_marker: tuple[int, str] | None = None
                    selected_index = event_indices.get(lane.label, 0)
                    if timeline_mode == "calls":
                        visible_calls = max(1, min(call_window_size, lane_width, max(1, len(lane.events))))
                        start = min(max(0, call_offset), max(0, len(lane.events) - visible_calls))
                        stop = min(len(lane.events), start + visible_calls)
                        glyphs = packed_call_timeline_cells(lane.events, lane_width, call_offset, call_window_size)
                        for rel, idx in enumerate(range(start, stop)):
                            event = lane.events[idx]
                            cells[rel] = (glyphs[rel], category_color_attr(event.category))
                            if is_active and idx == selected_index:
                                selected_col = rel
                                selected_marker = (rel, event.category)
                    else:
                        for idx, event in enumerate(lane.events):
                            if event.end_us < offset_us or event.start_us > offset_us + window_us:
                                continue
                            left = max(0, min(lane_width - 1, int((event.start_us - offset_us) / window_us * lane_width)))
                            right = max(left + 1, min(lane_width, int((event.end_us - offset_us) / window_us * lane_width) + 1))
                            marker_col = max(0, min(lane_width - 1, (left + right - 1) // 2))
                            cells[marker_col] = (event_char(event.category), category_color_attr(event.category))
                            if is_active and idx == selected_index:
                                selected_col = marker_col
                                selected_marker = (marker_col, event.category)
                    if selected_marker is not None:
                        marker_col, category = selected_marker
                        cells[marker_col] = ("@", category_color_attr(category, selected_event=True))
                    return cells, selected_col

                def put_timeline(row: int, col: int, prefix: str, lane: TraceLane, cells: list[tuple[str, int]], attr: int) -> None:
                    label = f"{prefix} {lane.label:<7}|"
                    put(row, col, label, attr)
                    cursor_col = col + len(label)
                    for ch, ch_attr in cells:
                        if cursor_col >= width - 1:
                            return
                        put(row, cursor_col, ch, ch_attr)
                        cursor_col += 1
                    put(row, cursor_col, f"| CUDA total={lane.total_cuda_us / 1000.0:.3f}ms", attr)

                for lane in (compare_data.eager, compare_data.compile):
                    is_active = lane.label == active_lane
                    attr = curses.A_BOLD if is_active else curses.A_NORMAL
                    prefix = "FOCUS>" if is_active else "      "
                    cells, selected_col = lane_cells(lane, is_active)
                    put_timeline(y, x + 2, prefix, lane, cells, attr)
                    y += 1
                    if is_active and selected_col is not None and y < height:
                        marker = " " * selected_col + "^ current kernel"
                        put(y, x + 18, marker, curses.A_BOLD | (curses.color_pair(6) if curses.has_colors() else 0))
                        y += 1
                put(y, x + 2, "legend: @ current  G GEMM  F fused  A attn  N AdaLN  L norm", curses.A_DIM)
                y += 1
                put(y, x + 2, "        R residual  U GELU  . other; right = total CUDA time", curses.A_DIM)
                y += 1
                active = compare_data.eager if active_lane == "eager" else compare_data.compile
                if active.events:
                    idx = min(max(0, event_indices.get(active_lane, 0)), len(active.events) - 1)
                    event = active.events[idx]
                    glyph = event_char(event.category)
                    put(y, x + 2, f"selected {active_lane}[{idx + 1}/{len(active.events)}] {glyph} {event.category}", curses.A_BOLD)
                    y += 1
                    put(y, x + 4, f"start={event.start_us / 1000.0:.3f} ms  end={event.end_us / 1000.0:.3f} ms  dur={event.dur_us / 1000.0:.3f} ms", curses.A_DIM)
                    y += 1
                    preview = event.name if len(event.name) <= 72 else event.name[:69] + "..."
                    y = put_wrapped(y, x + 4, "kernel: " + preview, curses.A_DIM)
                    if inspect_event and preview != event.name:
                        y = put_wrapped(y, x + 4, "full: " + event.name, curses.A_DIM)
                    elif not inspect_event and preview != event.name:
                        put(y, x + 4, "enter: show full kernel name", curses.A_DIM)
                        y += 1
                y += 1
            elif compare_lines:
                put(y, x + 2, "structured timeline unavailable; showing text comparison", curses.A_DIM)
                y += 1
            else:
                put(y, x + 2, "press c on a compile variant to profile eager and compile traces", curses.A_DIM)
                stdscr.refresh()
                return

            if compare_lines:
                put(y, x + 2, f"details rows {compare_scroll + 1}-{min(len(compare_lines), compare_scroll + max(1, height - y - 1))} of {len(compare_lines)}  scroll={compare_scroll_step} (a/f)", curses.A_DIM)
                y += 1
                for line in compare_lines[compare_scroll:]:
                    if y >= height:
                        break
                    if " delta=" in line:
                        y = put_delta_line(y, x + 4, line)
                    else:
                        y = put_wrapped(y, x + 4, line, detail_line_attr(line))
            stdscr.refresh()
            return

        if cfg.compiled:
            compile_attr = curses.A_BOLD | (curses.color_pair(3) if curses.has_colors() else 0)
            fused_attr = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0)
            put(y, x, "COMPILE WRAPPER [model = torch.compile(model)]", compile_attr)
            y += 1
            put(y, x + 4, "source rows are before torch.compile lowering", curses.A_DIM)
            y += 1
            if show_detail:
                put(y, x + 4, "detail: compile wraps the whole model/module; exact fusion is trace-dependent.", curses.A_DIM)
                y += 1
            put(y, x, "compile trace:", curses.A_BOLD)
            y += 1
            if compile_evidence is None:
                put(y, x + 4, "not loaded; pass --ditblock-compile-trace <compile_forward.json>", curses.A_DIM)
                y += 1
            else:
                y = put_wrapped(y, x + 4, "green inline rows below are actual Inductor-generated Triton categories from this trace", fused_attr)
                if y < height:
                    put(y, x + 4, "press e for full trace evidence", curses.A_DIM)
                    y += 1
            y += 1
        elif show_compile_evidence:
            put(y, x, "compile trace view", curses.A_BOLD)
            y += 1
            put(y, x + 4, "select a compile variant to view trace-derived Inductor kernels", curses.A_DIM)
            stdscr.refresh()
            return

        if show_compile_evidence:
            put(y, x, "compile trace evidence", curses.A_BOLD)
            y += 1
            if not cfg.compiled:
                put(y, x + 4, "not a compile variant", curses.A_DIM)
                stdscr.refresh()
                return
            if compile_evidence is None:
                put(y, x + 4, "not loaded; pass --ditblock-compile-trace <compile_forward.json>", curses.A_DIM)
                stdscr.refresh()
                return
            y = put_wrapped(y, x + 4, "trace: " + str(compile_evidence.source), curses.A_DIM)
            y = put_wrapped(y, x + 4, "green = actual Inductor-generated Triton kernel from this trace", curses.color_pair(1) | curses.A_BOLD)
            y += 1
            for line in compile_evidence.lines:
                if y >= height:
                    break
                is_detail = line.startswith("  ")
                indent = 8 if is_detail else 4
                attr = curses.A_DIM if is_detail else (curses.color_pair(1) | curses.A_BOLD)
                text = line.strip() if is_detail else "INDUCTOR FUSED [" + line.strip() + "]"
                y = put_wrapped(y, x + indent, text, attr)
            stdscr.refresh()
            return

        for section, rows in ditblock_fusion_rows(cfg):
            if y >= height:
                break
            put(y, x, section + ":", curses.A_BOLD)
            y += 1
            for status, text, detail, boundary in rows:
                if y >= height:
                    break
                attr = curses.A_NORMAL
                label = "torch "
                if status == "fused":
                    attr = curses.color_pair(1) | curses.A_BOLD
                    label = "FUSED "
                elif status == "custom":
                    attr = curses.color_pair(2) | curses.A_BOLD
                    label = "CUSTOM "
                put(y, x + 2, label + "[" + text + "]", attr)
                y += 1
                if cfg.compiled:
                    hits = compile_row_hits(text, compile_categories)
                    if hits and y < height:
                        fused_attr = curses.A_BOLD | (curses.color_pair(1) if curses.has_colors() else 0)
                        y = put_wrapped(y, x + 6, "INDUCTOR FUSED [" + ", ".join(hits) + "]", fused_attr)
                if show_detail and y < height:
                    put(y, x + 6, "detail: " + detail, curses.A_DIM)
                    y += 1
                    if y < height:
                        boundary_label = "source boundary before compile/custom lowering" if cfg.compiled else "kernel boundary"
                        put(y, x + 6, boundary_label + ": " + boundary, curses.A_DIM)
                        y += 1
        stdscr.refresh()

    def main(stdscr) -> None:
        nonlocal compile_evidence
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_CYAN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_MAGENTA, -1)
            curses.init_pair(5, curses.COLOR_BLUE, -1)
            curses.init_pair(6, curses.COLOR_RED, -1)
            curses.init_pair(7, curses.COLOR_WHITE, -1)
        selected = 0
        show_detail = False
        show_compile_evidence = False
        show_compare = initial_show_compare
        status_lines: tuple[str, ...] = ("loaded existing eager/compile traces",) if initial_show_compare else ()
        compare_lines: tuple[str, ...] = initial_compare_lines
        compare_data: TraceCompareData | None = initial_compare_data
        compare_scroll = 0
        compare_scroll_step = 1
        timeline_offset_us = 0.0
        timeline_window_us = max(compare_data.eager.span_us, compare_data.compile.span_us, 1.0) if compare_data is not None else 1.0
        call_offset = 0
        call_window_size = 76
        timeline_mode = "time"
        active_lane = "compile"
        event_indices = {"eager": 0, "compile": 0}
        inspect_event = False

        def compare_span_us() -> float:
            if compare_data is None:
                return 1.0
            return max(compare_data.eager.span_us, compare_data.compile.span_us, 1.0)

        def focused_lane() -> TraceLane | None:
            if compare_data is None:
                return None
            return compare_data.eager if active_lane == "eager" else compare_data.compile

        def clamp_timeline_offset(offset_us: float) -> float:
            scale_us = compare_span_us()
            return min(max(0.0, scale_us - timeline_window_us), max(0.0, offset_us))

        def clamp_call_offset(offset: int) -> int:
            lane = focused_lane()
            if lane is None:
                return 0
            window = max(1, min(call_window_size, max(1, len(lane.events))))
            return min(max(0, len(lane.events) - window), max(0, offset))

        def ensure_selected_call_visible(*, center: bool = False) -> None:
            nonlocal call_offset
            lane = focused_lane()
            if lane is None or not lane.events:
                return
            idx = min(max(0, event_indices.get(active_lane, 0)), len(lane.events) - 1)
            event_indices[active_lane] = idx
            window = max(1, call_window_size)
            if center:
                call_offset = max(0, idx - window // 2)
            elif idx < call_offset + max(1, window // 5):
                call_offset = max(0, idx - max(1, window // 5))
            elif idx >= call_offset + window - max(1, window // 5):
                call_offset = max(0, idx - window + max(1, window // 5) + 1)
            call_offset = clamp_call_offset(call_offset)

        def selected_event() -> TraceEvent | None:
            lane = focused_lane()
            if lane is None or not lane.events:
                return None
            idx = min(max(0, event_indices.get(active_lane, 0)), len(lane.events) - 1)
            event_indices[active_lane] = idx
            return lane.events[idx]

        def ensure_selected_visible(*, zoom_if_full: bool = False, center: bool = False) -> None:
            nonlocal timeline_offset_us, timeline_window_us
            event = selected_event()
            if compare_data is None or event is None:
                return
            scale_us = compare_span_us()
            if zoom_if_full and timeline_window_us >= scale_us * 0.99:
                timeline_window_us = max(64.0, scale_us / 2048.0)
            event_center = (event.start_us + event.end_us) / 2.0
            left_margin = timeline_window_us * 0.20
            right_margin = timeline_window_us * 0.80
            if center:
                timeline_offset_us = clamp_timeline_offset(event_center - timeline_window_us / 2.0)
            elif event_center < timeline_offset_us + left_margin:
                timeline_offset_us = clamp_timeline_offset(event_center - left_margin)
            elif event_center > timeline_offset_us + right_margin:
                timeline_offset_us = clamp_timeline_offset(event_center - right_margin)

        while True:
            draw(
                stdscr,
                selected,
                show_detail,
                show_compile_evidence,
                show_compare,
                status_lines,
                compare_lines,
                compare_data,
                compare_scroll,
                compare_scroll_step,
                timeline_offset_us,
                timeline_window_us,
                call_offset,
                call_window_size,
                timeline_mode,
                active_lane,
                event_indices,
                inspect_event,
            )
            key = stdscr.getch()
            if key == 27:
                stdscr.nodelay(True)
                seq1 = stdscr.getch()
                seq2 = stdscr.getch()
                stdscr.nodelay(False)
                if seq1 == ord("[") and seq2 == ord("A"):
                    key = curses.KEY_UP
                elif seq1 == ord("[") and seq2 == ord("B"):
                    key = curses.KEY_DOWN
                elif seq1 == ord("[") and seq2 == ord("C"):
                    key = curses.KEY_RIGHT
                elif seq1 == ord("[") and seq2 == ord("D"):
                    key = curses.KEY_LEFT
                else:
                    continue
            if key == ord("q"):
                return
            if key == curses.KEY_DOWN:
                selected = min(selected + 1, len(variant_names) - 1)
                show_compare = False
                show_compile_evidence = False
            elif key == curses.KEY_UP:
                selected = max(selected - 1, 0)
                show_compare = False
                show_compile_evidence = False
            elif key in (ord("J"), curses.KEY_NPAGE):
                selected = min(selected + 1, len(variant_names) - 1)
                show_compare = False
                show_compile_evidence = False
            elif key in (ord("K"), curses.KEY_PPAGE):
                selected = max(selected - 1, 0)
                show_compare = False
                show_compile_evidence = False
            elif key == ord("j"):
                if show_compare:
                    compare_scroll = min(compare_scroll + compare_scroll_step, max(0, len(compare_lines) - 1))
                else:
                    selected = min(selected + 1, len(variant_names) - 1)
            elif key == ord("k"):
                if show_compare:
                    compare_scroll = max(compare_scroll - compare_scroll_step, 0)
                else:
                    selected = max(selected - 1, 0)
            elif key == ord("a"):
                if show_compare:
                    compare_scroll_step = max(1, compare_scroll_step // 2)
            elif key == ord("f"):
                if show_compare:
                    compare_scroll_step = min(128, max(1, compare_scroll_step * 2))
            elif key in (ord("\n"), curses.KEY_ENTER):
                if show_compare:
                    inspect_event = not inspect_event
                else:
                    show_detail = not show_detail
            elif key in (ord("d"), ord(" ")):
                show_detail = not show_detail
            elif key == ord("e"):
                show_compile_evidence = not show_compile_evidence
                show_compare = False
            elif key == ord("t"):
                if compare_data is not None or compare_lines:
                    show_compare = True
                    show_compile_evidence = False
                    status_lines = ("showing loaded comparison; press c to rerun for selected variant",)
                else:
                    status_lines = ("no loaded comparison yet; press c to generate one",)
            elif key == ord("m"):
                if show_compare and compare_data is not None:
                    timeline_mode = "calls" if timeline_mode == "time" else "time"
                    if timeline_mode == "calls":
                        ensure_selected_call_visible(center=True)
                    else:
                        ensure_selected_visible(zoom_if_full=True, center=True)
            elif key == ord("	"):
                if show_compare and compare_data is not None:
                    active_lane = "eager" if active_lane == "compile" else "compile"
                    if timeline_mode == "calls":
                        ensure_selected_call_visible(center=True)
                    else:
                        ensure_selected_visible(zoom_if_full=True, center=True)
            elif key in (ord("h"), curses.KEY_LEFT):
                if show_compare and compare_data is not None:
                    if timeline_mode == "calls":
                        call_offset = clamp_call_offset(call_offset - max(1, call_window_size // 4))
                    else:
                        scale_us = compare_span_us()
                        if timeline_window_us >= scale_us * 0.99:
                            timeline_window_us = max(1.0, scale_us / 4.0)
                        timeline_offset_us = clamp_timeline_offset(timeline_offset_us - timeline_window_us * 0.25)
            elif key in (ord("l"), curses.KEY_RIGHT):
                if show_compare and compare_data is not None:
                    if timeline_mode == "calls":
                        call_offset = clamp_call_offset(call_offset + max(1, call_window_size // 4))
                    else:
                        scale_us = compare_span_us()
                        if timeline_window_us >= scale_us * 0.99:
                            timeline_window_us = max(1.0, scale_us / 4.0)
                        timeline_offset_us = clamp_timeline_offset(timeline_offset_us + timeline_window_us * 0.25)
            elif key in (ord("+"), ord("=")):
                if show_compare and compare_data is not None:
                    if timeline_mode == "calls":
                        old_window = call_window_size
                        call_window_size = max(1, int(call_window_size / 1.6))
                        call_offset = clamp_call_offset(call_offset + max(0, (old_window - call_window_size) // 2))
                        ensure_selected_call_visible()
                    else:
                        center = timeline_offset_us + timeline_window_us / 2.0
                        timeline_window_us = max(1.0, timeline_window_us / 1.6)
                        timeline_offset_us = clamp_timeline_offset(center - timeline_window_us / 2.0)
            elif key in (ord("-"), ord("_")):
                if show_compare and compare_data is not None:
                    if timeline_mode == "calls":
                        lane = focused_lane()
                        old_window = call_window_size
                        max_calls = len(lane.events) if lane is not None and lane.events else call_window_size
                        call_window_size = min(max_calls, max(1, int(call_window_size * 1.6)))
                        call_offset = clamp_call_offset(call_offset - max(0, (call_window_size - old_window) // 2))
                        ensure_selected_call_visible()
                    else:
                        scale_us = compare_span_us()
                        center = timeline_offset_us + timeline_window_us / 2.0
                        timeline_window_us = min(scale_us, timeline_window_us * 1.6)
                        timeline_offset_us = clamp_timeline_offset(center - timeline_window_us / 2.0)
            elif key == ord("["):
                if show_compare and compare_data is not None:
                    lane = focused_lane()
                    if lane is not None and lane.events:
                        event_indices[active_lane] = max(0, event_indices.get(active_lane, 0) - 1)
                        if timeline_mode == "calls":
                            ensure_selected_call_visible()
                        else:
                            ensure_selected_visible(zoom_if_full=True)
            elif key == ord("]"):
                if show_compare and compare_data is not None:
                    lane = focused_lane()
                    if lane is not None and lane.events:
                        event_indices[active_lane] = min(len(lane.events) - 1, event_indices.get(active_lane, 0) + 1)
                        if timeline_mode == "calls":
                            ensure_selected_call_visible()
                        else:
                            ensure_selected_visible(zoom_if_full=True)
            elif key == ord("r"):
                if trace_runner is None:
                    status_lines = ("trace runner not configured",)
                    continue
                variant_name = variant_names[selected]
                curses.endwin()
                print(f"profiling trace for {variant_name}...", flush=True)
                new_evidence, status_lines = trace_runner(variant_name)
                if new_evidence is not None:
                    compile_evidence = new_evidence
                show_compile_evidence = variant_config(variant_name).compiled
                show_compare = False
            elif key == ord("c"):
                if compare_runner is None:
                    status_lines = ("compare runner not configured",)
                    continue
                variant_name = variant_names[selected]
                curses.endwin()
                print(f"profiling eager vs {variant_name}...", flush=True)
                new_evidence, compare_lines, compare_data = compare_runner(variant_name)
                if new_evidence is not None:
                    compile_evidence = new_evidence
                status_lines = ("comparison updated",)
                show_compile_evidence = False
                compare_scroll = 0
                compare_scroll_step = 1
                timeline_offset_us = 0.0
                timeline_window_us = max(compare_data.eager.span_us, compare_data.compile.span_us, 1.0) if compare_data is not None else 1.0
                call_offset = 0
                call_window_size = 76
                timeline_mode = "time"
                active_lane = "compile"
                event_indices = {"eager": 0, "compile": 0}
                inspect_event = False
                show_compare = True

    curses.wrapper(main)
