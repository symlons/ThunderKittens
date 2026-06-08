from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REGULAR_TIMM_VARIANT = {
    "fused": False,
    "fused_residual": False,
    "tk_mlp": False,
    "fused_input_projection": False,
    "fused_output_projection": False,
    "fused_epilogue_only": False,
    "attention_backend": "timm",
}

REGULAR_TIMM_BLOCK_KWARGS = {
    "fused_adaln_enabled": False,
    "fused_residual_enabled": False,
    "tk_mlp_enabled": False,
    "fused_input_projection_enabled": False,
    "fused_output_projection_enabled": False,
    "fused_epilogue_only_enabled": False,
    "attention_backend": "timm",
}


CUDA_BREAKDOWN_ORDER = [
    "unfused_attention",
    "linear_qkv",
    "linear_attention_out",
    "linear_dit_mlp_fc1",
    "mlp_gelu",
    "linear_dit_mlp_fc2",
    "adaln_modulation",
    "linear_block_adaln_modulation",
    "linear_final_adaln_modulation",
    "linear_final",
    "linear_timestep_fc1",
    "linear_timestep_fc2",
    "patch_embed_conv",
    "memset",
    "other_cuda",
]


CATEGORY_DESCRIPTIONS = {
    "adaln_modulation": "Original DiTBlock.modulate(norm(x), shift, scale).",
    "adaln_modulation_backward": "Backward reductions/pointwise work for AdaLN modulation and layernorm scale/shift/gate gradients.",
    "adaln_modulation_silu": "Original adaLN_modulation(c): SiLU activation.",
    "attention": "Original Attention forward: cuDNN SDPA/flash attention kernel.",
    "attention_wrapper": "Original Attention forward: PyTorch SDPA wrapper overhead.",
    "final_modulation": "Original FinalLayer modulate(norm_final(x), shift, scale).",
    "final_norm_layout": "Original final norm/layout around unpatchify/final layer.",
    "gated_residual": "Original x + gate.unsqueeze(1) * branch_out.",
    "gated_residual_backward": "Backward reductions/pointwise work for gated residual branch and gate gradients.",
    "layer_norm": "Original norm1/norm2 native layernorm work.",
    "layer_norm_backward": "Backward work for norm1/norm2 native layernorm.",
    "layer_norm_wrapper": "Original layernorm wrapper dispatch.",
    "layout_copy_contiguous": "Original clone/copy/contiguous layout materialization.",
    "layout_split_chunk": "Original chunk/split of adaLN modulation vectors.",
    "layout_transpose": "Original transpose/permute layout views.",
    "layout_view": "Original view/reshape/unsqueeze layout views.",
    "layout_reduction": "Compiled layout/copy/permute/stack plus reduction work.",
    "linear_gemm": "Original Linear GEMM/addmm; not fused by compile.",
    "linear_adaln_modulation_epilogue": "Compiled addmm epilogue and pointwise split/view work around AdaLN modulation linears.",
    "linear_block_adaln_modulation": "DiTBlock AdaLN modulation linear: hidden -> 6*hidden.",
    "linear_final_adaln_modulation": "FinalLayer AdaLN modulation linear: hidden -> 2*hidden.",
    "linear_attention_out": "Attention output projection linear.",
    "linear_final": "FinalLayer output projection linear.",
    "linear_dit_mlp_fc1": "DiT block MLP fc1 linear: hidden -> mlp_hidden.",
    "linear_dit_mlp_fc2": "DiT block MLP fc2 linear: mlp_hidden -> hidden.",
    "linear_qkv": "Attention qkv input projection linear: hidden -> 3*hidden.",
    "linear_timestep_fc1": "TimestepEmbedder fc1: frequency embedding -> hidden.",
    "linear_timestep_fc2": "TimestepEmbedder fc2: hidden -> hidden.",
    "linear_wrapper": "Original Linear wrapper dispatch around addmm.",
    "mlp_gelu": "Original timm Mlp GELU after fc1.",
    "patch_embed_bias": "Original patch embedding bias/add pointwise work around Conv3d.",
    "timestep_embedding": "TimestepEmbedder.timestep_embedding sin/cos frequency embedding.",
    "patch_embed_conv": "Original PatchEmbed3D Conv3d cuDNN kernel; not fused.",
    "patch_embed_conv_wrapper": "Original Conv3d wrapper dispatch overhead.",
    "pointwise_add": "Original eager pointwise adds not otherwise attributed.",
    "pointwise_mul": "Original eager pointwise multiplies not otherwise attributed.",
    "unknown_triton": "Compiled Triton kernel not present in the static map.",
}


def parse_spatial(value: str) -> tuple[int, int, int]:
    parts = value.replace(",", "x").replace(":", "x").split("x")
    if len(parts) == 1:
        side = int(parts[0])
        return side, side, side
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    raise argparse.ArgumentTypeError(f"spatial must be N or DxHxW; got {value!r}")


def spatial_from_tokens(tokens: int) -> tuple[int, int, int]:
    side = round(tokens ** (1.0 / 3.0))
    if side ** 3 == tokens:
        return side, side, side
    raise argparse.ArgumentTypeError(
        f"--tokens must be a perfect cube for this 3D DiT input path; got {tokens}. "
        "Pass --spatial DxHxW explicitly for non-cube token counts."
    )


def resolve_spatial(spatial: tuple[int, int, int] | None, tokens: int | None) -> tuple[int, int, int]:
    if spatial is None:
        return spatial_from_tokens(tokens) if tokens is not None else (8, 8, 8)
    if tokens is not None and spatial[0] * spatial[1] * spatial[2] != tokens:
        raise ValueError(f"--spatial {spatial} has {spatial[0] * spatial[1] * spatial[2]} tokens, not --tokens {tokens}")
    return spatial


def early_torchinductor_cache_dir(
    *,
    flag: str = "--compile-cache-dir",
    default_relative: str = "profile_artifacts/torchinductor_cache",
) -> tuple[Path, bool]:
    default = Path(__file__).resolve().parent / default_relative
    args = sys.argv[1:]
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return Path(args[idx + 1]).expanduser(), True
        if arg.startswith(f"{flag}="):
            return Path(arg.split("=", 1)[1]).expanduser(), True
    return default, False


def configure_torchinductor_cache(cache_dir: Path, *, explicit: bool) -> None:
    if explicit:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
    else:
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
