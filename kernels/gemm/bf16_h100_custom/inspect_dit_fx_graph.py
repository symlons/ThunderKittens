from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.fx
import torch.nn as nn
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from dit3d_e2e_bench import DiTBlock, dit_config
from dit_profile_utils import REGULAR_TIMM_BLOCK_KWARGS


class FunctionLog(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        print(f"Function Log: {resolve_name(func)}")
        return func(*args, **(kwargs or {}))


class DispatchLog(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print(f"Dispatch Log: {func}")
        return func(*args, **(kwargs or {}))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}")


def write_svg(graph_module: torch.fx.GraphModule, name: str, path: Path) -> None:
    try:
        drawer = FxGraphDrawer(graph_module, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(drawer.get_dot_graph().create_svg())
        print(f"wrote {path}")
    except Exception as exc:
        print(f"skip svg {path}: {type(exc).__name__}: {exc}")


def make_regular_block(model_name: str) -> nn.Module:
    cfg = dit_config(model_name)
    block = DiTBlock(cfg["hidden_size"], cfg["num_heads"], **REGULAR_TIMM_BLOCK_KWARGS)
    return block.eval()


def export_graphs(module: nn.Module, args: tuple[torch.Tensor, ...], out_dir: Path, name: str, core_aten: bool) -> None:
    try:
        traced = torch.fx.symbolic_trace(module)
        write_text(out_dir / f"{name}_symbolic_fx.txt", str(traced.graph))
        write_svg(traced, f"{name}_symbolic_fx", out_dir / f"{name}_symbolic_fx.svg")
    except Exception as exc:
        print(f"skip symbolic FX: {type(exc).__name__}: {exc}")

    try:
        exported = torch.export.export(module, args)
        exported_gm = exported.graph_module
        write_text(out_dir / f"{name}_export_aten.txt", str(exported_gm))
        write_svg(exported_gm, f"{name}_export_aten", out_dir / f"{name}_export_aten.svg")
    except Exception as exc:
        print(f"skip torch.export ATen: {type(exc).__name__}: {exc}")
        return

    if not core_aten:
        return
    try:
        core = exported.run_decompositions()
        core_gm = core.graph_module
        write_text(out_dir / f"{name}_core_aten.txt", str(core_gm))
        write_svg(core_gm, f"{name}_core_aten", out_dir / f"{name}_core_aten.svg")
    except Exception as exc:
        print(f"skip Core ATen export: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and visualize the regular timm DiT block FX/export graphs.")
    parser.add_argument("--model", choices=["S", "L", "XL"], default="L")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, default=Path("profile_artifacts/dit_fx_graphs"))
    parser.add_argument("--core-aten", action="store_true", help="Also export decomposed Core ATen graph.")
    parser.add_argument("--log-function", action="store_true", help="Log TorchFunctionMode calls.")
    parser.add_argument("--log-dispatch", action="store_true", help="Log TorchDispatchMode calls.")
    args = parser.parse_args()

    torch.manual_seed(123)
    block = make_regular_block(args.model)
    cfg = dit_config(args.model)
    hidden = cfg["hidden_size"]
    x = torch.randn(args.batch, args.tokens, hidden)
    c = torch.randn(args.batch, hidden)
    name = f"dit_{args.model.lower()}_block_b{args.batch}_t{args.tokens}"

    print(f"regular timm DiT-{args.model} block: batch={args.batch} tokens={args.tokens} hidden={hidden}")
    export_graphs(block, (x, c), args.out_dir, name, args.core_aten)

    if args.log_function:
        print("\nTorchFunctionMode logging:")
        with torch.inference_mode(), FunctionLog():
            block(x, c)

    if args.log_dispatch:
        print("\nTorchDispatchMode logging:")
        with torch.inference_mode(), DispatchLog():
            block(x, c)


if __name__ == "__main__":
    main()
