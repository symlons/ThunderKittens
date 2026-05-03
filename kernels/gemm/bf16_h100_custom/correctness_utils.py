"""
Reusable correctness testing primitives with multi-baseline comparison.

Concepts:
  Baseline      - a named reference tensor at a given precision
  TensorSpec    - describes one output tensor (name, dtype, custom value, baselines)
  Comparison    - the framework auto-generates all-vs-all pairwise comparisons
  Report        - auto-rendered Markdown table (stdout + optional file)
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import io
from textwrap import dedent

import torch


# ---- Input generation ----

def linear_inputs(M: int, K: int, N: int, seed: int = 42) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "x":   torch.randn(M, K, device="cuda", dtype=torch.bfloat16),
        "W":   torch.randn(K, N, device="cuda", dtype=torch.bfloat16),
        "b":   torch.randn(1, N, device="cuda", dtype=torch.bfloat16),
        "dy":  torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
    }


def gelu_bwd_inputs(M: int, N: int, seed: int = 42) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "preact":      torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
        "grad_output": torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
    }


# ---- Single pair comparison ----

@dataclass(frozen=True)
class PairwiseDiff:
    """Result of comparing two named tensors element-wise."""
    left: str                       # display name of left side
    right: str                      # display name of right side
    max_diff: float
    mean_diff: float

    @staticmethod
    def of(left: str, a: torch.Tensor, right: str, b: torch.Tensor) -> "PairwiseDiff":
        a_f = a.float() if a.dtype != torch.float32 else a
        b_f = b.float() if b.dtype != torch.float32 else b
        diff = (a_f - b_f).abs()
        return PairwiseDiff(left=left, right=right, max_diff=diff.max().item(), mean_diff=diff.mean().item())


# ---- Tensor with multiple baselines ----

@dataclass
class TensorSpec:
    """
    One output tensor (e.g. "dz") with a custom value and one or more baselines.

    The framework computes every pairwise diff between baselines and the custom
    tensor, so you always see how every choice of precision stacks up.
    """
    label: str                                  # "dz", "dW", etc.
    custom: torch.Tensor                        # custom kernel output
    baselines: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    atol_pass: float = 1.0                      # tolerance for custom vs fp32_pass baseline

    def add_baseline(self, name: str, tensor: torch.Tensor) -> "TensorSpec":
        self.baselines.append((name, tensor))
        return self

    def diffs(self) -> list[PairwiseDiff]:
        """Return diffs between custom and every baseline."""
        results = []
        for name, ref in self.baselines:
            results.append(PairwiseDiff.of("custom", self.custom, name, ref))
        return results

    def all_pairwise(self) -> list[PairwiseDiff]:
        """Return ALL pairwise diffs (custom + baselines vs each other)."""
        named: list[tuple[str, torch.Tensor]] = [("custom", self.custom)] + self.baselines
        results = []
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                ln, la = named[i]
                rn, rb = named[j]
                results.append(PairwiseDiff.of(ln, la, rn, rb))
        return results


# ---- Suite ----

@dataclass
class CorrectnessSuite:
    """Named collection of TensorSpecs — e.g. 'gelu_backward', 'unfused_bwd'."""
    name: str
    specs: list[TensorSpec] = field(default_factory=list)

    def add_tensor(self, label: str, custom: torch.Tensor, atol: float = 1.0) -> TensorSpec:
        spec = TensorSpec(label=label, custom=custom, atol_pass=atol)
        self.specs.append(spec)
        return spec

    def add_baseline(self, label: str, baseline_name: str, tensor: torch.Tensor) -> None:
        spec_map = {s.label: s for s in self.specs}
        if label in spec_map:
            spec_map[label].add_baseline(baseline_name, tensor)

    def has_pass_baseline(self, label: str) -> bool:
        spec_map = {s.label: s for s in self.specs}
        if label not in spec_map:
            return False
        names = [n for n, _ in spec_map[label].baselines]
        return "fp32_pass" in names


# ---- Report generation ----

@dataclass
class Report:
    """Collected suites and convenience methods for pretty printing."""
    shape_info: str = ""
    suites: list[CorrectnessSuite] = field(default_factory=list)
    timestamp: str = ""

    def add_suite(self, suite: CorrectnessSuite) -> "Report":
        self.suites.append(suite)
        return self

    # -- console table --

    def console_table(self, all_pairwise: bool = False) -> str:
        """Return a terminal-friendly comparison table (one suite per section)."""
        buf = io.StringIO()
        buf.write("=" * 72 + "\n")
        buf.write("  CORRECTNESS — MULTI-BASELINE COMPARISON\n")
        if self.shape_info:
            buf.write(f"  Shape: {self.shape_info}\n")
        buf.write(f"  Time:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        buf.write("=" * 72 + "\n\n")

        for suite in self.suites:
            buf.write(f"--- {suite.name} ---\n")
            for spec in suite.specs:
                buf.write(f"\n  Tensor: {spec.label}  (dtype: {spec.custom.dtype})\n")

                if all_pairwise:
                    diffs = spec.all_pairwise()
                else:
                    diffs = spec.diffs()

                if not diffs:
                    buf.write("    (no baselines)\n")
                    continue

                header = f"    {'Pair':30s}  {'max_diff':>10s}  {'mean_diff':>12s}\n"
                buf.write(header)
                buf.write("    " + "-" * 54 + "\n")
                for d in diffs:
                    pair = f"{d.left} vs {d.right}"
                    buf.write(f"    {pair:30s}  {d.max_diff:10.6f}  {d.mean_diff:12.6e}\n")

                buf.write("\n")

            buf.write("\n")

        buf.write("=" * 72 + "\n")
        return buf.getvalue()

    # -- Markdown table (for report file) --

    def markdown_table(self, all_pairwise: bool = False) -> str:
        """Return a Markdown document with one table per tensor."""
        lines: list[str] = []
        lines.append("# Correctness Report\n")
        lines.append(f"Auto-generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
        if self.shape_info:
            lines.append(f"**Shape:** {self.shape_info}\n")

        for suite in self.suites:
            lines.append(f"## {suite.name}\n")
            for spec in suite.specs:
                lines.append(f"### Tensor `{spec.label}`  (dtype: `{spec.custom.dtype}`)\n")
                if all_pairwise:
                    diffs = spec.all_pairwise()
                else:
                    diffs = spec.diffs()

                if not diffs:
                    lines.append("*(no baselines)*\n")
                    continue

                lines.append("| Pair | Max Diff | Mean Diff |")
                lines.append("|---|---:|---:|")
                for d in diffs:
                    lines.append(f"| {d.left} vs {d.right} | {d.max_diff:.6f} | {d.mean_diff:.6e} |")
                lines.append("")

        return "\n".join(lines)

    def write_markdown(self, path: str, all_pairwise: bool = False) -> None:
        with open(path, "w") as f:
            f.write(self.markdown_table(all_pairwise=all_pairwise))
