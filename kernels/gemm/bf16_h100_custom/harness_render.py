"""Console and markdown rendering for benchmark results."""
from typing import Optional

from tk_bench import (
    DEFAULT_WARMUP,
    DEFAULT_ITERS,
    BenchResult,
    l2_cache_size_bytes,
)

FULL_STEP_COMPONENTS = {"custom_fwdbwd_fused", "torch_eager_fwdbwd", "torch_compile_fwdbwd"}


def speedup(reference_us: Optional[float], candidate_us: Optional[float]) -> str:
    if not reference_us or not candidate_us or candidate_us <= 0:
        return "\u2014"
    return f"{reference_us / candidate_us:.2f}x"


def render_bench_console(results: list[BenchResult]) -> str:
    us_by_name = {r.name: r.us for r in results}
    buf = []
    buf.append(f"\n{'='*70}")
    buf.append("Benchmark Results")
    buf.append(f"{'='*70}")
    buf.append("")
    buf.append("Note: speedups are shown only for full fwd+bwd rows.")
    buf.append(f"\n{'Component':<24s}  {'Time (us)':>10}  {'TFLOPS':>8}  {'full vs eager':>14}  {'full vs compile':>15}")
    buf.append("-" * 70)
    for r in results:
        parts = f"{r.name:<24s}  {r.us:>10.1f}"
        tflops_str = f"{r.tflops:>8.0f}" if r.tflops is not None else "        "
        parts += f"  {tflops_str}"
        if r.name in FULL_STEP_COMPONENTS:
            if "eager" not in r.name and us_by_name.get("torch_eager_fwdbwd"):
                s = speedup(us_by_name["torch_eager_fwdbwd"], r.us)
                parts += f"  {s:>13s}x"
            elif "compile" not in r.name and us_by_name.get("torch_compile_fwdbwd"):
                s = speedup(us_by_name["torch_compile_fwdbwd"], r.us)
                parts += f"  {'':>8s}  {s:>13s}x"
            else:
                parts += "  " + " " * 14 + "  " + " " * 15
        else:
            parts += "  " + " " * 14 + "  " + " " * 15
        buf.append(parts)
    if "custom_bwd_unfused" in us_by_name and "custom_bwd_fused" in us_by_name:
        s = us_by_name["custom_bwd_unfused"] / us_by_name["custom_bwd_fused"]
        buf.append(f"{'fused vs unfused bwd':<24s}{'':>10}{'':>8}  {s:>5.2f}x{'':>10}")
    buf.append("")
    return "\n".join(buf)


def write_bench_report(path: str, results: list[BenchResult], M: int, K: int, N: int) -> None:
    us_by_name = {r.name: r.us for r in results}
    lines = [
        f"# Benchmark Report\n",
        f"Shape: M={M}, K={K}, N={N}, dtype=bf16\n",
        f"Convention: {DEFAULT_WARMUP} warmup, {DEFAULT_ITERS} iters, "
        f"L2-defeating groups, 2 CUDA events\n",
        f"L2 cache: {l2_cache_size_bytes() / (1024*1024):.0f} MB\n",
        "Speedups are shown only for rows that perform a full fwd+bwd step.\n",
        "",
        "| Component | Time (us) | TFLOPS | full vs eager | full vs compile |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        tflops = f"{r.tflops:.0f}" if r.tflops is not None else "\u2014"
        s_e = "\u2014"
        s_c = "\u2014"
        if r.name in FULL_STEP_COMPONENTS:
            if "eager" not in r.name and us_by_name.get("torch_eager_fwdbwd"):
                s_e = f"{us_by_name['torch_eager_fwdbwd'] / r.us:.2f}x"
            if "compile" not in r.name and us_by_name.get("torch_compile_fwdbwd"):
                s_c = f"{us_by_name['torch_compile_fwdbwd'] / r.us:.2f}x"
        lines.append(f"| {r.name} | {r.us:.1f} | {tflops} | {s_e} | {s_c} |")
    if "custom_bwd_unfused" in us_by_name and "custom_bwd_fused" in us_by_name:
        s = us_by_name["custom_bwd_unfused"] / us_by_name["custom_bwd_fused"]
        lines.append(f"| fused vs unfused bwd | \u2014 | \u2014 | {s:.2f}x | \u2014 |")
    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
