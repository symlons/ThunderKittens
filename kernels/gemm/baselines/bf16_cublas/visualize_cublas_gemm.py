from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GemmPoint:
    m: int
    n: int
    k: int
    us: float
    tflops: float
    cta_m: int
    cta_n: int
    ctas: int
    waves: int
    wave_fill: float
    tail_ctas: int


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def default_axis() -> list[int]:
    values = set()
    values.update(range(256, 2304, 128))
    values.update(range(2304, 4608, 256))
    values.update(range(4608, 12289, 512))
    return sorted(values)


def input_group_count(input_bytes: int, l2_bytes: int) -> int:
    return 1 if input_bytes >= 3 * l2_bytes else int(3 * l2_bytes / input_bytes) + 1


def make_groups(m: int, n: int, k: int, groups_n: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    groups = []
    for group_idx in range(groups_n):
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1000 + 17 * group_idx + m + 3 * n + 5 * k)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16, generator=gen)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16, generator=gen)
        groups.append((a, b))
    return groups


def benchmark_one(
    m: int,
    n: int,
    k: int,
    *,
    sm_count: int,
    l2_bytes: int,
    tile_m: int,
    tile_n: int,
    warmup: int,
    iters: int,
    repeats: int,
    max_groups: int,
) -> GemmPoint:
    bytes_per_group = 2 * (m * k + k * n + m * n)
    groups_n = min(max_groups, input_group_count(bytes_per_group, l2_bytes))
    groups = make_groups(m, n, k, groups_n)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    def run(idx: int) -> None:
        a, b = groups[idx % groups_n]
        torch.mm(a, b, out=out)

    torch.cuda.synchronize()
    for i in range(warmup):
        run(i)
    torch.cuda.synchronize()

    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        start.record()
        for i in range(iters):
            run(repeat * iters + i)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)

    us = statistics.median(samples)
    tflops = (2.0 * m * n * k) / (us * 1e-6) / 1e12
    cta_m = math.ceil(m / tile_m)
    cta_n = math.ceil(n / tile_n)
    ctas = cta_m * cta_n
    waves = max(1, math.ceil(ctas / sm_count))
    tail_ctas = ctas - (waves - 1) * sm_count
    wave_fill = ctas / (waves * sm_count)
    return GemmPoint(m, n, k, us, tflops, cta_m, cta_n, ctas, waves, wave_fill, tail_ctas)


def read_csv(path: Path) -> list[GemmPoint]:
    points = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            points.append(
                GemmPoint(
                    m=int(row["m"]),
                    n=int(row["n"]),
                    k=int(row["k"]),
                    us=float(row["us"]),
                    tflops=float(row["tflops"]),
                    cta_m=int(row["cta_m"]),
                    cta_n=int(row["cta_n"]),
                    ctas=int(row["ctas"]),
                    waves=int(row["waves"]),
                    wave_fill=float(row["wave_fill"]),
                    tail_ctas=int(row["tail_ctas"]),
                )
            )
    return points


def write_csv(path: Path, points: list[GemmPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def color_scale(value: float, lo: float, hi: float, *, invert: bool = False) -> str:
    if hi <= lo:
        t = 1.0
    else:
        t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
    if invert:
        t = 1.0 - t
    stops = [
        (29, 43, 83),
        (58, 96, 115),
        (110, 154, 118),
        (194, 178, 92),
        (225, 111, 74),
    ]
    pos = t * (len(stops) - 1)
    i = min(len(stops) - 2, int(pos))
    frac = pos - i
    a = stops[i]
    b = stops[i + 1]
    rgb = tuple(round(a[j] + frac * (b[j] - a[j])) for j in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def svg_heatmap(
    points: list[GemmPoint],
    *,
    metric: str,
    title: str,
    subtitle: str,
    width: int = 1160,
    height: int = 760,
) -> str:
    xs = sorted({p.m for p in points})
    ys = sorted({p.n for p in points})
    by_shape = {(p.m, p.n): p for p in points}
    values = [float(getattr(p, metric)) for p in points]
    lo, hi = min(values), max(values)
    margin_l, margin_r, margin_t, margin_b = 88, 28, 82, 86
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    cell_w = plot_w / len(xs)
    cell_h = plot_h / len(ys)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_l}" y="34" class="chart-title">{html.escape(title)}</text>',
        f'<text x="{margin_l}" y="58" class="chart-subtitle">{html.escape(subtitle)}</text>',
    ]
    for yi, n in enumerate(ys):
        for xi, m in enumerate(xs):
            point = by_shape.get((m, n))
            if point is None:
                continue
            value = float(getattr(point, metric))
            x = margin_l + xi * cell_w
            y = margin_t + (len(ys) - 1 - yi) * cell_h
            fill = color_scale(value, lo, hi, invert=(metric == "us"))
            label = (
                f"M={point.m} N={point.n} K={point.k}\\n"
                f"{point.tflops:.1f} TFLOP/s, {point.us:.2f} us\\n"
                f"CTAs={point.ctas}, waves={point.waves}, tail={point.tail_ctas}, fill={point.wave_fill:.1%}"
            )
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.5:.2f}" height="{cell_h + 0.5:.2f}" '
                f'fill="{fill}"><title>{html.escape(label)}</title></rect>'
            )
            if point.wave_fill < 0.72:
                parts.append(
                    f'<circle cx="{x + cell_w / 2:.2f}" cy="{y + cell_h / 2:.2f}" '
                    f'r="{max(2.0, min(cell_w, cell_h) * 0.18):.2f}" class="tail-dot">'
                    f'<title>{html.escape(label)}</title></circle>'
                )
    for xi, m in enumerate(xs):
        if xi % max(1, len(xs) // 12) == 0 or xi == len(xs) - 1:
            x = margin_l + xi * cell_w + cell_w / 2
            parts.append(f'<text x="{x:.2f}" y="{height - 48}" class="tick" transform="rotate(45 {x:.2f} {height - 48})">{m}</text>')
    for yi, n in enumerate(ys):
        if yi % max(1, len(ys) // 10) == 0 or yi == len(ys) - 1:
            y = margin_t + (len(ys) - 1 - yi) * cell_h + cell_h / 2
            parts.append(f'<text x="{margin_l - 12}" y="{y + 4:.2f}" text-anchor="end" class="tick">{n}</text>')
    parts.extend(
        [
            f'<text x="{margin_l + plot_w / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis">M dimension</text>',
            f'<text x="22" y="{margin_t + plot_h / 2:.2f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {margin_t + plot_h / 2:.2f})">N dimension</text>',
            f'<text x="{width - margin_r - 200}" y="36" class="legend">range: {lo:.2f} to {hi:.2f}</text>',
            f'<text x="{width - margin_r - 200}" y="58" class="legend">hollow dots: estimated underfilled final wave</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def svg_sweep(points: list[GemmPoint], *, width: int = 1160, height: int = 560) -> str:
    ks = sorted({p.k for p in points})
    if len(ks) != 1:
        title = "cuBLAS BF16 GEMM shape sweep"
    else:
        title = f"cuBLAS BF16 GEMM M sweep at K={ks[0]}"
    ns = sorted({p.n for p in points})
    xs = sorted({p.m for p in points})
    max_tflops = max(p.tflops for p in points)
    margin_l, margin_r, margin_t, margin_b = 82, 34, 68, 72
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    colors = ["#e06f4a", "#c2b25c", "#6e9a76", "#3a6073", "#6d5a8c", "#b35676"]

    def sx(m: int) -> float:
        return margin_l + (m - xs[0]) / max(1, xs[-1] - xs[0]) * plot_w

    def sy(v: float) -> float:
        return margin_t + (1.0 - v / max_tflops) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_l}" y="34" class="chart-title">{html.escape(title)}</text>',
        f'<text x="{margin_l}" y="56" class="chart-subtitle">Line breaks and dips line up with changes in CTA waves and poorly filled final waves.</text>',
        f'<line x1="{margin_l}" y1="{margin_t + plot_h}" x2="{margin_l + plot_w}" y2="{margin_t + plot_h}" class="axis-line"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t + plot_h}" class="axis-line"/>',
    ]
    for tick in range(0, 6):
        value = max_tflops * tick / 5
        y = sy(value)
        parts.append(f'<line x1="{margin_l}" y1="{y:.2f}" x2="{margin_l + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y + 4:.2f}" text-anchor="end" class="tick">{value:.0f}</text>')
    for xi, m in enumerate(xs):
        if xi % max(1, len(xs) // 10) == 0 or xi == len(xs) - 1:
            x = sx(m)
            parts.append(f'<line x1="{x:.2f}" y1="{margin_t}" x2="{x:.2f}" y2="{margin_t + plot_h}" class="grid"/>')
            parts.append(f'<text x="{x:.2f}" y="{height - 38}" class="tick" text-anchor="middle">{m}</text>')
    for idx, n in enumerate(ns):
        series = sorted((p for p in points if p.n == n), key=lambda p: p.m)
        color = colors[idx % len(colors)]
        path = " ".join(("M" if i == 0 else "L") + f"{sx(p.m):.2f},{sy(p.tflops):.2f}" for i, p in enumerate(series))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        for p in series:
            r = 4.5 if p.wave_fill < 0.72 else 2.7
            cls = "sweep-point low-fill" if p.wave_fill < 0.72 else "sweep-point"
            label = (
                f"M={p.m} N={p.n} K={p.k}\\n"
                f"{p.tflops:.1f} TFLOP/s, {p.us:.2f} us\\n"
                f"CTAs={p.ctas}, waves={p.waves}, tail={p.tail_ctas}, fill={p.wave_fill:.1%}"
            )
            parts.append(
                f'<circle cx="{sx(p.m):.2f}" cy="{sy(p.tflops):.2f}" r="{r}" '
                f'fill="{color}" class="{cls}"><title>{html.escape(label)}</title></circle>'
            )
        parts.append(f'<text x="{margin_l + plot_w - 70}" y="{margin_t + 22 + idx * 18}" class="legend" fill="{color}">N={n}</text>')
    parts.extend(
        [
            f'<text x="{margin_l + plot_w / 2:.2f}" y="{height - 8}" text-anchor="middle" class="axis">M dimension</text>',
            f'<text x="20" y="{margin_t + plot_h / 2:.2f}" text-anchor="middle" class="axis" transform="rotate(-90 20 {margin_t + plot_h / 2:.2f})">TFLOP/s</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def write_html(path: Path, points: list[GemmPoint], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps([asdict(point) for point in points], indent=2)
    best = max(points, key=lambda p: p.tflops)
    worst_fill = min(points, key=lambda p: p.wave_fill)
    heat_tflops = svg_heatmap(
        points,
        metric="tflops",
        title=f"cuBLAS BF16 GEMM throughput heatmap, K={points[0].k}",
        subtitle="Color is measured TFLOP/s. Hollow markers identify shapes with an underfilled estimated final CTA wave.",
    )
    heat_fill = svg_heatmap(
        points,
        metric="wave_fill",
        title="Estimated wave quantization",
        subtitle=f"Assumes {args.tile_m}x{args.tile_n} CTA tiles over {args.sm_count} SMs; use this as a cliff locator, not a cuBLAS algorithm claim.",
    )
    sweep = svg_sweep(points)
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cuBLAS GEMM Shape Cliffs</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #11161a;
  --panel: #171f23;
  --ink: #edf2ef;
  --muted: #9fb1ac;
  --grid: rgba(237, 242, 239, 0.12);
  --line: rgba(237, 242, 239, 0.55);
}}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
main {{
  max-width: 1240px;
  margin: 0 auto;
  padding: 28px 20px 42px;
}}
h1 {{
  margin: 0 0 12px;
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 0;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0 24px;
}}
.stat {{
  background: var(--panel);
  border: 1px solid rgba(237, 242, 239, 0.11);
  border-radius: 8px;
  padding: 12px 14px;
}}
.label {{
  color: var(--muted);
  font-size: 12px;
}}
.value {{
  margin-top: 4px;
  font-size: 19px;
  font-weight: 700;
}}
section {{
  margin-top: 22px;
  background: var(--panel);
  border: 1px solid rgba(237, 242, 239, 0.11);
  border-radius: 8px;
  overflow: auto;
}}
svg {{
  display: block;
  width: 100%;
  min-width: 940px;
  height: auto;
}}
.chart-title {{ fill: var(--ink); font-size: 23px; font-weight: 700; }}
.chart-subtitle, .legend {{ fill: var(--muted); font-size: 13px; }}
.axis {{ fill: var(--muted); font-size: 13px; font-weight: 600; }}
.tick {{ fill: var(--muted); font-size: 11px; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis-line {{ stroke: var(--line); stroke-width: 1.2; }}
.tail-dot {{
  fill: none;
  stroke: #edf2ef;
  stroke-width: 2;
  opacity: 0.9;
}}
.sweep-point {{
  stroke: #11161a;
  stroke-width: 1.2;
}}
.low-fill {{
  stroke: #edf2ef;
  stroke-width: 2;
}}
details {{
  margin-top: 18px;
  color: var(--muted);
}}
pre {{
  overflow: auto;
  background: #0d1114;
  border-radius: 8px;
  padding: 14px;
  color: #dce5e1;
}}
@media (max-width: 760px) {{
  .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
</style>
</head>
<body>
<main>
<h1>cuBLAS GEMM Shape Cliffs</h1>
<div class="summary">
  <div class="stat"><div class="label">points</div><div class="value">{len(points)}</div></div>
  <div class="stat"><div class="label">best measured</div><div class="value">{best.tflops:.1f} TFLOP/s</div></div>
  <div class="stat"><div class="label">best shape</div><div class="value">M={best.m} N={best.n}</div></div>
  <div class="stat"><div class="label">lowest wave fill</div><div class="value">{worst_fill.wave_fill:.1%}</div></div>
</div>
<section>{heat_tflops}</section>
<section>{heat_fill}</section>
<section>{sweep}</section>
<details>
<summary>Embedded benchmark data</summary>
<pre>{html.escape(payload)}</pre>
</details>
</main>
</body>
</html>
"""
    path.write_text(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark and visualize BF16 cuBLAS GEMM shape cliffs.")
    parser.add_argument("--m", type=parse_int_list, default=default_axis(), help="Comma-separated M values.")
    parser.add_argument("--n", type=parse_int_list, default=default_axis(), help="Comma-separated N values.")
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--tile-m", type=int, default=128, help="Estimated CTA tile M for wave-fill visualization.")
    parser.add_argument("--tile-n", type=int, default=128, help="Estimated CTA tile N for wave-fill visualization.")
    parser.add_argument("--sm-count", type=int, help="Override SM count used for wave-fill labels.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--csv", type=Path, default=Path("cublas_gemm_shape_sweep.csv"))
    parser.add_argument("--html", type=Path, default=Path("cublas_gemm_shape_sweep.html"))
    parser.add_argument("--input-csv", type=Path, help="Skip benchmarking and render an existing CSV.")
    args = parser.parse_args()

    if args.input_csv:
        points = read_csv(args.input_csv)
        if args.sm_count is None:
            first = points[0]
            args.sm_count = round(first.ctas / (first.waves * first.wave_fill))
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required unless --input-csv is provided")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties(0)
        if args.sm_count is None:
            args.sm_count = props.multi_processor_count
        l2_bytes = props.L2_cache_size
        print(f"gpu={props.name} sm_count={args.sm_count} l2={l2_bytes} torch={torch.__version__}", flush=True)
        points = []
        total = len(args.m) * len(args.n)
        shape_iter = ((m, n) for n in args.n for m in args.m)
        for idx, (m, n) in enumerate(shape_iter, start=1):
            point = benchmark_one(
                m,
                n,
                args.k,
                sm_count=args.sm_count,
                l2_bytes=l2_bytes,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                max_groups=args.max_groups,
            )
            points.append(point)
            print(
                f"[{idx:4d}/{total}] M={m:5d} N={n:5d} K={args.k:5d} "
                f"{point.us:8.2f} us {point.tflops:7.1f} TFLOP/s "
                f"waves={point.waves:3d} fill={point.wave_fill:5.1%}",
                flush=True,
            )
        write_csv(args.csv, points)
        print(f"wrote {args.csv}", flush=True)
        time.sleep(0.2)

    write_html(args.html, points, args)
    print(f"wrote {args.html}", flush=True)


if __name__ == "__main__":
    main()
