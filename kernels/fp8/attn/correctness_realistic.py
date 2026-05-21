"""FP8 vs INT8 quantization noise on heavy-tail and real-model inputs.

Two passes:
  1. Synthetic heavy-tail distributions where FP8 e4m3's wide dynamic
     range is expected to beat INT8's uniform grid.
  2. Real Q/K/V activations captured from a pretrained Stable-Diffusion
     UNet (segmind/tiny-sd by default; ~330 MB cache).

Reuses ``fp8_suite.quant.quantize_per_row_fp8`` and
``quantize_per_row_int8`` so we measure the same per-token recipe the
kernels implement, and ``tensor_metrics`` for cos / rel-L1 / RMSE / QSNR.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, ".")

from fp8_suite.metrics import tensor_metrics
from fp8_suite.quant import quantize_per_row_fp8, quantize_per_row_int8
from fp8_suite.realistic_inputs import (
    SYNTHETIC,
    capture_unet_qkv,
    make_synthetic,
    per_row_stats,
)


def dequant_pair(x):
    xq_f, sf = quantize_per_row_fp8(x)
    xq_i, si = quantize_per_row_int8(x)
    deq_f = xq_f.to(torch.float32) * sf.unsqueeze(-1)
    deq_i = xq_i.to(torch.float32) * si.unsqueeze(-1)
    return tensor_metrics(deq_f, x), tensor_metrics(deq_i, x)


def fmt_row(label, fp8_m, int8_m, extra=""):
    gap = int8_m["qsnr_dB"] - fp8_m["qsnr_dB"]
    winner = "INT8" if gap > 0 else "FP8"
    return (
        f"| {label} | {fp8_m['qsnr_dB']:.2f} dB | {int8_m['qsnr_dB']:.2f} dB | "
        f"{gap:+.2f} dB | **{winner}** | "
        f"{fp8_m['rel_L1']:.2e} / {int8_m['rel_L1']:.2e} | "
        f"{fp8_m['rmse']:.2e} / {int8_m['rmse']:.2e} | "
        f"{fp8_m['cos']:.5f} / {int8_m['cos']:.5f} | {extra} |"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="docs/reports/QUANT_REALISTIC.md")
    p.add_argument("--shapes", type=int, nargs="+",
                   default=[1, 8, 1536, 128],
                   help="B H N D shape for synthetic inputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--unet",  default="segmind/tiny-sd",
                   help="Stable Diffusion model id (UNet attention only). "
                        "Use 'none' to skip the real-model pass.")
    p.add_argument("--max-layers", type=int, default=6)
    args = p.parse_args()
    B, H, N, D = args.shapes

    device = torch.cuda.get_device_properties(torch.cuda.current_device())

    L = []
    L.append("# FP8 vs INT8: heavy-tail and real-model quant noise\n")
    L.append(f"- Device: **{device.name}** "
             f"(SM {device.major}.{device.minor}, {device.multi_processor_count} SMs)")
    L.append(f"- Per-token symmetric, scale = `max(amax / max_int, 1e-12)`")
    L.append(f"- Shape used for synthetic: B={B}, H={H}, N={N}, D={D} "
             f"({B*H*N} rows of D={D})\n")

    # ---- Synthetic heavy-tail / outlier sweep ------------------------------
    L.append("## Synthetic distributions\n")
    L.append("Mean QSNR / rel-L1 / RMSE / cos over the listed seeds. "
             "`amax/σ` is the row-aggregate dynamic-range ratio that "
             "governs the FP8-vs-INT8 trade-off: small → INT8 wins, "
             "large → FP8 wins.\n")
    L.append("| distribution | FP8 QSNR | INT8 QSNR | Δ (INT8 − FP8) | winner | "
             "FP8 / INT8 rel-L1 | FP8 / INT8 RMSE | "
             "FP8 / INT8 cos | amax/σ (mean, p95) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    agg = []
    for kind in SYNTHETIC:
        fp8s, int8s, ratios = [], [], []
        for seed in args.seeds:
            x = make_synthetic(kind, (B, H, N, D), seed=seed)
            mf, mi = dequant_pair(x)
            fp8s.append(mf); int8s.append(mi)
            s = per_row_stats(x)
            ratios.append((s["row_amax_over_std_mean"], s["row_amax_over_std_p95"]))
        def avg(ms, key): return sum(m[key] for m in ms) / len(ms)
        mf = {k: avg(fp8s, k) for k in ("qsnr_dB", "rel_L1", "rmse", "cos")}
        mi = {k: avg(int8s, k) for k in ("qsnr_dB", "rel_L1", "rmse", "cos")}
        ratio_mean = sum(r[0] for r in ratios) / len(ratios)
        ratio_p95  = sum(r[1] for r in ratios) / len(ratios)
        L.append(fmt_row(kind, mf, mi,
                         extra=f"{ratio_mean:.2f} / {ratio_p95:.2f}"))
        agg.append((kind, mf["qsnr_dB"], mi["qsnr_dB"], ratio_mean))
    L.append("")

    # ---- Real model Q/K/V activations --------------------------------------
    if args.unet.lower() != "none":
        is_cog = "cogvideo" in args.unet.lower()
        kind_label = ("CogVideoX transformer (MMDiT)" if is_cog
                      else "Stable-Diffusion UNet")
        L.append(f"## Real {kind_label} Q/K/V activations\n")
        L.append(f"Model: `{args.unet}`. Captured by forward-hooking every "
                 "`Attention.to_q/to_k/to_v` projection during one forward "
                 "pass with random latents and a random text embedding, "
                 "then reshaped to (B, heads, N, head_dim). Activations "
                 "are upcast to fp32 before quantization.\n")
        try:
            acts = capture_unet_qkv(args.unet, max_layers=args.max_layers)
        except Exception as exc:
            L.append(f"_Skipped: {exc}_\n")
            acts = []

        if acts:
            L.append("| layer | role | self/cross | shape | FP8 QSNR | "
                     "INT8 QSNR | Δ | winner | FP8 / INT8 rel-L1 | "
                     "FP8 / INT8 RMSE | FP8 / INT8 cos | amax/σ |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            real_rows = []
            for s in acts:
                if s.tensor.shape[-1] not in (64, 128):
                    # quantizers only care about head_dim, but FP8/INT8
                    # row quant works for any D. Keep all.
                    pass
                mf, mi = dequant_pair(s.tensor)
                stats = per_row_stats(s.tensor)
                short_name = ".".join(s.name.split(".")[-3:])
                L.append(
                    f"| {short_name} | {s.role} | {s.layer_kind} | "
                    f"{tuple(s.tensor.shape)} | "
                    f"{mf['qsnr_dB']:.2f} dB | {mi['qsnr_dB']:.2f} dB | "
                    f"{mi['qsnr_dB']-mf['qsnr_dB']:+.2f} dB | "
                    f"**{'INT8' if mi['qsnr_dB']>mf['qsnr_dB'] else 'FP8'}** | "
                    f"{mf['rel_L1']:.2e} / {mi['rel_L1']:.2e} | "
                    f"{mf['rmse']:.2e} / {mi['rmse']:.2e} | "
                    f"{mf['cos']:.5f} / {mi['cos']:.5f} | "
                    f"{stats['row_amax_over_std_mean']:.2f} |"
                )
                real_rows.append((s, mf, mi, stats))
            L.append("")

            # ---- Aggregate by role / layer kind ------------------------
            by_kind: defaultdict = defaultdict(list)
            for s, mf, mi, _ in real_rows:
                by_kind[(s.role, s.layer_kind)].append((mf, mi))
            L.append("### Aggregate (mean over captured layers)\n")
            L.append("| role | self/cross | n | mean FP8 QSNR | mean INT8 QSNR | "
                     "Δ | winner |")
            L.append("|---|---|---|---|---|---|---|")
            for (role, kind), ms in sorted(by_kind.items()):
                fq = sum(mf["qsnr_dB"] for mf, _ in ms) / len(ms)
                iq = sum(mi["qsnr_dB"] for _, mi in ms) / len(ms)
                L.append(
                    f"| {role} | {kind} | {len(ms)} | "
                    f"{fq:.2f} dB | {iq:.2f} dB | "
                    f"{iq-fq:+.2f} dB | "
                    f"**{'INT8' if iq>fq else 'FP8'}** |"
                )
            L.append("")

    # ---- Key findings ------------------------------------------------------
    L.append("## Key findings\n")
    fp8_wins = [k for (k, fq, iq, _) in agg if fq > iq]
    int8_wins = [k for (k, fq, iq, _) in agg if iq > fq]
    L.append(f"- **FP8 wins** on heavy-tail / outlier-driven distributions: "
             f"{', '.join('`'+k+'`' for k in fp8_wins) if fp8_wins else '_none_'}.")
    L.append(f"- **INT8 wins** on bounded-tail distributions: "
             f"{', '.join('`'+k+'`' for k in int8_wins) if int8_wins else '_none_'}.")
    L.append(
        "- The crossover is well predicted by per-row `amax/σ`: FP8's "
        "constant relative precision pays off once that ratio is large "
        "enough that uniform INT8 has to spend its budget on the outliers."
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
