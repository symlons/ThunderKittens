from pathlib import Path

import torch

from smooth_core import (
    bkv,
    bq,
    cw,
    fake_quant_k,
    fake_quant_q,
    fp8_quantize_v_per_channel,
    qk_granularities,
    qk_modes,
    quantized_values,
)
from smooth_report import color, title_color


# Logical ordering for smoothing combinations (from "do nothing" up to "all").
SMOOTH_ORDER = ["none", "Q", "K", "V", "Q+K", "Q+V", "K+V", "Q+K+V"]


def _qsnr_db(x, xhat, dim=None):
    x = x.detach().float()
    noise = (x - xhat.detach().float())
    if dim is None:
        signal_power = (x * x).sum()
        noise_power = (noise * noise).sum()
    else:
        signal_power = (x * x).sum(dim=dim)
        noise_power = (noise * noise).sum(dim=dim)
    return 10.0 * torch.log10(signal_power.clamp_min(1e-30) / noise_power.clamp_min(1e-30))


def _reconstruct_q(x, mode, granularity="per_thread"):
    y = torch.empty_like(x, dtype=torch.float32)
    for qs in range(0, x.shape[0], bq):
        qe = min(qs + bq, x.shape[0])
        y[qs:qe] = fake_quant_q(x[qs:qe], mode, granularity, cw)
    return y


def _reconstruct_k(x, mode, granularity="per_thread"):
    y = torch.empty_like(x, dtype=torch.float32)
    for ks in range(0, x.shape[0], bkv):
        ke = min(ks + bkv, x.shape[0])
        y[ks:ke] = fake_quant_k(x[ks:ke], mode, granularity)
    return y


def _reconstruct_v(x):
    vhat, scale = fp8_quantize_v_per_channel(x)
    return vhat * scale


def _q_smooth(x):
    y = torch.empty_like(x)
    for qs in range(0, x.shape[0], bq):
        qe = min(qs + bq, x.shape[0])
        block = x[qs:qe]
        y[qs:qe] = block - block.mean(dim=0, keepdim=True)
    return y


def _smooth_reconstruction_cases(Q, K, V, qk_mode):
    Qg = _q_smooth(Q)
    Kg = K - K.mean(dim=0, keepdim=True)
    Vg = V - V.mean(dim=0, keepdim=True)
    return [
        ("Q", "before", Q, _reconstruct_q(Q, qk_mode)),
        ("Q", "after smoothing", Qg, _reconstruct_q(Qg, qk_mode)),
        ("K", "before", K, _reconstruct_k(K, qk_mode)),
        ("K", "after smoothing", Kg, _reconstruct_k(Kg, qk_mode)),
        ("V", "before", V, _reconstruct_v(V)),
        ("V", "after smoothing", Vg, _reconstruct_v(Vg)),
    ]


# ---------------------------------------------------------------------------
# Generic helpers for cleaner heatmaps
# ---------------------------------------------------------------------------


def _luminance(rgba):
    # Rec.709 luminance from RGBA in [0,1]
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _annotate_heatmap(ax, matrix, fmt, cmap, vmin, vmax, fontsize=7, mask=None):
    """Annotate every cell with luminance-aware text color."""
    import numpy as np
    norm = lambda v: (v - vmin) / max(vmax - vmin, 1e-12)
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            v = matrix[i, j]
            if np.isnan(v) or (mask is not None and mask[i, j]):
                continue
            rgba = cmap(norm(v))
            color_text = "white" if _luminance(rgba) < 0.55 else "black"
            ax.text(j, i, fmt(v), ha="center", va="center",
                    color=color_text, fontsize=fontsize)


def _smooth_order(present):
    seen = set(present)
    ordered = [s for s in SMOOTH_ORDER if s in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


# ---------------------------------------------------------------------------
# Tensor visualisations (heatmaps, channel means, hist, QSNR)
# ---------------------------------------------------------------------------


def _save_qsnr_plot(Q, K, V, qk_mode, plot_dir, plt):
    cases = _smooth_reconstruction_cases(Q, K, V, qk_mode)
    tensor_names = ["Q", "K", "V"]
    labels = ["before", "after smoothing"]
    colors = {"before": "#4C78A8", "after smoothing": "#F58518"}
    channel_qsnr_by_tensor = {tensor_name: {} for tensor_name in tensor_names}
    total_qsnr_by_tensor = {tensor_name: {} for tensor_name in tensor_names}

    for name, label, x, xhat in cases:
        channel_qsnr_by_tensor[name][label] = _qsnr_db(x, xhat, dim=0).detach().cpu().numpy()
        total_qsnr_by_tensor[name][label] = _qsnr_db(x, xhat).item()

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, tensor_name in zip(axes, tensor_names):
        for label in labels:
            channel_qsnr = channel_qsnr_by_tensor[tensor_name][label]
            total_qsnr = total_qsnr_by_tensor[tensor_name][label]
            ax.plot(channel_qsnr, label=f"{label} ({total_qsnr:.1f} dB)", color=colors[label])
        ax.set_title(f"{tensor_name} reconstruction QSNR  [{qk_mode}]")
        ax.set_ylabel("QSNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("channel")
    fig.savefig(plot_dir / f"reconstruction_qsnr_{qk_mode}.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for ax, tensor_name in zip(axes, tensor_names):
        for label in labels:
            ax.hist(
                channel_qsnr_by_tensor[tensor_name][label],
                bins=32,
                density=True,
                histtype="stepfilled",
                alpha=0.35,
                label=label,
                color=colors[label],
            )
        ax.set_title(f"{tensor_name} reconstruction QSNR density  [{qk_mode}]")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("QSNR (dB)")
    fig.savefig(plot_dir / f"reconstruction_qsnr_density_{qk_mode}.png", dpi=160)
    plt.close(fig)


def _save_granularity_qsnr(Q, K, qk_mode, plot_dir, plt):
    """Per-channel reconstruction QSNR for Q,K across different granularities."""
    grans = qk_granularities
    palette = {
        "per_tensor": "#1f77b4",
        "per_block": "#ff7f0e",
        "per_token": "#2ca02c",
        "per_thread": "#d62728",
    }
    qsnr_q = {g: _qsnr_db(Q, _reconstruct_q(Q, qk_mode, g), dim=0).cpu().numpy() for g in grans}
    qsnr_k = {g: _qsnr_db(K, _reconstruct_k(K, qk_mode, g), dim=0).cpu().numpy() for g in grans}
    total_q = {g: _qsnr_db(Q, _reconstruct_q(Q, qk_mode, g)).item() for g in grans}
    total_k = {g: _qsnr_db(K, _reconstruct_k(K, qk_mode, g)).item() for g in grans}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for g in grans:
        axes[0].plot(qsnr_q[g], label=f"{g} ({total_q[g]:.1f} dB)", color=palette[g])
        axes[1].plot(qsnr_k[g], label=f"{g} ({total_k[g]:.1f} dB)", color=palette[g])
    axes[0].set_title(f"Q reconstruction QSNR by granularity  [{qk_mode}]")
    axes[1].set_title(f"K reconstruction QSNR by granularity  [{qk_mode}]")
    for ax in axes:
        ax.set_ylabel("QSNR (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("channel")
    fig.savefig(plot_dir / f"granularity_qsnr_{qk_mode}.png", dpi=160)
    plt.close(fig)


def save_tensor_plots(Q, K, V, qk_mode, plot_dir):
    if qk_mode not in qk_modes:
        raise ValueError(f"unknown Q/K dtype {qk_mode}; valid choices are {qk_modes}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    Qg = Q - Q.mean(dim=0, keepdim=True)
    Kg = K - K.mean(dim=0, keepdim=True)
    Vg = V - V.mean(dim=0, keepdim=True)

    tensors = [
        ("Q", Q, Qg),
        ("K", K, Kg),
        ("V", V, Vg),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
    for row, (name, before, after) in enumerate(tensors):
        for col, (label, x) in enumerate([("before", before), ("after smoothing", after)]):
            arr = x.detach().float().abs().cpu()
            vmin = 0.0
            vmax = torch.quantile(arr.flatten(), 0.99).item()
            true_max = arr.max().item()
            image = axes[row, col].imshow(arr, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{name} {label}")
            axes[row, col].set_xlabel("channel")
            axes[row, col].set_ylabel("token")
            cbar = fig.colorbar(image, ax=axes[row, col])
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
            cbar.set_label(f"|value|, max={true_max:.2f}")

    fig.savefig(plot_dir / "tensor_heatmaps.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
    for ax, (name, before, after) in zip(axes, tensors):
        ax.plot(before.mean(dim=0).detach().float().cpu().numpy(), label="before")
        ax.plot(after.mean(dim=0).detach().float().cpu().numpy(), label="after smoothing")
        ax.set_title(f"{name} channel mean")
        ax.set_xlabel("channel")
        ax.set_ylabel("mean value")
        ax.legend()
    fig.savefig(plot_dir / "channel_means.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    hist_tensors = [("Q", Q, Qg), ("K", K, Kg)]
    for row, (name, before, after) in enumerate(hist_tensors):
        for col, (label, x) in enumerate([("before", before), ("after smoothing", after)]):
            values = quantized_values(x, qk_mode).detach().float().cpu().flatten().numpy()
            axes[row, col].hist(values, bins=80)
            axes[row, col].set_title(f"{name} quantized values ({label}, {qk_mode})")
            axes[row, col].set_xlabel("quantized value")
            axes[row, col].set_ylabel("count")
    fig.savefig(plot_dir / f"quantized_value_hist_{qk_mode}.png", dpi=160)
    plt.close(fig)

    _save_qsnr_plot(Q, K, V, qk_mode, plot_dir, plt)
    _save_granularity_qsnr(Q, K, qk_mode, plot_dir, plt)

    print(f"\n{color('saved plots', title_color)}")
    print(plot_dir / "tensor_heatmaps.png")
    print(plot_dir / "channel_means.png")
    print(plot_dir / f"quantized_value_hist_{qk_mode}.png")
    print(plot_dir / f"reconstruction_qsnr_{qk_mode}.png")
    print(plot_dir / f"reconstruction_qsnr_density_{qk_mode}.png")
    print(plot_dir / f"granularity_qsnr_{qk_mode}.png")


# ---------------------------------------------------------------------------
# Ablation result plots
# ---------------------------------------------------------------------------


def save_ablation_plots(rows, plot_dir):
    """Heatmap of out_l1 / QSNR across (qk_mode, smooth)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    qk_list = sorted({r["qk_mode"] for r in rows}, key=lambda x: qk_modes.index(x))
    smooth_list = _smooth_order({r["smooth"] for r in rows})

    matrix_l1 = np.full((len(qk_list), len(smooth_list)), np.nan)
    matrix_qsnr = np.full((len(qk_list), len(smooth_list)), np.nan)
    for r in rows:
        i = qk_list.index(r["qk_mode"])
        j = smooth_list.index(r["smooth"])
        matrix_l1[i, j] = r["out_l1"]
        matrix_qsnr[i, j] = r["out_qsnr"]

    cmap_l1 = plt.get_cmap("magma_r")
    cmap_qsnr = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 3.0 + 0.45 * len(qk_list)), constrained_layout=True
    )

    vmin0, vmax0 = np.nanmin(matrix_l1), np.nanmax(matrix_l1)
    im0 = axes[0].imshow(matrix_l1, aspect="auto", cmap=cmap_l1, vmin=vmin0, vmax=vmax0)
    axes[0].set_xticks(range(len(smooth_list)))
    axes[0].set_xticklabels(smooth_list, rotation=30, ha="right")
    axes[0].set_yticks(range(len(qk_list)))
    axes[0].set_yticklabels(qk_list)
    axes[0].set_title("output rel-L1 error  (lower is better)")
    axes[0].set_xticks(np.arange(-.5, len(smooth_list), 1), minor=True)
    axes[0].set_yticks(np.arange(-.5, len(qk_list), 1), minor=True)
    axes[0].grid(which="minor", color="white", linewidth=0.5)
    axes[0].tick_params(which="minor", length=0)
    _annotate_heatmap(axes[0], matrix_l1,
                      fmt=lambda v: f"{v:.3f}",
                      cmap=cmap_l1, vmin=vmin0, vmax=vmax0)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)

    vmin1, vmax1 = np.nanmin(matrix_qsnr), np.nanmax(matrix_qsnr)
    im1 = axes[1].imshow(matrix_qsnr, aspect="auto", cmap=cmap_qsnr, vmin=vmin1, vmax=vmax1)
    axes[1].set_xticks(range(len(smooth_list)))
    axes[1].set_xticklabels(smooth_list, rotation=30, ha="right")
    axes[1].set_yticks(range(len(qk_list)))
    axes[1].set_yticklabels(qk_list)
    axes[1].set_title("output QSNR (dB)  (higher is better)")
    axes[1].set_xticks(np.arange(-.5, len(smooth_list), 1), minor=True)
    axes[1].set_yticks(np.arange(-.5, len(qk_list), 1), minor=True)
    axes[1].grid(which="minor", color="white", linewidth=0.5)
    axes[1].tick_params(which="minor", length=0)
    _annotate_heatmap(axes[1], matrix_qsnr,
                      fmt=lambda v: f"{v:.1f}",
                      cmap=cmap_qsnr, vmin=vmin1, vmax=vmax1)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)

    fig.suptitle("Forward attention: dtype × smoothing", y=1.02, fontsize=12)
    fig.savefig(plot_dir / "ablation_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(plot_dir / "ablation_heatmap.png")


def save_granularity_plots(rows, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    qk_list = sorted({r["qk_mode"] for r in rows}, key=lambda x: qk_modes.index(x))
    grans = qk_granularities
    width = 0.18
    x = np.arange(len(qk_list))

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    palette = {
        "per_tensor": "#1f77b4",
        "per_block": "#ff7f0e",
        "per_token": "#2ca02c",
        "per_thread": "#d62728",
    }

    all_vals = []
    for i, g in enumerate(grans):
        vals = []
        for qk in qk_list:
            cell = [r for r in rows if r["qk_mode"] == qk and r["granularity"] == g]
            vals.append(cell[0]["out_qsnr"] if cell else np.nan)
        all_vals.extend([v for v in vals if not np.isnan(v)])
        bars = ax.bar(x + (i - len(grans) / 2 + 0.5) * width, vals, width,
                      label=g, color=palette.get(g))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                        rotation=90 if v < 5 else 0)

    ax.set_xticks(x)
    ax.set_xticklabels(qk_list)
    ax.set_ylabel("output QSNR (dB)")
    ax.set_title("Quantization granularity sweep (Q+K mean-subtracted)")
    ax.grid(True, axis="y", alpha=0.3)
    if all_vals:
        ax.set_ylim(max(0.0, min(all_vals) - 4), max(all_vals) + 4)
    ax.legend(loc="lower right")

    fig.savefig(plot_dir / "granularity_sweep.png", dpi=160)
    plt.close(fig)
    print(plot_dir / "granularity_sweep.png")


def save_smoothquant_plots(rows, plot_dir, qk_mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    alphas = [r["alpha"] for r in rows]
    out_l1 = [r["out_l1"] for r in rows]
    qsnr = [r["out_qsnr"] for r in rows]
    best_i = int(min(range(len(alphas)), key=lambda i: out_l1[i]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(alphas, out_l1, "-o", color="#d62728")
    axes[0].axvline(alphas[best_i], color="#999", ls="--", alpha=0.6)
    axes[0].annotate(f"best α={alphas[best_i]:.2f}\nout_l1={out_l1[best_i]:.3f}",
                     xy=(alphas[best_i], out_l1[best_i]),
                     xytext=(8, 8), textcoords="offset points",
                     fontsize=9, color="#444")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("output rel-L1")
    axes[0].set_title(f"SmoothQuant α sweep ({qk_mode}) — rel-L1")
    axes[0].grid(True, alpha=0.3)

    best_qsnr_i = int(max(range(len(alphas)), key=lambda i: qsnr[i]))
    axes[1].plot(alphas, qsnr, "-o", color="#2ca02c")
    axes[1].axvline(alphas[best_qsnr_i], color="#999", ls="--", alpha=0.6)
    axes[1].annotate(f"best α={alphas[best_qsnr_i]:.2f}\nQSNR={qsnr[best_qsnr_i]:.1f} dB",
                     xy=(alphas[best_qsnr_i], qsnr[best_qsnr_i]),
                     xytext=(8, -16), textcoords="offset points",
                     fontsize=9, color="#444")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("output QSNR (dB)")
    axes[1].set_title(f"SmoothQuant α sweep ({qk_mode}) — QSNR")
    axes[1].grid(True, alpha=0.3)

    fig.savefig(plot_dir / f"smoothquant_sweep_{qk_mode}.png", dpi=160)
    plt.close(fig)
    print(plot_dir / f"smoothquant_sweep_{qk_mode}.png")


def save_backward_plots(rows, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    qk_list = sorted({r["qk_mode"] for r in rows}, key=lambda x: qk_modes.index(x))
    smooth_list = _smooth_order({r["smooth"] for r in rows})

    keys_titles = [("dQ_qsnr", "dQ QSNR"), ("dK_qsnr", "dK QSNR"), ("dV_qsnr", "dV QSNR")]

    matrices = {}
    for key, _ in keys_titles:
        m = np.full((len(qk_list), len(smooth_list)), np.nan)
        for r in rows:
            i = qk_list.index(r["qk_mode"])
            j = smooth_list.index(r["smooth"])
            m[i, j] = r[key]
        matrices[key] = m

    vmin = min(np.nanmin(m) for m in matrices.values())
    vmax = max(np.nanmax(m) for m in matrices.values())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 2.8 + 0.45 * len(qk_list)),
        sharey=True,
        constrained_layout=True,
    )
    last_im = None
    for ax, (key, title) in zip(axes, keys_titles):
        m = matrices[key]
        im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        last_im = im
        ax.set_xticks(range(len(smooth_list)))
        ax.set_xticklabels(smooth_list, rotation=30, ha="right")
        ax.set_xticks(np.arange(-.5, len(smooth_list), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(qk_list), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(f"{title} (dB)")
        _annotate_heatmap(ax, m, fmt=lambda v: f"{v:.1f}", cmap=cmap,
                          vmin=vmin, vmax=vmax)

    axes[0].set_yticks(range(len(qk_list)))
    axes[0].set_yticklabels(qk_list)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("QSNR (dB) — shared scale")

    fig.suptitle("Backward attention: dtype × smoothing  (higher is better)",
                 y=1.05, fontsize=12)
    fig.savefig(plot_dir / "backward_qsnr_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(plot_dir / "backward_qsnr_heatmap.png")


# ---------------------------------------------------------------------------
# Combined "summary" plots: forward vs backward at-a-glance
# ---------------------------------------------------------------------------


def save_summary_plot(fwd_rows, bwd_rows, plot_dir, smooth_filter="Q+K+V"):
    """Single chart: best-recipe forward+backward QSNR per dtype.

    `smooth_filter` is the smoothing setup used to compare dtypes head-to-head.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    qk_list = sorted({r["qk_mode"] for r in fwd_rows},
                     key=lambda x: qk_modes.index(x))

    def lookup(rows, qk, key):
        for r in rows:
            if r["qk_mode"] == qk and r["smooth"] == smooth_filter:
                return r[key]
        return np.nan

    fwd_qsnr = [lookup(fwd_rows, qk, "out_qsnr") for qk in qk_list]
    dq_qsnr = [lookup(bwd_rows, qk, "dQ_qsnr") for qk in qk_list]
    dk_qsnr = [lookup(bwd_rows, qk, "dK_qsnr") for qk in qk_list]
    dv_qsnr = [lookup(bwd_rows, qk, "dV_qsnr") for qk in qk_list]

    x = np.arange(len(qk_list))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
    bars = [
        ("output O", fwd_qsnr, "#1f77b4"),
        ("dQ", dq_qsnr, "#ff7f0e"),
        ("dK", dk_qsnr, "#2ca02c"),
        ("dV", dv_qsnr, "#d62728"),
    ]
    for i, (label, vals, c) in enumerate(bars):
        offset = (i - len(bars) / 2 + 0.5) * width
        bs = ax.bar(x + offset, vals, width, label=label, color=c)
        for b, v in zip(bs, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(qk_list)
    ax.set_ylabel("QSNR (dB)")
    ax.set_title(f"Forward + backward QSNR per Q/K dtype  [smooth={smooth_filter}]")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    fig.savefig(plot_dir / "summary_qsnr.png", dpi=160)
    plt.close(fig)
    print(plot_dir / "summary_qsnr.png")
