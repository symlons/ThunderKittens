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
    """Heatmap of out_l1 across (qk_mode, smooth) and per-dtype QSNR bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    qk_list = sorted({r["qk_mode"] for r in rows}, key=lambda x: qk_modes.index(x))
    smooth_list = sorted({r["smooth"] for r in rows})

    matrix_l1 = np.full((len(qk_list), len(smooth_list)), np.nan)
    matrix_qsnr = np.full((len(qk_list), len(smooth_list)), np.nan)
    for r in rows:
        i = qk_list.index(r["qk_mode"])
        j = smooth_list.index(r["smooth"])
        matrix_l1[i, j] = r["out_l1"]
        matrix_qsnr[i, j] = r["out_qsnr"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4 + 0.4 * len(qk_list)), constrained_layout=True)

    im0 = axes[0].imshow(matrix_l1, aspect="auto", cmap="viridis_r")
    axes[0].set_xticks(range(len(smooth_list)))
    axes[0].set_xticklabels(smooth_list, rotation=45, ha="right")
    axes[0].set_yticks(range(len(qk_list)))
    axes[0].set_yticklabels(qk_list)
    axes[0].set_title("output rel-L1 error (lower is better)")
    for i in range(len(qk_list)):
        for j in range(len(smooth_list)):
            v = matrix_l1[i, j]
            if not np.isnan(v):
                axes[0].text(j, i, f"{v:.2e}", ha="center", va="center",
                             color="white" if v > np.nanmedian(matrix_l1) else "black",
                             fontsize=7)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(matrix_qsnr, aspect="auto", cmap="viridis")
    axes[1].set_xticks(range(len(smooth_list)))
    axes[1].set_xticklabels(smooth_list, rotation=45, ha="right")
    axes[1].set_yticks(range(len(qk_list)))
    axes[1].set_yticklabels(qk_list)
    axes[1].set_title("output QSNR (dB, higher is better)")
    for i in range(len(qk_list)):
        for j in range(len(smooth_list)):
            v = matrix_qsnr[i, j]
            if not np.isnan(v):
                axes[1].text(j, i, f"{v:.1f}", ha="center", va="center",
                             color="black" if v > np.nanmedian(matrix_qsnr) else "white",
                             fontsize=7)
    fig.colorbar(im1, ax=axes[1])

    fig.savefig(plot_dir / "ablation_heatmap.png", dpi=160)
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

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, g in enumerate(grans):
        vals = []
        for qk in qk_list:
            cell = [r for r in rows if r["qk_mode"] == qk and r["granularity"] == g]
            vals.append(cell[0]["out_qsnr"] if cell else np.nan)
        ax.bar(x + (i - len(grans) / 2 + 0.5) * width, vals, width, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels(qk_list)
    ax.set_ylabel("output QSNR (dB)")
    ax.set_title("Quantization granularity sweep")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(alphas, [r["out_l1"] for r in rows], "-o", color="#d62728")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("output rel-L1")
    axes[0].set_title(f"SmoothQuant alpha sweep ({qk_mode})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, [r["out_qsnr"] for r in rows], "-o", color="#2ca02c")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("output QSNR (dB)")
    axes[1].set_title(f"SmoothQuant alpha sweep ({qk_mode})")
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
    smooth_list = sorted({r["smooth"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4 + 0.3 * len(qk_list)), constrained_layout=True)
    for ax, key, title in zip(axes, ["dQ_qsnr", "dK_qsnr", "dV_qsnr"],
                               ["dQ QSNR (dB)", "dK QSNR (dB)", "dV QSNR (dB)"]):
        matrix = np.full((len(qk_list), len(smooth_list)), np.nan)
        for r in rows:
            i = qk_list.index(r["qk_mode"])
            j = smooth_list.index(r["smooth"])
            matrix[i, j] = r[key]
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(smooth_list)))
        ax.set_xticklabels(smooth_list, rotation=45, ha="right")
        ax.set_yticks(range(len(qk_list)))
        ax.set_yticklabels(qk_list)
        ax.set_title(title)
        for i in range(len(qk_list)):
            for j in range(len(smooth_list)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="black" if v > np.nanmedian(matrix) else "white",
                            fontsize=7)
        fig.colorbar(im, ax=ax)

    fig.savefig(plot_dir / "backward_qsnr_heatmap.png", dpi=160)
    plt.close(fig)
    print(plot_dir / "backward_qsnr_heatmap.png")
