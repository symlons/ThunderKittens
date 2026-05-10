from pathlib import Path

import torch

from smooth_core import quantized_values
from smooth_report import color, title_color


def save_tensor_plots(Q, K, V, qk_mode, plot_dir):
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

    print(f"\n{color('saved plots', title_color)}")
    print(plot_dir / "tensor_heatmaps.png")
    print(plot_dir / "channel_means.png")
    print(plot_dir / f"quantized_value_hist_{qk_mode}.png")
