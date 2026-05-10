from pathlib import Path

import torch

from smooth_core import (
    bkv,
    bq,
    cw,
    fp8_quantize_v_per_channel,
    qk_modes,
    quantize_k_per_thread,
    quantize_q_per_thread,
    quantized_values,
)
from smooth_report import color, title_color


def _dequantize(q, scale):
    return q.to(torch.float32) * scale


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


def _reconstruct_q(x, mode):
    y = torch.empty_like(x, dtype=torch.float32)
    for qs in range(0, x.shape[0], bq):
        qe = min(qs + bq, x.shape[0])
        q, scale = quantize_q_per_thread(x[qs:qe], cw, mode)
        y[qs:qe] = _dequantize(q, scale)
    return y


def _reconstruct_k(x, mode):
    y = torch.empty_like(x, dtype=torch.float32)
    for ks in range(0, x.shape[0], bkv):
        ke = min(ks + bkv, x.shape[0])
        q, scale = quantize_k_per_thread(x[ks:ke], mode)
        y[ks:ke] = _dequantize(q, scale)
    return y


def _reconstruct_v(x):
    vhat, scale = fp8_quantize_v_per_channel(x)
    return _dequantize(vhat, scale)


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
        ax.set_title(f"{tensor_name} reconstruction QSNR")
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
        ax.set_title(f"{tensor_name} reconstruction QSNR density")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("QSNR (dB)")
    fig.savefig(plot_dir / f"reconstruction_qsnr_density_{qk_mode}.png", dpi=160)
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

    print(f"\n{color('saved plots', title_color)}")
    print(plot_dir / "tensor_heatmaps.png")
    print(plot_dir / "channel_means.png")
    print(plot_dir / f"quantized_value_hist_{qk_mode}.png")
    print(plot_dir / f"reconstruction_qsnr_{qk_mode}.png")
    print(plot_dir / f"reconstruction_qsnr_density_{qk_mode}.png")
