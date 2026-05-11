import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from smooth_core import (
    bq,
    build_ablation_rows,
    build_backward_rows,
    build_granularity_rows,
    build_smoothquant_rows,
    make_inputs,
    qk_granularities,
    qk_modes,
    raw_int4_fp8_attention,
    raw_int4_scores,
    reference_attention,
    sa2_attention,
    sa2_scores,
    smooth_labels,
)
from smooth_report import (
    color,
    filtered_rows,
    print_ablation,
    print_backward_table,
    print_granularity_table,
    print_quick_comparison,
    print_range_table,
    print_setup,
    print_smoothquant_table,
    title_color,
)
from smooth_viz import (
    save_ablation_plots,
    save_backward_plots,
    save_granularity_plots,
    save_smoothquant_plots,
    save_summary_plot,
    save_tensor_plots,
)


def compute_demo(args):
    Q, K, V = make_inputs(source=args.source, path=args.source_path,
                          head_index=args.head_index, n=args.n)
    S_ref, P_ref, O_ref = reference_attention(Q, K, V)

    S_raw = raw_int4_scores(Q, K)
    P_raw = F.softmax(S_raw, dim=-1)
    O_raw = raw_int4_fp8_attention(Q, K, V)

    S_sa2 = sa2_scores(Q, K)
    S_sa2_exact = sa2_scores(Q, K, exact=True)
    P_sa2 = F.softmax(S_sa2, dim=-1)
    O_sa2 = sa2_attention(Q, K, V)
    O_sa2_v = sa2_attention(Q, K, V, smooth_v=True)

    rows = build_ablation_rows(Q, K, V, S_ref, P_ref, O_ref, granularity=args.granularity)

    return {
        "Q": Q, "K": K, "V": V,
        "refs": (S_ref, P_ref, O_ref),
        "quick": (S_raw, S_ref, S_sa2, S_sa2_exact, P_raw, P_ref, P_sa2,
                  O_raw, O_ref, O_sa2, O_sa2_v),
        "rows": rows,
    }


def print_ranges(Q, K, V):
    Kg = K - K.mean(dim=0, keepdim=True)
    Vg = V - V.mean(dim=0, keepdim=True)

    Qg = torch.empty_like(Q)
    for qs in range(0, Q.shape[0], bq):
        qe = min(qs + bq, Q.shape[0])
        Qi = Q[qs:qe]
        Qg[qs:qe] = Qi - Qi.mean(dim=0, keepdim=True)

    print_range_table([
        ("Q", Q.abs().max().item(), Qg.abs().max().item()),
        ("K", K.abs().max().item(), Kg.abs().max().item()),
        ("V", V.abs().max().item(), Vg.abs().max().item()),
    ])


def show_report(data, qk_value, smooth_value, sort_by, group_by):
    rows = filtered_rows(data["rows"], qk_value, smooth_value)
    print_ablation(rows, sort_by=sort_by, group_by=group_by)


def prompt_choice(prompt, choices, current):
    print(f"\n{color(prompt, title_color)}")
    print(f"current: {current}")
    for i, choice in enumerate(choices, start=1):
        print(f"{i}. {choice}")
    value = input("> ").strip()
    if not value:
        return current
    if value.isdigit() and 1 <= int(value) <= len(choices):
        return choices[int(value) - 1]
    return value


def interactive_menu(data, args):
    qk_value = args.qk
    smooth_value = args.smooth
    sort_by = args.sort_by
    group_by = args.group_by

    while True:
        print(f"\n{color('interactive menu', title_color)}")
        print(f"qk={qk_value}  smooth={smooth_value}  sort_by={sort_by}  group_by={group_by}")
        print("1. show current table")
        print("2. set qk filter")
        print("3. set smoothing filter")
        print("4. set sort metric")
        print("5. set grouping")
        print("6. save tensor plots")
        print("7. show quick comparison")
        print("8. quit")

        choice = input("> ").strip()
        if choice == "1":
            show_report(data, qk_value, smooth_value, sort_by, group_by)
        elif choice == "2":
            qk_value = prompt_choice("qk filter", ["all"] + qk_modes, qk_value)
        elif choice == "3":
            smooth_value = prompt_choice("smoothing filter", ["all"] + smooth_labels, smooth_value)
        elif choice == "4":
            sort_by = prompt_choice("sort metric",
                ["out_l1", "out_cos", "out_qsnr", "softmax_l1", "score_l1"], sort_by)
        elif choice == "5":
            group_by = prompt_choice("grouping", ["none", "qk_mode", "smooth"], group_by)
        elif choice == "6":
            plot_qk = prompt_choice("plot qk dtype", qk_modes, args.plot_qk)
            plot_dir = input(f"plot dir [{args.plot_dir}]> ").strip() or args.plot_dir
            save_tensor_plots(data["Q"], data["K"], data["V"], plot_qk, Path(plot_dir))
        elif choice == "7":
            print_quick_comparison(*data["quick"])
            print_ranges(data["Q"], data["K"], data["V"])
        elif choice == "8" or choice.lower() in {"q", "quit", "exit"}:
            return
        else:
            print("unknown choice")


def parse_args():
    parser = argparse.ArgumentParser(description="SageAttention2 smoothing and quantization simulation")
    parser.add_argument("--qk", default="all", help="Q/K dtype filter: all,int4,int8,fp8_e4m3,fp8_e5m2 or comma list")
    parser.add_argument("--smooth", default="all", help="Smoothing filter: all,none,Q,K,V,Q+K,... or comma list")
    parser.add_argument("--granularity", default="per_thread", choices=qk_granularities,
                        help="Q/K quantization granularity for the main ablation")
    parser.add_argument("--sort-by", default="out_l1",
                        choices=["out_l1", "out_cos", "out_qsnr", "softmax_l1", "score_l1"])
    parser.add_argument("--group-by", default="none", choices=["none", "qk_mode", "smooth", "granularity"])
    parser.add_argument("--source", default="synthetic", choices=["synthetic", "cogvideox"],
                        help="Where Q,K,V come from")
    parser.add_argument("--source-path", default=None,
                        help="Path to the captured Q,K,V tensors when --source=cogvideox")
    parser.add_argument("--head-index", type=int, default=0,
                        help="Which attention head from the captured tensor to use")
    parser.add_argument("--n", type=int, default=256,
                        help="Truncate token count for synthetic / loaded tensors")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--plots", action="store_true",
                        help="Save tensor heatmaps, channel means, quantized hist, QSNR")
    parser.add_argument("--ablation-plots", action="store_true",
                        help="Save the dtype x smoothing heatmap")
    parser.add_argument("--granularity-sweep", action="store_true",
                        help="Run the per-tensor/per-block/per-token/per-thread sweep")
    parser.add_argument("--smoothquant-sweep", action="store_true",
                        help="Run the SmoothQuant alpha sweep")
    parser.add_argument("--smoothquant-qk", default="int4", choices=qk_modes,
                        help="Q/K dtype for the SmoothQuant alpha sweep")
    parser.add_argument("--smoothquant-alphas", default=None,
                        help="Comma list of alphas (default: 0,0.1,...,1.0)")
    parser.add_argument("--bwd", action="store_true",
                        help="Run the backward pass ablation")
    parser.add_argument("--plot-qk", default="int4", choices=qk_modes)
    parser.add_argument("--plot-dir", default="smooth_plots")
    parser.add_argument("--summary-smooth", default="Q+K+V",
                        help="Smoothing combo used for the summary fwd+bwd QSNR plot")
    return parser.parse_args()


def main():
    args = parse_args()
    data = compute_demo(args)
    Q, K, V = data["Q"], data["K"], data["V"]

    print_setup(Q, K, V, source=args.source)
    print_quick_comparison(*data["quick"])
    print_ranges(Q, K, V)

    if args.interactive:
        interactive_menu(data, args)
    else:
        show_report(data, args.qk, args.smooth, args.sort_by, args.group_by)

    plot_dir = Path(args.plot_dir)

    if args.granularity_sweep:
        rows = build_granularity_rows(Q, K, V, *data["refs"])
        print_granularity_table(rows)
        save_granularity_plots(rows, plot_dir)

    if args.smoothquant_sweep:
        alphas = None
        if args.smoothquant_alphas:
            alphas = [float(a) for a in args.smoothquant_alphas.split(",")]
        rows = build_smoothquant_rows(Q, K, V, *data["refs"],
                                      qk_mode=args.smoothquant_qk,
                                      granularity=args.granularity,
                                      alphas=alphas)
        print_smoothquant_table(rows)
        save_smoothquant_plots(rows, plot_dir, args.smoothquant_qk)

    bwd_rows = None
    if args.bwd:
        torch.manual_seed(1)
        dO = torch.randn_like(Q)
        bwd_rows = build_backward_rows(Q, K, V, dO, granularity=args.granularity)
        print_backward_table(bwd_rows)
        save_backward_plots(bwd_rows, plot_dir)

    if args.ablation_plots:
        save_ablation_plots(data["rows"], plot_dir)
        if bwd_rows is not None:
            save_summary_plot(data["rows"], bwd_rows, plot_dir,
                              smooth_filter=args.summary_smooth)

    if args.plots:
        save_tensor_plots(Q, K, V, args.plot_qk, plot_dir)


if __name__ == "__main__":
    main()
