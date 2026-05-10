import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from smooth_core import (
    bq,
    build_ablation_rows,
    make_inputs,
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
    print_quick_comparison,
    print_range_table,
    print_setup,
    title_color,
)
from smooth_viz import save_tensor_plots


def compute_demo():
    Q, K, V = make_inputs()
    S_ref, P_ref, O_ref = reference_attention(Q, K, V)

    S_raw = raw_int4_scores(Q, K)
    P_raw = F.softmax(S_raw, dim=-1)
    O_raw = raw_int4_fp8_attention(Q, K, V)

    S_sa2 = sa2_scores(Q, K)
    S_sa2_exact = sa2_scores(Q, K, exact=True)
    P_sa2 = F.softmax(S_sa2, dim=-1)
    O_sa2 = sa2_attention(Q, K, V)
    O_sa2_v = sa2_attention(Q, K, V, smooth_v=True)

    rows = build_ablation_rows(Q, K, V, S_ref, P_ref, O_ref)

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "refs": (S_ref, P_ref, O_ref),
        "quick": (S_raw, S_ref, S_sa2, S_sa2_exact, P_raw, P_ref, P_sa2, O_raw, O_ref, O_sa2, O_sa2_v),
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
            sort_by = prompt_choice("sort metric", ["out_l1", "out_cos", "softmax_l1", "score_l1"], sort_by)
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
    parser.add_argument("--qk", default="all", help="Q/K dtype filter: all,int4,int8,fp8_e4m3 or comma-separated list")
    parser.add_argument("--smooth", default="all", help="Smoothing filter: all,none,Q,K,V,Q+K,Q+V,K+V,Q+K+V or comma-separated list")
    parser.add_argument("--sort-by", default="out_l1", choices=["out_l1", "out_cos", "softmax_l1", "score_l1"])
    parser.add_argument("--group-by", default="none", choices=["none", "qk_mode", "smooth"])
    parser.add_argument("--interactive", action="store_true", help="Open a prompt menu for filtering, grouping, sorting, and plotting")
    parser.add_argument("--plots", action="store_true", help="Save tensor heatmaps, channel means, and quantized-value histograms")
    parser.add_argument("--plot-qk", default="int4", choices=qk_modes, help="Q/K dtype used for quantized-value histogram")
    parser.add_argument("--plot-dir", default="smooth_plots")
    return parser.parse_args()


def main():
    args = parse_args()
    data = compute_demo()
    Q, K, V = data["Q"], data["K"], data["V"]

    print_setup(Q, K, V)
    print_quick_comparison(*data["quick"])
    print_ranges(Q, K, V)

    if args.interactive:
        interactive_menu(data, args)
    else:
        show_report(data, args.qk, args.smooth, args.sort_by, args.group_by)

    if args.plots:
        save_tensor_plots(Q, K, V, args.plot_qk, Path(args.plot_dir))


if __name__ == "__main__":
    main()
