from smooth_core import (
    bkv,
    bq,
    cos,
    cw,
    metrics,
    qk_modes,
    rel_l1,
    smooth_labels,
)


reset = "\033[0m"
title_color = "\033[1;96m"
sa2_color = "\033[94m"


def color(text, code):
    return f"{code}{text}{reset}"


def parse_list(value, choices):
    if value == "all":
        return choices
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(items) - set(choices))
    if invalid:
        raise ValueError(f"invalid choices {invalid}; valid choices are {choices}")
    return items


def sorted_rows(rows, sort_by):
    reverse = sort_by == "out_cos"
    return sorted(rows, key=lambda row: row[sort_by], reverse=reverse)


def filter_rows(rows, qk_filter, smooth_filter):
    return [
        row for row in rows
        if row["qk_mode"] in qk_filter and row["smooth"] in smooth_filter
    ]


def print_setup(Q, K, V):
    print("--------------------------------------")
    print(color("SageAttention2 smoothing demo", title_color))
    print("Q shape: ", Q.shape)
    print("K shape: ", K.shape)
    print("V shape: ", V.shape)
    print("block_q: ", bq)
    print("block_k: ", bkv)
    print("warps:   ", cw)
    print("--------------------------------------")


def print_metric_table(title, rows):
    print(f"\n{color(title, title_color)}")
    gap = "  "
    print(gap.join([
        f"{'case':<28}",
        f"{'rel_l1':>12}",
        f"{'cos':>12}",
    ]))
    for name, x, ref in rows:
        l1, cs = metrics(x, ref)
        line = gap.join([
            f"{name:<28}",
            f"{l1:12.6e}",
            f"{cs:12.6f}",
        ])
        if name.startswith("sa2"):
            line = color(line, sa2_color)
        print(line)


def print_range_table(rows):
    print(f"\n{color('ranges', title_color)}")
    gap = "  "
    print(gap.join([
        f"{'tensor':<8}",
        f"{'before':>12}",
        f"{'after':>12}",
        f"{'ratio':>12}",
    ]))
    for name, before, after in rows:
        ratio = after / max(before, 1e-12)
        print(gap.join([
            f"{name:<8}",
            f"{before:12.6f}",
            f"{after:12.6f}",
            f"{ratio:12.6f}",
        ]))


def print_ablation_table(title, rows, sort_by):
    print(f"\n{color(title, title_color)}")
    gap = "  "
    print(gap.join([
        f"{'#':>3}",
        f"{'qk':<10}",
        f"{'smooth':<8}",
        f"{'score_l1':>12}",
        f"{'softmax_l1':>12}",
        f"{'out_l1':>12}",
        f"{'out_cos':>12}",
    ]))

    for rank, row in enumerate(sorted_rows(rows, sort_by), start=1):
        line = gap.join([
            f"{rank:>2}.",
            f"{row['qk_mode']:<10}",
            f"{row['smooth']:<8}",
            f"{row['score_l1']:12.6e}",
            f"{row['softmax_l1']:12.6e}",
            f"{row['out_l1']:12.6e}",
            f"{row['out_cos']:12.6f}",
        ])
        if row["smooth"] == "Q+K+V":
            line = color(line, sa2_color)
        print(line)


def print_best_by_dtype(rows):
    print(f"\n{color('best by qk dtype', title_color)}")
    gap = "  "
    print(gap.join([
        f"{'qk':<10}",
        f"{'smooth':<8}",
        f"{'out_l1':>12}",
        f"{'out_cos':>12}",
    ]))

    best_rows = []
    for qk_mode in qk_modes:
        dtype_rows = [row for row in rows if row["qk_mode"] == qk_mode]
        if not dtype_rows:
            continue
        best = min(dtype_rows, key=lambda row: row["out_l1"])
        best_rows.append(best)
        print(gap.join([
            f"{best['qk_mode']:<10}",
            f"{best['smooth']:<8}",
            f"{best['out_l1']:12.6e}",
            f"{best['out_cos']:12.6f}",
        ]))

    if best_rows:
        overall = min(best_rows, key=lambda row: row["out_l1"])
        print(color(gap.join([
            f"{'overall':<10}",
            f"{overall['qk_mode']} {overall['smooth']:<8}",
            f"{overall['out_l1']:12.6e}",
            f"{overall['out_cos']:12.6f}",
        ]), sa2_color))


def print_grouped(rows, group_by, sort_by):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_by], []).append(row)

    for group_name in sorted(groups):
        print_ablation_table(f"smoothing ablation: {group_by}={group_name}", groups[group_name], sort_by)


def print_ablation(rows, sort_by="out_l1", group_by="none"):
    if not rows:
        print(color("\nno rows match the selected filters", title_color))
        return

    if group_by == "none":
        print_ablation_table("smoothing ablation", rows, sort_by)
    else:
        print_grouped(rows, group_by, sort_by)

    print_best_by_dtype(rows)


def filtered_rows(rows, qk_value, smooth_value):
    qk_filter = parse_list(qk_value, qk_modes)
    smooth_filter = parse_list(smooth_value, smooth_labels)
    return filter_rows(rows, qk_filter, smooth_filter)


def print_quick_comparison(S_raw, S_ref, S_sa2, S_sa2_exact, P_raw, P_ref, P_sa2, O_raw, O_ref, O_sa2, O_sa2_v):
    from smooth_core import row_center, row_max_center

    print_metric_table("scores", [
        ("raw int4", S_raw, S_ref),
        ("sa2 exact score", S_sa2_exact, S_ref),
    ])
    print_metric_table("scores, row centered", [
        ("raw int4", row_center(S_raw), row_center(S_ref)),
        ("sa2 smoothed", row_center(S_sa2), row_center(S_ref)),
    ])
    print_metric_table("scores, row max centered", [
        ("raw int4", row_max_center(S_raw), row_max_center(S_ref)),
        ("sa2 smoothed", row_max_center(S_sa2), row_max_center(S_ref)),
    ])
    print_metric_table("softmax", [
        ("raw int4", P_raw, P_ref),
        ("sa2 smoothed", P_sa2, P_ref),
    ])
    print_metric_table("output", [
        ("raw int4 qk + fp8 pv", O_raw, O_ref),
        ("sa2 int4 qk + fp8 pv", O_sa2, O_ref),
        ("sa2 + smooth V", O_sa2_v, O_ref),
    ])
