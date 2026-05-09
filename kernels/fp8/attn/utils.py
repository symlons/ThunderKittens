import torch

def stats(x): return x.mean().item(), x.median().item(), x.max().item(), x.min().item()

names = ["mean", "median", "max", "min"]

colors = {
    "o":  "\033[1;96m",  # cyan
    "qg": "\033[1;92m",  # ght green
    "kg": "\033[1;93m",  # yellow
    "vg": "\033[1;95m",  # magenta
}

bg_colors = {
    "o":  "\033[46m",    # cyan (less bright)
    "qg": "\033[42m",    # green
    "kg": "\033[43m",    # yellow
    "vg": "\033[45m",    # mal magenta
}
reset = "\033[0m"
tk_color = "\033[94m"  # light blue

def tk_fmt(value): return f"{tk_color}{value}{reset}"

def run_pairs(pairs):
    for label, fa2_x, ref_x, tk_x in pairs:
        fa2 = stats(fa2_x)
        ref = stats(ref_x)
        tk  = stats(tk_x)

        fa2_diff = ref_x - fa2_x
        fa2_abs_diff = torch.abs(fa2_diff)

        tk_diff = ref_x - tk_x
        tk_abs_diff = torch.abs(tk_diff)

        fa2_sum_diff = fa2_abs_diff.sum().item()
        tk_sum_diff  = tk_abs_diff.sum().item()

        sum_abs = torch.abs(ref_x).sum().item()

        fa2_max_diff = fa2_abs_diff.max().item()
        tk_max_diff  = tk_abs_diff.max().item()

        fa2_rel_l1 = fa2_sum_diff / (sum_abs + 1e-8)
        tk_rel_l1  = tk_sum_diff / (sum_abs + 1e-8)

        ref_max = torch.abs(ref_x).max().item() + 1e-8

        fa2_rel_max = fa2_max_diff / ref_max
        tk_rel_max  = tk_max_diff / ref_max

        c  = colors[label]
        bg = bg_colors[label]

        print(f"\n{c}{label.upper()}{reset}")

        gap = "  "
        print(gap.join([
            f"{'stat':<8}",
            f"{'fa2':>12}",
            tk_fmt(f"{'tk':>12}"),
            f"{'ref':>12}",
            f"{'fa2_abs':>12}",
            tk_fmt(f"{'tk_abs':>12}"),
            f"{'fa2_rel':>12}",
            tk_fmt(f"{'tk_rel':>12}"),
        ]))

        for n, a, t, b in zip(names, fa2, tk, ref):
            fa2_abs_err = abs(a - b)
            tk_abs_err  = abs(t - b)

            fa2_rel_err = fa2_abs_err / (abs(b) + 1e-8)
            tk_rel_err  = tk_abs_err / (abs(b) + 1e-8)

            print(gap.join([
                f"{n:<8}",
                f"{a:12.6f}",
                tk_fmt(f"{t:12.6f}"),
                f"{b:12.6f}",
                f"{fa2_abs_err:12.6f}",
                tk_fmt(f"{tk_abs_err:12.6f}"),
                f"{fa2_rel_err:12.6e}",
                tk_fmt(f"{tk_rel_err:12.6e}"),
            ]))

        print()
        print(f"{'fa2_L1':<12}{fa2_sum_diff:12.6f}")
        print(tk_fmt(f"{'tk_L1':<12}{tk_sum_diff:12.6f}"))

        print(f"{'fa2_relL1':<12}{fa2_rel_l1:12.6e}")
        print(tk_fmt(f"{'tk_relL1':<12}{tk_rel_l1:12.6e}"))

        print(f"{'fa2_max':<12}{fa2_max_diff:12.6f}")
        print(tk_fmt(f"{'tk_max':<12}{tk_max_diff:12.6f}"))

        print(f"{'fa2_relMax':<12}{fa2_rel_max:12.6e}")
        print(tk_fmt(f"{'tk_relMax':<12}{tk_rel_max:12.6e}"))

        print(bg + " " * 1 + reset)

def print_utils(q, k, v, o, q_grad, k_grad, v_grad, l_vec, d_vec):
    print("--------------------------------------")
    print("Q shape: ",      q.shape)
    print("K shape: ",      k.shape)
    print("V shape: ",      v.shape)
    print("O shape: ",      o.shape)
    print("Q grad shape: ", q_grad.shape)
    print("K grad shape: ", k_grad.shape)
    print("V grad shape: ", v_grad.shape)
    print("L shape: ",      l_vec.shape)
    print("D shape: ",      d_vec.shape)
    print("--------------------------------------")

    print(f'Average magnitude of OUTPUT tensor: {o.abs().mean()}')
    print(f'1/100 magnitude of OUTPUT tensor:   {o.abs().mean()/100}')
    print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
    print(f'1/100 magnitude of Q_GRAD tensor:   {q_grad.abs().mean()/100}')
    print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
    print(f'1/100 magnitude of K_GRAD tensor:   {k_grad.abs().mean()/100}')
    print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
    print(f'1/100 magnitude of V_GRAD tensor:   {v_grad.abs().mean()/100}')
    print(f'Average magnitude of L tensor:      {l_vec.abs().mean()}')
    print(f'1/100 magnitude of L tensor:        {l_vec.abs().mean()/100}')
    print(f'Average magnitude of D tensor:      {d_vec.abs().mean()}')
    print(f'1/100 magnitude of D tensor:        {d_vec.abs().mean()/100}')
    print("--------------------------------------")
