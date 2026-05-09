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

def run_pairs(pairs):
    for label, fa2_x, ref_x in pairs:
        fa2 = stats(fa2_x)
        ref = stats(ref_x)

        diff = ref_x - fa2_x
        abs_diff = torch.abs(diff)

        sum_diff = abs_diff.sum().item()
        sum_abs  = torch.abs(ref_x).sum().item()
        max_diff = abs_diff.max().item()

        rel_l1  = sum_diff / (sum_abs + 1e-8)
        rel_max = max_diff / (torch.abs(ref_x).max().item() + 1e-8)

        c  = colors[label]
        bg = bg_colors[label]

        print(f"\n{c}{label.upper()}{reset}")

        print(f"{'stat':<8}{'fa2':>12}{'ref':>12}{'abs':>12}{'rel':>12}")
        for n, a, b in zip(names, fa2, ref):
            abs_err = abs(a - b)
            rel_err = abs_err / (abs(b) + 1e-8)
            print(f"{n:<8}{a:12.6f}{b:12.6f}{abs_err:12.6f}{rel_err:12.6e}")

        print()
        print(f"{'L1':<8}{sum_diff:12.6f}")
        print(f"{'relL1':<8}{rel_l1:12.6e}")
        print(f"{'max':<8}{max_diff:12.6f}")
        print(f"{'relMax':<8}{rel_max:12.6e}")

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
