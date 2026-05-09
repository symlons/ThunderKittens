import torch
import numpy as np
import math

# Fixed configuration matching harness.impl
B = 4
N = 3072 # token dim
D = 128 # hidden dim
H = 16 # headdim
causal = False
save_to_file = False


SEED = 2024
def splitmix_bf16(count: int, seed: int, min_val: float, max_val: float) -> torch.Tensor:
    """Splitmix64 hash -> uniform bf16, bitwise identical to common.cuh fill_kernel."""
    idx = np.arange(count, dtype=np.uint64)
    x = np.uint64(seed) + idx
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    u = (x >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / 16777216.0)
    vals = u * np.float32(max_val - min_val) + np.float32(min_val)
    return torch.from_numpy(vals).to(torch.bfloat16).cuda()

def create_inputs(B, S, H, Dqk, Dv, seed):
    """Create a single set of (Q, K, V, O, LSE) tensors."""
    count_qk = B * S * H * Dqk
    count_v = B * S * H * Dv
    Q = splitmix_bf16(count_qk, seed, -1.0, 1.0).view(B, S, H, Dqk)
    K = splitmix_bf16(count_qk, seed + 1, -1.0, 1.0).view(B, S, H, Dqk)
    V = splitmix_bf16(count_v, seed + 2, -1.0, 1.0).view(B, S, H, Dv)
    O = torch.zeros(B, S, H, Dv, dtype=torch.bfloat16, device="cuda")
    LSE = torch.zeros(B, H, 1, S, dtype=torch.float32, device="cuda")
    return Q, K, V, O, LSE

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
grad_output = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))
# q, k, v, o, lse = create_inputs(B, S, H, Dqk, Dv, SEED)

# B, N, H, D
scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(D)
print(scores.shape)

scores = torch.nn.functional.softmax(scores, dim=-1).type_as(q) 
o = torch.matmul(scores, v)  # (bs, n_local_heads, seqlen, head_dim)
o.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad

softmax_scale = 1 / math.sqrt(D)
l_vec = torch.empty((B, H, N, N), dtype=torch.bfloat16, device=q.device)
for i in range(H):
    l_vec[:, i] = torch.einsum("bnd,bmd->bnm", q[:, i], k[:, i]) * softmax_scale

max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec_sum = l_vec.sum(dim=-1, keepdim=True) # changes shape to (B,H,N,1)
print("l_vec_sum.shape: ", l_vec_sum.shape)
l_vec = max_vec + torch.log(l_vec_sum)

d_vec = torch.mul(o.to(torch.bfloat16), grad_output.to(torch.bfloat16))
d_vec = d_vec.to(torch.bfloat16).sum(dim=-1, keepdim=True)


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

def fa2_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    return output, Q.grad, K.grad, V.grad

fa2_o, fa2_q_grad, fa2_k_grad, fa2_v_grad = fa2_test(q, k, v, grad_output, causal)

def stats(x): return x.mean().item(), x.median().item(), x.max().item(), x.min().item()

names = ["mean", "median", "max", "min"]
pairs = [
    ("o",  fa2_o,  o),
    ("qg", fa2_q_grad, q_grad),
    ("kg", fa2_k_grad, k_grad),
    ("vg", fa2_v_grad, v_grad),
]

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

if save_to_file:
    filename = 'randn_inputs.txt'

    # Convert tensors to numpy arrays
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    og_f = grad_output.to(torch.float32).flatten().detach().cpu().numpy()
    l_vecf = l_vec.to(torch.float32).flatten().detach().cpu().numpy()
    d_vecf = d_vec.to(torch.float32).flatten().detach().cpu().numpy()
    qg_f = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    kg_f = k_grad.to(torch.float32).flatten().detach().cpu().numpy()
    vg_f = v_grad.to(torch.float32).flatten().detach().cpu().numpy()

    with open(filename, 'wb') as f:
        for name, arr in [('Q', qf), ('K', kf), ('V', vf), ('O', of),
                          ('L', l_vecf), ('D', d_vecf), ('grad_output', og_f),
                          ('q_grad', qg_f), ('k_grad', kg_f), ('v_grad', vg_f)]:
            print(f"Writing {name}...")
            np.savetxt(f, arr.reshape(1, -1), fmt='%.8g', delimiter=' ', newline=' ')
