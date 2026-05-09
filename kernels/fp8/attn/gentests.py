import torch
import numpy as np
import math
from utils import stats, print_utils, run_pairs
from references import fa2_test

B = 4
N = 3072 # token dim
D = 64 # hidd_dim/num_heads: 1024/16=64
H = 16 # headdim
causal = False
save_to_file = False

torch.random.manual_seed(42)
q = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
k = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
v = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
grad_output = (torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda'))

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

print_utils(
    q,
    k,
    v,
    o,
    q_grad,
    k_grad,
    v_grad,
    l_vec,
    d_vec
)

fa2_o, fa2_q_grad, fa2_k_grad, fa2_v_grad = fa2_test(q, k, v, grad_output, causal)
pairs = [
    ("o",  fa2_o,  o),
    # ("qg", fa2_q_grad, q_grad),
    # ("kg", fa2_k_grad, k_grad),
    # ("vg", fa2_v_grad, v_grad),
]
run_pairs(pairs)

if save_to_file:
    filename = 'randn_inputs.txt'

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
