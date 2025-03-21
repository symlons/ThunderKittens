print('before imports')

import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 16
H = 16
# N = 2048 if len(sys.argv) <= 2 else int(sys.argv[2])
N = 256
# D = 128 if len(sys.argv) <= 3 else int(sys.argv[3])
D = 128

softmax_scale = 1 / math.sqrt(D)
print("Starting")

TESTNAME = sys.argv[1]
device = 'mps'

if TESTNAME == 'ones':
    q           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
    k           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
    v           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
    k           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
    v           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
    grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device=device)
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    q = q.to(dtype=torch.bfloat16, device=device)
    
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = k.to(dtype=torch.bfloat16, device=device)
    
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = v.to(dtype=torch.bfloat16, device=device)
    
    grad_output = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    grad_output = grad_output.to(dtype=torch.bfloat16, device=device)
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
    v = (torch.arange(D, dtype=torch.bfloat16, device=device)/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
    
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device=device)
else:
    print('Invalid test name')
    sys.exit(0)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

fn = f'{TESTNAME}_{N}_{D}.txt'
with open(fn, 'w') as f:
    # inputs
    qf = q.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy() #
    kf = k.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    of = o.transpose(1,2).to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(B*H*N*D):
        f.write(repr(qf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(kf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(vf[i]))
        f.write(' ')
    for i in trange(B*H*N*D):
        f.write(repr(of[i]))
        f.write(' ')
