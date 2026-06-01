from __future__ import annotations

import argparse
import statistics

import torch

from _C import (
    bf16_quantize_delayed,
    bf16_quantize_rowwise_transpose_delayed,
    bf16_quantize_transpose_delayed,
    fp8_dgrad_2xacc_scaled,
    fp8_gemm_k1024_bf16_out_wide_scaled,
    fp8_gemm_k4096_bf16_out_bias,
    fp8_wgrad_2xacc_scaled,
)


def bench(fn, warmup=5, iters=20):
    times=[]
    for i in range(warmup+iters):
        torch.cuda.synchronize(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record(); out=fn(); e.record(); torch.cuda.synchronize()
        if i>=warmup: times.append(s.elapsed_time(e))
        del out
    return statistics.mean(times), min(times)


def run(rows: int, n: int, k: int, warmup: int, iters: int):
    dev='cuda'; qs=torch.ones((1,),device=dev,dtype=torch.float32)
    x=torch.randn((rows,k),device=dev,dtype=torch.bfloat16)
    w=torch.randn((n,k),device=dev,dtype=torch.bfloat16)
    dy=torch.randn((rows,n),device=dev,dtype=torch.bfloat16)
    b=torch.randn((n,),device=dev,dtype=torch.bfloat16)
    qx,qx_t,_=bf16_quantize_rowwise_transpose_delayed(x,qs)
    qw,qw_t,_=bf16_quantize_rowwise_transpose_delayed(w,qs)
    qdy,qdy_t,_=bf16_quantize_rowwise_transpose_delayed(dy,qs)
    torch.cuda.synchronize()
    print(f"\nlinear rows={rows} n={n} k={k}")
    for name,fn in [
        ('x_quant_row_T', lambda: bf16_quantize_rowwise_transpose_delayed(x,qs)),
        ('w_quant_row_T', lambda: bf16_quantize_rowwise_transpose_delayed(w,qs)),
        ('dy_quant_row_T', lambda: bf16_quantize_rowwise_transpose_delayed(dy,qs)),
        ('x_quant_row_only', lambda: bf16_quantize_delayed(x,qs)),
        ('x_quant_T_only', lambda: bf16_quantize_transpose_delayed(x,qs)),
    ]:
        mean,mn=bench(fn,warmup,iters); print(f"{name}: mean={mean:.6f} min={mn:.6f} ms")
    if k==1024:
        fwd=lambda: fp8_gemm_k1024_bf16_out_wide_scaled(qx,qw,1.0,1.0) + b
    elif k==4096:
        fwd=lambda: fp8_gemm_k4096_bf16_out_bias(qx,qw,b)
    else:
        return
    for name,fn in [
        ('fwd_gemm', fwd),
        ('dgrad_gemm', lambda: fp8_dgrad_2xacc_scaled(qdy,qw_t,1.0,1.0)),
        ('wgrad_gemm', lambda: fp8_wgrad_2xacc_scaled(qx_t,qdy_t,1.0,1.0)),
        ('bias_reduce_torch', lambda: dy.float().sum(dim=0).to(torch.bfloat16)),
    ]:
        mean,mn=bench(fn,warmup,iters); print(f"{name}: mean={mean:.6f} min={mn:.6f} ms")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rows',type=int,default=65536); ap.add_argument('--warmup',type=int,default=5); ap.add_argument('--iters',type=int,default=10)
    args=ap.parse_args()
    for n,k in [(3072,1024),(1024,1024),(4096,1024),(1024,4096)]:
        run(args.rows,n,k,args.warmup,args.iters)

if __name__=='__main__': main()
