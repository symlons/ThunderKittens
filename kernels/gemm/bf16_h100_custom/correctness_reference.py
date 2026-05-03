"""
PyTorch reference baselines at three precision levels.

fp32_pass       Strict fp32 math — the golden reference
fp32_rounded    fp32 math rounded to bf16 — best match for bf16 custom outputs
autocast_bf16   What torch.autocast(dtype=torch.bfloat16) actually uses.

Returns dicts of named gradient tensors.
"""
from typing import Optional
import torch
import torch.nn.functional as F


# ---- Helpers ----

def _preact_from_gelu_bwd(preact_bf16: torch.Tensor, dy_bf16: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Compute d(preact) = d(GELU) * GELU' using PyTorch autograd.
    If dtype is given, cast inputs to that dtype before differentiation.
    """
    p = preact_bf16.to(dtype) if dtype else preact_bf16.float()
    g = dy_bf16.to(dtype) if dtype else dy_bf16.float()
    p.requires_grad_(True)
    y = F.gelu(p, approximate="tanh")
    y.backward(g)
    return p.grad


def _autograd_linear_bwd(x_bf16: torch.Tensor, W_bf16: torch.Tensor, b_bf16: torch.Tensor, dy_bf16: torch.Tensor, dtype: Optional[torch.dtype] = None) -> dict[str, torch.Tensor]:
    """
    Full backward using PyTorch autograd at a chosen precision.
    Returns {dz, dW, dx, db} each in the given dtype (or fp32 if dtype is None).
    """
    cast = lambda t: t.to(dtype) if dtype else t.float()
    x = cast(x_bf16)
    W = cast(W_bf16)
    b = cast(b_bf16)
    g = cast(dy_bf16)

    preact = (x @ W + b).requires_grad_(True)
    y = F.gelu(preact, approximate="tanh")
    loss = (y * g).sum()
    loss.backward()

    dz = preact.grad
    dW = x.T @ dz
    dx = dz @ W.T
    db = dz.sum(0)
    return {"dz": dz, "dW": dW, "dx": dx, "db": db}


# ---- fp32 strict reference ----

def gelu_bwd_fp32(preact: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"dz": _preact_from_gelu_bwd(preact, dy, dtype=torch.float32)}


def linear_bwd_fp32(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    return _autograd_linear_bwd(x, W, b, dy, dtype=torch.float32)


# ---- fp32 reference rounded to bf16 (where custom output is bf16) ----

def gelu_bwd_fp32_rounded(preact: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    fp32 = gelu_bwd_fp32(preact, dy)
    return {"dz": fp32["dz"].bfloat16()}


def linear_bwd_fp32_rounded(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    fp32 = linear_bwd_fp32(x, W, b, dy)
    return {
        "dz": fp32["dz"].bfloat16(),
        "dW": fp32["dW"].bfloat16(),
        "dx": fp32["dx"].bfloat16(),
        "db": fp32["db"],  # custom db is fp32, leave as-is
    }


# ---- autocast bf16: what torch.autocast actually uses ----

def gelu_bwd_autocast_bf16(preact: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Simulates torch.autocast(dtype=torch.bfloat16) GELU backward.
    Autocast keeps matmul in TF32/fp32 but elementwise ops run in bf16.
    GELU backward is elementwise, so we differentiate in bf16 directly.
    """
    return {"dz": _preact_from_gelu_bwd(preact, dy, dtype=torch.bfloat16)}


def linear_bwd_autocast_bf16(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Simulates torch.autocast(dtype=torch.bfloat16) backward for
    y = GELU(x @ W + b).
    """
    return _autograd_linear_bwd(x, W, b, dy, dtype=torch.bfloat16)


# ---- Raw bf16: pure bf16 PyTorch (no autocast promotion) ----

def gelu_bwd_raw_bf16(preact: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Pure bf16 computation — every tensor and intermediate stays in bf16.
    This is the worst-case numerical baseline.
    """
    p = preact.bfloat16().requires_grad_(True)
    g = dy.bfloat16()
    y = F.gelu(p, approximate="tanh")
    y.backward(g)
    return {"dz": p.grad}


def linear_bwd_raw_bf16(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    Pure bf16 PyTorch autograd with bf16 x, W, b, dy.
    """
    return _autograd_linear_bwd(x, W, b, dy, dtype=torch.bfloat16)


# ---- cuBLAS bf16 with FP32 accumulation ----

def gelu_bwd_cublas_bf16(preact: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    cuBLAS bf16 GEMM accumulates in FP32 internally, but dz is elementwise-only
    so it runs in bf16 on GPU just like raw_bf16.
    """
    return gelu_bwd_raw_bf16(preact, dy)


def linear_bwd_cublas_bf16(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, dy: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    cuBLAS bf16 GEMM takes bf16 inputs but accumulates in fp32.

    Simulates this by:
    - dz from bf16 elementwise (GELU', no GEMM involved)
    - dW/dx/db from bf16 x, dz, W, b but with fp32 accumulation for the GEMM parts
    """
    # Reconstruct preactivation from inputs
    preact = x @ W + b

    # dz is elementwise GELU backward in bf16
    dz = _preact_from_gelu_bwd(preact, dy, dtype=torch.bfloat16)

    # cuBLAS bf16 GEMM: bf16 inputs, fp32 accumulator
    x_f = x.float()
    W_f = W.float()
    dz_f = dz.float()

    dW = (x_f.T @ dz_f).bfloat16()
    dx = (dz_f @ W_f.T).bfloat16()
    db = dz_f.sum(0)
    return {"dz": dz, "dW": dW, "dx": dx, "db": db}


# ---- Forward references ----

def forward_fp32(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    x_f, W_f, b_f = x.float(), W.float(), b.float()
    preact = x_f @ W_f + b_f
    y = F.gelu(preact, approximate="tanh")
    return {"preact": preact, "y": y}


def forward_autocast_bf16(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        preact = x @ W + b
        y = F.gelu(preact, approximate="tanh")
    return {"preact": preact, "y": y}


def forward_raw_bf16(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    preact = x @ W + b
    y = F.gelu(preact, approximate="tanh")
    return {"preact": preact, "y": y}
