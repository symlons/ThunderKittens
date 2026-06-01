# Output Projection Fusion Artifacts

This directory captures the comparison between the current TK output-projection fusion boundary and the compile-visible epilogue boundary.

- `compile_fused_output_proj` routes output projection through `tk_dit::gemm_linear_gated_residual`, which fuses GEMM, projected storage, gate multiply, and residual add in one TK kernel. That kernel still spills registers.
- `compile_fused_output_proj_epilogue` leaves the projection as a normal torch linear, so `torch.compile` can use cuBLAS/NVJET for GEMM and fuse only the cheap pointwise epilogue work around it.

The diagram highlights the practical result: our fused boundary is too large for the current kernel quality; the compile-visible boundary gives the best library GEMM while still fusing pointwise work.
