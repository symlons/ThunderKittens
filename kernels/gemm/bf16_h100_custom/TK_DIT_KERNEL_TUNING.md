# TK DiT Kernel Tuning Notes

Target case:

- GPU: NVIDIA H100 PCIe
- PyTorch: 2.11.0+cu130
- Model/block shape: `B=64`, `T=1024`, `D=1024`
- Flattened GEMM `M = B * T = 65536`

## What Changed

### Full-grid launch for the target shape

The TK LCF GEMM templates were using persistent grids (`128` or `132` CTAs) for
large projection shapes. For `M=65536`, that underutilized the scheduler for the
DiT projection GEMMs. The templates now use the full tile grid when `M == 65536`
and keep the older persistent policy for other shapes.

This affects:

- native GELU GEMM
- native linear GEMM
- gated linear residual GEMM
- out-only gated linear residual GEMM
- modulated / AdaLN linear GEMM templates

### Forward-only GELU output kernel

Added `gemm_custom_native_out`, a forward-only GEMM+bias+GELU kernel that stores
only the post-GELU output. The training path still uses the preactivation-saving
kernel where backward needs `preact`.

### Wide AdaLN projection routing

Wide projections such as attention QKV (`D -> 3D`) no longer use the inline
AdaLN+GEMM path. That path repeated the AdaLN transform for each output tile.
Instead, the code materializes AdaLN once and runs the tuned TK native linear.

### Forward-only gated residual output

Forward-only calls now use the out-only fused linear+gated-residual kernel, so
they avoid writing the intermediate projected tensor when autograd is disabled.

## Measured Impact

Isolated projection kernels at `B=64`, `T=1024`, `D=1024`:

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| QKV TK native linear | 3232 us | 2127 us | 1.52x |
| FC1 TK native linear | 4455 us | 2860 us | 1.56x |
| FC2 TK native linear | 3905 us | 2647 us | 1.48x |

After the launch-policy change, isolated fused epilogue cases are also
competitive with compiled torch:

| Case | torch.compile | TK | Speedup |
| --- | ---: | ---: | ---: |
| FC1 GEMM+GELU | 3858 us | 3416 us | 1.13x |
| FC2 linear+gated residual | 2943 us | 2780 us | 1.06x |

DiTBlock timing at `B=64`, `T=1024`, `D=1024`:

| Mode | Eager | torch.compile | Custom TK | TK vs eager | TK vs compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward | 16456 us | 12373 us | 12702 us | 1.30x | 0.97x |
| Forward+backward | 53572 us | 42490 us | 48003 us | 1.12x | 0.89x |

Full DiT-L train-step timing at `B=64`, `T=1024`:

| Variant | Time | Speedup vs eager |
| --- | ---: | ---: |
| Eager | 1.393 s | 1.00x |
| torch.compile | 1.010 s | 1.38x |
| Custom TK | 1.124 s | 1.24x |

## Current Gap

The target forward path is now close to compile, but forward+backward is still
behind. The remaining gap is mostly in backward and full-model overhead:

- custom projection GEMM backward still relies on slower TK weight-gradient /
  input-gradient paths in some cases
- gated linear residual templates still show small register spills
- full DiT includes Python/custom-op boundaries and attention/backward work that
  are not solved by the forward projection launch-policy fix

## Next Work

1. Tune backward GEMMs for `M=65536`, especially FC1/FC2 weight-gradient and
   input-gradient paths.
2. Reduce spills in `gated_linear_template` and `gated_linear_out_template`.
3. Add a dedicated fused QKV+AdaLN kernel that computes AdaLN once per input tile
   and reuses it across Q/K/V output tiles without repeating normalization.
4. Add a benchmark guard for `B=64,T=1024,D=1024` so regressions in launch policy
   and fused epilogues are caught automatically.
5. Revalidate larger token counts before applying the full-grid launch policy
   beyond `M=65536`.
