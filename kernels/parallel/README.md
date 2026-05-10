# ThunderKittens Multi-GPU Kernels ("ParallelKittens")

Each directory contains a single operator (e.g., fused all-gather GEMM). All kernels assume 8 GPUs; in order to change this, you must modify the `static constexpr int NUM_DEVICES` field in the kernel code.

## How to run

First export the `ARCH` environment variable.

```bash
export ARCH=SM90 # or SM100
```

Note that SM100 (Blackwell) is supported if the directory includes a `*_b200.cu` file or a file with no architecture-specific suffix. Likewise, SM90 (Hopper) is supported if the directory includes a `*_h100.cu` file or a file with no architecture-specific suffix.

```bash
cd <directory> # navigate to the desired operator directory
make run      # compiles and executes with an 8-GPU torchrun configuration
```

Or you can compile without execution by simply running `make`.
