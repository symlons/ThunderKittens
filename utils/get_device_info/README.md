# get_device_info

Two utilities that query CUDA device capabilities and print them with color-coded output.

## Files

| Target | What it prints |
|---|---|
| `device_info` | Name, CC, SM count, memory, L2 cache, clocks, shared memory, registers, bandwidth, peak TFLOPS |
| `async_engines` | Async engine count, H2D/D2H/compute overlap, memory pools, managed memory, cooperative launch, IPC, ECC |

## Build & Run

```bash
make -C utils/get_device_info run
```

Or build targets individually:

```bash
make -C utils/get_device_info device_info   # build only
make -C utils/get_device_info async_engines # build only
```

## Select GPU Architecture

```bash
make -C utils/get_device_info GPU=H100 run
make -C utils/get_device_info GPU=A100 run
```

Supported: `H100`, `B200`, `B300`, `A100`.

## Query L2 Cache in Your Own Code

```cpp
int l2_cache_size;
cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
// or
cudaDeviceProp props;
cudaGetDeviceProperties(&props, 0);
int l2 = props.l2CacheSize;
```
