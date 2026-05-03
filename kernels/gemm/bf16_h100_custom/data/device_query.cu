// device_query.cu - query GPU hardware specs at runtime
//
// Usage:
//   nvcc -O2 -O2 -O2 -O2 -O2 -arch=sm_90a device_query.cu -o device_query && ./device_query
//
// Prints newline-delimited key=value pairs to stdout (machine-parseable).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static const char *attr_err(cudaError_t e) {
    fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(e));
    return "";
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) attr_err(e);

    // clockRate and memoryClockRate deprecated in CUDA 11.2+, removed from struct in CUDA 13.
    int sm_clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);

    int max_sm_clock_khz = sm_clock_khz;

    // Theoretical peak HBM bandwidth
    // NOTE: mem_clock_khz is the base clock, NOT effective DDR rate.
    // HBM3e has 10 transfers/cycle effective, so multiply by 10 for effective rate.
    // Formula: (bus_width_bits / 8) * effective_clock_Hz * 1 / 1e12
    double hbm_bw_tbps = ((double)prop.memoryBusWidth / 8.0)
                        * (double)mem_clock_khz * 1e3
                        * 2.0 / 1e12;

    // ===== Compute peaks (Hopper / H200 architecture) =====
    // Per-SM: 144 FP32 CUDA cores + 4 FP64 CUDA cores
    // Per-SM: 4 HMA tensor units + 2 DP4A tensor units
    //
    // FP32:  CUDA cores = 144 FLOPs/cycle
    //        HMAs     = 4 * 256 = 1024 FLOPs/cycle  (mma.1688.f32.f32)
    //        Total    = 1168 FLOPs/cycle/SM
    //
    // FP64:  CUDA cores only  = 4 * 2 = 8 FLOPs/cycle/SM
    //
    // TF32:  HMAs only  = 4 * 256 = 1024 FLOPs/cycle/SM  (mma.1688.tf32.tf32)
    //
    // BF16 dense:  HMAs = 4 * 512 = 2048 FLOPs/cycle/SM  (2x mma.1688.bf16.bf16 per cycle)
    //
    // BF16 sparse:  2x dense via sparsity

    double peak_fp32_tflops = ((double)prop.multiProcessorCount
                             * (double)sm_clock_khz * 1e3
                             * (144.0 + 4.0 * 256.0)) / 1e12;  // CUDA cores + HMAs

    double peak_fp64_tflops = ((double)prop.multiProcessorCount
                             * (double)sm_clock_khz * 1e3
                             * (4.0 * 2.0)) / 1e12;  // 4 FP64 cores, 2 FLOPs each

    double peak_tf32_tflops = ((double)prop.multiProcessorCount
                              * (double)sm_clock_khz * 1e3
                              * (4.0 * 512.0)) / 1e12;  // HMA TF32

    double peak_bf16_tflops = ((double)prop.multiProcessorCount
                              * (double)sm_clock_khz * 1e3
                              * (4.0 * 896.0)) / 1e12;  // HMA BF16 ~911 TFLOPs on H200@1980MHz

    double peak_bf16_sparse_tflops = peak_bf16_tflops * 2.0;

    printf("name=%s\n", prop.name);
    printf("device_id=%d\n", dev);
    printf("compute_capability_major=%d\n", prop.major);
    printf("compute_capability_minor=%d\n", prop.minor);
    printf("multiprocessor_count=%d\n", prop.multiProcessorCount);
    printf("memory_total_bytes=%zu\n", prop.totalGlobalMem);
    printf("l2_cache_bytes=%zu\n", prop.l2CacheSize);
    printf("memory_bus_width_bits=%d\n", prop.memoryBusWidth);
    printf("clocks_khz=%d\n", sm_clock_khz);
    printf("clocks_throttle_max_khz=%d\n", max_sm_clock_khz);
    printf("memory_clock_khz=%d\n", mem_clock_khz);
    printf("hbm_bandwidth_tbps=%.2f\n", hbm_bw_tbps);
    printf("peak_fp32_tflops=%.1f\n", peak_fp32_tflops);
    printf("peak_fp64_tflops=%.1f\n", peak_fp64_tflops);
    printf("peak_tf32_tflops=%.1f\n", peak_tf32_tflops);
    printf("peak_bf16_tflops=%.1f\n", peak_bf16_tflops);
    printf("peak_bf16_sparse_tflops=%.1f\n", peak_bf16_sparse_tflops);
    printf("warp_size=%d\n", prop.warpSize);
    printf("max_threads_per_block=%d\n", prop.maxThreadsPerBlock);
    printf("max_shared_memory_per_block=%zu\n", prop.sharedMemPerBlock);
    printf("regs_per_block=%d\n", prop.regsPerBlock);
    printf("shared_mem_per_multiprocessor=%zu\n", prop.sharedMemPerMultiprocessor);
    printf("max_threads_per_multiprocessor=%d\n", prop.maxThreadsPerMultiProcessor);
    printf("max_registers_per_multiprocessor=%d\n", prop.regsPerMultiprocessor);

    int rv = 0;
    cudaRuntimeGetVersion(&rv);
    printf("cuda_version_runtime=%d\n", rv);

    return 0;
}
