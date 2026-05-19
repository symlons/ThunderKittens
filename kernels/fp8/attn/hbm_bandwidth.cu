// HBM bandwidth profiling kernel.
//
// Profiles single-GPU HBM bandwidth using a variety of copy/read/write
// kernels. Reports achieved GB/s along with a percentage of the theoretical
// peak (H100 SXM ~ 3.35 TB/s, H100 PCIe ~ 2.0 TB/s, B200 ~ 8 TB/s).
//
// Build:
//   nvcc -O3 -arch=sm_90a -std=c++17 hbm_bandwidth.cu -o hbm_bandwidth
// (for B200 use -arch=sm_100a)
//
// Reference layout adapted from
// ~/gpu-experiments/hopper/40-bandwidth-test.cu
//
// Usage:
//   ./hbm_bandwidth [device_id] [size_in_MB] [num_iters]

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define CUDACHECK(cmd) do {                                                     \
    cudaError_t e = (cmd);                                                      \
    if (e != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                     cudaGetErrorString(e));                                    \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

#define KB(x) ((size_t)(x) * 1024ULL)
#define MB(x) (KB(x) * 1024ULL)
#define GB(x) (MB(x) * 1024ULL)

static constexpr int    DEFAULT_DEV         = 0;
static constexpr size_t DEFAULT_SIZE_MB     = 512;     // 512 MB working set
static constexpr int    DEFAULT_NUM_ITERS   = 20;
static constexpr int    WARMUP_ITERS        = 3;

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// 1) Naive byte-wise copy.
__global__ void copy_u8(unsigned char* __restrict__ dst,
                        const unsigned char* __restrict__ src,
                        size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += stride) dst[i] = src[i];
}

// 2) 128-bit vectorized copy via uint4.
__global__ void copy_u128(uint4* __restrict__ dst,
                          const uint4* __restrict__ src,
                          size_t n_vec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n_vec; i += stride) dst[i] = src[i];
}

// 3) Vectorized copy with explicit PTX (weak ordering).
__global__ void copy_u128_ptx(uint4* __restrict__ dst,
                              const uint4* __restrict__ src,
                              size_t n_vec) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n_vec; i += stride) {
        uint4 v;
        asm volatile (
            "{ ld.weak.global.v4.u32 {%0, %1, %2, %3}, [%4];"
              "st.weak.global.v4.u32 [%5], {%0, %1, %2, %3}; }"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(src + i), "l"(dst + i)
            : "memory"
        );
    }
}

// 4) Read-only bandwidth: stream over src, accumulate, write one scalar at end.
__global__ void read_u128(const uint4* __restrict__ src,
                          size_t n_vec,
                          unsigned int* __restrict__ sink) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned int acc = 0;
    for (size_t i = idx; i < n_vec; i += stride) {
        uint4 v = src[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // Prevent the compiler from optimizing the loop away.
    if (acc == 0xDEADBEEFu) sink[idx] = acc;
}

// 5) Write-only bandwidth.
__global__ void write_u128(uint4* __restrict__ dst,
                           size_t n_vec,
                           unsigned int value) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 v = make_uint4(value, value, value, value);
    for (size_t i = idx; i < n_vec; i += stride) dst[i] = v;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

struct LaunchConfig {
    int blocks;
    int threads;
};

static LaunchConfig pick_launch(int device_id, size_t threads = 256) {
    cudaDeviceProp prop{};
    CUDACHECK(cudaGetDeviceProperties(&prop, device_id));
    // 32 blocks per SM is plenty to saturate HBM on H100/B200.
    int blocks = prop.multiProcessorCount * 32;
    return LaunchConfig{blocks, (int)threads};
}

template <typename Fn>
static double time_kernel_gbps(Fn launch, size_t bytes_moved, int num_iters) {
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERS; ++i) launch();
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; ++i) launch();
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDACHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDACHECK(cudaEventDestroy(start));
    CUDACHECK(cudaEventDestroy(stop));

    double seconds = (double)ms / 1000.0 / (double)num_iters;
    double gbps = (double)bytes_moved / seconds / (1024.0 * 1024.0 * 1024.0);
    return gbps;
}

static double peak_hbm_gbps(int device_id, const cudaDeviceProp& p) {
    // Prefer the newer cudaDeviceGetAttribute API (deprecated fields were
    // removed in CUDA 13). memory clock rate is in KHz, bus width in bits,
    // and HBM is DDR (factor 2).
    int clock_khz = 0;
    int bus_width = 0;
    cudaError_t e1 = cudaDeviceGetAttribute(
        &clock_khz, cudaDevAttrMemoryClockRate, device_id);
    cudaError_t e2 = cudaDeviceGetAttribute(
        &bus_width, cudaDevAttrGlobalMemoryBusWidth, device_id);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        (void)p;
        return 0.0;
    }
    double clock_hz = (double)clock_khz * 1.0e3;
    double bytes_per_cycle = (double)bus_width / 8.0;
    double bw_bytes_per_s = 2.0 * clock_hz * bytes_per_cycle;
    return bw_bytes_per_s / (1024.0 * 1024.0 * 1024.0);
}

static void report(const std::string& name, double gbps, double peak) {
    double pct = peak > 0.0 ? (gbps / peak) * 100.0 : 0.0;
    std::printf("  %-28s %10.2f GB/s  (%.1f%% of peak)\n",
                name.c_str(), gbps, pct);
}

int main(int argc, char** argv) {
    int device_id      = argc > 1 ? std::atoi(argv[1]) : DEFAULT_DEV;
    size_t size_mb     = argc > 2 ? (size_t)std::atoll(argv[2]) : DEFAULT_SIZE_MB;
    int num_iters      = argc > 3 ? std::atoi(argv[3]) : DEFAULT_NUM_ITERS;

    CUDACHECK(cudaSetDevice(device_id));
    cudaDeviceProp prop{};
    CUDACHECK(cudaGetDeviceProperties(&prop, device_id));

    size_t size = MB(size_mb);
    // Round to multiple of sizeof(uint4) = 16 bytes.
    size &= ~(size_t)15;
    size_t n_vec = size / sizeof(uint4);

    double peak = peak_hbm_gbps(device_id, prop);

    std::printf("Device %d: %s (SM %d.%d, %d SMs)\n",
                device_id, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);
    std::printf("HBM peak (from device props): %.2f GB/s\n", peak);
    std::printf("Working set: %.1f MB (%zu bytes)\n",
                (double)size / (1024.0 * 1024.0), size);
    std::printf("Iterations per measurement: %d (warmup %d)\n\n",
                num_iters, WARMUP_ITERS);

    // Allocate buffers.
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    unsigned int*  d_sink = nullptr;
    CUDACHECK(cudaMalloc(&d_src, size));
    CUDACHECK(cudaMalloc(&d_dst, size));
    CUDACHECK(cudaMalloc(&d_sink, sizeof(unsigned int) * 1024));
    CUDACHECK(cudaMemset(d_src, 0xA5, size));
    CUDACHECK(cudaMemset(d_dst, 0x00, size));
    CUDACHECK(cudaMemset(d_sink, 0x00, sizeof(unsigned int) * 1024));

    auto cfg = pick_launch(device_id);
    std::printf("Launch: <<<%d, %d>>>\n\n", cfg.blocks, cfg.threads);

    std::printf("== Single-GPU HBM bandwidth ==\n");

    // cudaMemcpy D2D (touches each byte twice -> 2 * size).
    {
        auto launch = [&]() {
            CUDACHECK(cudaMemcpyAsync(d_dst, d_src, size,
                                      cudaMemcpyDeviceToDevice));
        };
        double gbps = time_kernel_gbps(launch, 2 * size, num_iters);
        report("cudaMemcpy D2D", gbps, peak);
    }

    // Kernel: byte copy.
    {
        auto launch = [&]() {
            copy_u8<<<cfg.blocks, cfg.threads>>>(d_dst, d_src, size);
        };
        double gbps = time_kernel_gbps(launch, 2 * size, num_iters);
        report("kernel copy (u8)", gbps, peak);
    }

    // Kernel: 128-bit copy.
    {
        auto launch = [&]() {
            copy_u128<<<cfg.blocks, cfg.threads>>>(
                reinterpret_cast<uint4*>(d_dst),
                reinterpret_cast<const uint4*>(d_src),
                n_vec);
        };
        double gbps = time_kernel_gbps(launch, 2 * size, num_iters);
        report("kernel copy (uint4)", gbps, peak);
    }

    // Kernel: 128-bit PTX copy.
    {
        auto launch = [&]() {
            copy_u128_ptx<<<cfg.blocks, cfg.threads>>>(
                reinterpret_cast<uint4*>(d_dst),
                reinterpret_cast<const uint4*>(d_src),
                n_vec);
        };
        double gbps = time_kernel_gbps(launch, 2 * size, num_iters);
        report("kernel copy (uint4 ptx)", gbps, peak);
    }

    // Read-only.
    {
        auto launch = [&]() {
            read_u128<<<cfg.blocks, cfg.threads>>>(
                reinterpret_cast<const uint4*>(d_src), n_vec, d_sink);
        };
        double gbps = time_kernel_gbps(launch, size, num_iters);
        report("kernel read-only", gbps, peak);
    }

    // Write-only.
    {
        auto launch = [&]() {
            write_u128<<<cfg.blocks, cfg.threads>>>(
                reinterpret_cast<uint4*>(d_dst), n_vec, 0x12345678u);
        };
        double gbps = time_kernel_gbps(launch, size, num_iters);
        report("kernel write-only", gbps, peak);
    }

    // cudaMemset.
    {
        auto launch = [&]() {
            CUDACHECK(cudaMemsetAsync(d_dst, 0x00, size));
        };
        double gbps = time_kernel_gbps(launch, size, num_iters);
        report("cudaMemset", gbps, peak);
    }

    CUDACHECK(cudaFree(d_src));
    CUDACHECK(cudaFree(d_dst));
    CUDACHECK(cudaFree(d_sink));
    return 0;
}
