#include "mma.cuh"

#ifdef TEST_GROUP_MMA_TENSOR_MMA

namespace {

template<typename T>
using accum_t = std::conditional_t<std::is_same_v<T, kittens::int8> || std::is_same_v<T, kittens::uint8>, int, float>;

template<typename T, bool TA, bool TB>
static void host_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<accum_t<T>> &o, bool acc) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            accum_t<T> sum = 0;
            for(int k = 0; k < K; k++) {
                const int a_idx = TA ? k*M + m : m*K + k;
                const int b_idx = TB ? n*K + k : k*N + n;
                sum += accum_t<T>(float(a[a_idx])) * accum_t<T>(float(b[b_idx]));
            }
            o[m*N+n] = acc ? 2*sum : sum;
        }
    }
}

template<typename T>
static void fill_input(std::vector<T> &v) {
    for(int i = 0; i < v.size(); i++) {
        if constexpr (std::is_same_v<T, kittens::int8>) {
            v[i] = T((i % 5) - 2);
        }
        else {
            v[i] = T((i % 5) + 1);
        }
    }
}

template<bool TA, bool TB, bool ACC, kittens::ducks::tt::all D, typename A, typename B>
__device__ static inline void run_mma(D &d, const A &a, const B &b, kittens::semaphore &sem) {
    if constexpr (TA && TB) {
        if constexpr (ACC) kittens::group<4>::mma_AtBt(d, a, b, sem);
        else               kittens::group<4>::mm_AtBt (d, a, b, sem);
    }
    else if constexpr (TA) {
        if constexpr (ACC) kittens::group<4>::mma_AtB(d, a, b, sem);
        else               kittens::group<4>::mm_AtB (d, a, b, sem);
    }
    else if constexpr (TB) {
        if constexpr (ACC) kittens::group<4>::mma_ABt(d, a, b, sem);
        else               kittens::group<4>::mm_ABt (d, a, b, sem);
    }
    else {
        if constexpr (ACC) kittens::group<4>::mma_AB(d, a, b, sem);
        else               kittens::group<4>::mm_AB (d, a, b, sem);
    }
}

template<typename T, bool TS, bool TA, bool TB, bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__global__ void tcgen05_wrapper(const __grid_constant__ GL_A a_gl, const __grid_constant__ GL_B b_gl, const __grid_constant__ GL_O o_gl) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    using G = kittens::group<4>;
    using O = accum_t<T>;
    using D_TT = kittens::tt<O, M, N>;
    using D_RT = kittens::rt<O, M/G::GROUP_WARPS, N>;
    using B_ST = kittens::st<T, TB ? N : K, TB ? K : N>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    B_ST (&b_smem) = al.allocate<B_ST>();

    kittens::tensor_allocator<1, 1> tm_alloc{};
    D_TT d_tt;
    if constexpr (kittens::ducks::tt::full<D_TT>) {
        d_tt = tm_alloc.template allocate<D_TT>(0);
    }
    else {
        d_tt = tm_alloc.template allocate<D_TT>(0, 0);
    }

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    G::load(b_smem, b_gl, {});
    __syncthreads();
    if constexpr (TS) {
        static_assert(!TA, "TMEM A cannot be transposed.");
        using A_TT = kittens::tt<T, M, K>;
        using A_RT = kittens::rt<T, M/G::GROUP_WARPS, K>;
        A_TT a_tt;
        if constexpr (kittens::ducks::tt::full<A_TT>) {
            a_tt = tm_alloc.template allocate<A_TT>(128);
        }
        else {
            a_tt = tm_alloc.template allocate<A_TT>(0, 128);
        }
        A_RT a_reg;
        if constexpr (std::is_same_v<T, kittens::fp8e4m3> || std::is_same_v<T, kittens::fp8e5m2>) {
            using A_ST = kittens::st<T, M, K>;
            A_ST (&a_smem) = al.allocate<A_ST>();
            G::load(a_smem, a_gl, {});
            __syncthreads();
            G::load(a_reg, a_smem);
        }
        else {
            G::load(a_reg, a_gl, {});
        }
        G::store_async(a_tt, a_reg);
        kittens::tensor_store_wait();
        __syncthreads();
        if constexpr (ACC) {
            run_mma<TA, TB, false>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 0);
            run_mma<TA, TB, true>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 1);
        }
        else {
            run_mma<TA, TB, false>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 0);
        }
    }
    else {
        using A_ST = kittens::st<T, TA ? K : M, TA ? M : K>;
        A_ST (&a_smem) = al.allocate<A_ST>();
        G::load(a_smem, a_gl, {});
        __syncthreads();
        if constexpr (ACC) {
            run_mma<TA, TB, false>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 0);
            run_mma<TA, TB, true>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 1);
        }
        else {
            run_mma<TA, TB, false>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 0);
        }
    }

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, {});
}

template<typename T, bool TS, bool TA, bool TB, bool ACC>
static void run_one(test_data &results, const std::string &label) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    using O = accum_t<T>;
    constexpr int A_ROWS = TA ? K : M;
    constexpr int A_COLS = TA ? M : K;
    constexpr int B_ROWS = TB ? N : K;
    constexpr int B_COLS = TB ? K : N;

    test_info this_result;
    this_result.label = label;
    if constexpr ((TS && TA) || (TS && sizeof(T) == 1 && !std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2>)) {
        this_result.result = test_result::INVALID;
        results.push_back(this_result);
        return;
    }

    std::vector<T> h_a(A_ROWS*A_COLS);
    std::vector<T> h_b(B_ROWS*B_COLS);
    std::vector<O> h_o(M*N, 0);
    std::vector<O> h_ref(M*N, 0);
    fill_input(h_a);
    fill_input(h_b);
    host_ref<T, TA, TB>(h_a, h_b, h_ref, ACC);

    T *d_a, *d_b;
    O *d_o;
    cudaMalloc(&d_a, h_a.size() * sizeof(T));
    cudaMalloc(&d_b, h_b.size() * sizeof(T));
    cudaMalloc(&d_o, h_o.size() * sizeof(O));
    CudaCheckError();
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, h_o.size() * sizeof(O));
    CudaCheckError();

    using GL_A = kittens::gl<T, 1, 1, A_ROWS, A_COLS>;
    using GL_B = kittens::gl<T, 1, 1, B_ROWS, B_COLS>;
    using GL_O = kittens::gl<O, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_wrapper<T, TS, TA, TB, ACC, GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY-1024
    );
    tcgen05_wrapper<T, TS, TA, TB, ACC, GL_A, GL_B, GL_O><<<1, kittens::group<4>::GROUP_THREADS, kittens::MAX_SHARED_MEMORY-1024>>>(a_gl, b_gl, o_gl);
    CudaCheckError();
    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(O), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for(int i = 0; i < h_o.size(); i++) {
        if constexpr (std::is_same_v<O, int>) {
            if(h_o[i] != h_ref[i]) {
                good = false;
                break;
            }
        }
        else {
            if(std::abs(float(h_o[i] - h_ref[i])) > 1e-3f) {
                good = false;
                break;
            }
        }
    }
    std::cout << "test `" << label << "`";
    if(good) std::cout << " -- PASSED" << std::endl;
    else     std::cout << " ----- ALERT! FAILED test `" << label << "` -----" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
    this_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(this_result);
}

template<typename T>
static void run_type(test_data &results, const std::string &type_name) {
    run_one<T, false, false, false, false>(results, "tcgen05_st_st_mm_AB=" + type_name);
    run_one<T, false, false, true,  false>(results, "tcgen05_st_st_mm_ABt=" + type_name);
    run_one<T, false, true,  false, false>(results, "tcgen05_st_st_mm_AtB=" + type_name);
    run_one<T, false, true,  true,  false>(results, "tcgen05_st_st_mm_AtBt=" + type_name);
    run_one<T, false, false, false, true >(results, "tcgen05_st_st_mma_AB=" + type_name);
    run_one<T, false, false, true,  true >(results, "tcgen05_st_st_mma_ABt=" + type_name);
    run_one<T, false, true,  false, true >(results, "tcgen05_st_st_mma_AtB=" + type_name);
    run_one<T, false, true,  true,  true >(results, "tcgen05_st_st_mma_AtBt=" + type_name);

    run_one<T, true,  false, false, false>(results, "tcgen05_tt_st_mm_AB=" + type_name);
    run_one<T, true,  false, true,  false>(results, "tcgen05_tt_st_mm_ABt=" + type_name);
    run_one<T, true,  false, false, true >(results, "tcgen05_tt_st_mma_AB=" + type_name);
    run_one<T, true,  false, true,  true >(results, "tcgen05_tt_st_mma_ABt=" + type_name);
}

}

void group::mma::tensor::mma::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/mma/tensor/mma tests! -----\n" << std::endl;
    run_type<kittens::bf16>(results, "bf16");
    run_type<kittens::half>(results, "half");
    run_type<kittens::fp8e4m3>(results, "fp8e4m3");
    run_type<kittens::fp8e5m2>(results, "fp8e5m2");
    run_type<kittens::int8>(results, "int8");
    run_type<kittens::uint8>(results, "uint8");
    std::cout << std::endl;
}

#endif
