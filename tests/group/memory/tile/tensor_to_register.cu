#include "tensor_to_register.cuh"

#if defined(TEST_GROUP_MEMORY_TILE_TENSOR_TO_REGISTER) && defined(KITTENS_SM10X)

template<typename T>
struct group_tensor_reg_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all RL> using valid = std::bool_constant<
        ( H%NW==0 && W*H<=64 ) &&
        ( NW==1 || NW==4 || NW==8 ) &&
        ( 16*H==kittens::MAX_TENSOR_ROWS/2 || 16*H==kittens::MAX_TENSOR_ROWS ) &&
        ( NW==1 || 16*H==kittens::MAX_TENSOR_ROWS ) &&
        ( (16*H) / (4 / sizeof(T)) <= kittens::MAX_TENSOR_ROWS ) &&
        ( (16*W) / (4 / sizeof(T)) <= kittens::MAX_TENSOR_COLS ) &&
        ( sizeof(T) != 1 || W%2 == 0 )
    >;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_tensor_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_tensor_reg_loadstore_gmem=half" :
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "group_tensor_reg_loadstore_gmem=fp8e4m3" :
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "group_tensor_reg_loadstore_gmem=fp8e5m2" :
                                                      std::is_same_v<T, kittens::int8> ? "group_tensor_reg_loadstore_gmem=int8" :
                                                      std::is_same_v<T, kittens::uint8> ? "group_tensor_reg_loadstore_gmem=uint8" :
                                                      std::is_same_v<T, int> ? "group_tensor_reg_loadstore_gmem=int" :
                                                                                         "group_tensor_reg_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all RL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        using RT = kittens::rt<dtype, 16*H/NW, 16*W, RL>;
        using TT = kittens::tt<dtype, 16*H, 16*W>;

        kittens::tensor_allocator<1, 1> tm_alloc{};
        TT tensor_tile;
        if constexpr (kittens::ducks::tt::full<TT>) {
            tensor_tile = tm_alloc.template allocate<TT>(0);
        }
        else {
            tensor_tile = tm_alloc.template allocate<TT>(0, 0);
        }

        RT src_reg;
        RT dst_reg;
        if constexpr (sizeof(dtype) == 1) {
            using ST = kittens::st<dtype, 16*H, 16*W>;
            extern __shared__ kittens::alignment_dummy __shm[];
            kittens::shared_allocator<16> al((int*)&__shm[0]);
            ST &shared_tile = al.allocate<ST>();
            G::load(shared_tile, input, {});
            __syncthreads();
            G::load(src_reg, shared_tile);
            G::store_async(tensor_tile, src_reg);
            kittens::tensor_store_wait();
            __syncthreads();
            G::load_async(dst_reg, tensor_tile);
            kittens::tensor_load_wait();
            __syncthreads();
            G::store(shared_tile, dst_reg);
            __syncthreads();
            G::store(output, shared_tile, {});
        }
        else {
            G::load(src_reg, input, {});
            G::store_async(tensor_tile, src_reg);
            kittens::tensor_store_wait();
            G::load_async(dst_reg, tensor_tile);
            kittens::tensor_load_wait();
            G::store(output, dst_reg, {});
        }
    }
};

void group::memory::tile::tensor_to_register::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/memory/tile/tensor_to_register tests! -----\n" << std::endl;
    wrapper_2d<group_tensor_reg_load_store<float>, 8, 2, 4, kittens::ducks::rt_layout::row>::run(results);
    wrapper_2d<group_tensor_reg_load_store<float>, 8, 2, 8, kittens::ducks::rt_layout::row>::run(results);

    wrapper_2d<group_tensor_reg_load_store<int>, 8, 2, 4, kittens::ducks::rt_layout::row>::run(results);
    wrapper_2d<group_tensor_reg_load_store<int>, 8, 2, 8, kittens::ducks::rt_layout::row>::run(results);

    wrapper_2d<group_tensor_reg_load_store<kittens::fp8e4m3>, 8, 2, 4, kittens::ducks::rt_layout::row>::run(results);
    wrapper_2d<group_tensor_reg_load_store<kittens::fp8e4m3>, 8, 2, 8, kittens::ducks::rt_layout::row>::run(results);

    wrapper_2d<group_tensor_reg_load_store<kittens::fp8e5m2>, 8, 2, 4, kittens::ducks::rt_layout::row>::run(results);
    wrapper_2d<group_tensor_reg_load_store<kittens::fp8e5m2>, 8, 2, 8, kittens::ducks::rt_layout::row>::run(results);

    std::cout << std::endl;
}

#endif
