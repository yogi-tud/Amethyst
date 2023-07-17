#pragma once

#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "utils.cuh"
#include "fast_prng.cuh"
#include <cstring>

template <int bits>
struct tight_type {
    static constexpr size_t bytecount = (bits + 7) / 8;
    typedef typename std::conditional<
        (bytecount > 4),
        uint64_t,
        typename std::conditional<(bytecount > 2), uint32_t, typename std::conditional<(bytecount > 1), uint16_t, uint8_t>::type>::type>::type type;
    typedef typename std::conditional<(bytecount > 4), unsigned long long, unsigned int>::type cuda_type;
};

#define THREADS_PER_WARP 32

struct warp_stats {
    size_t idx;
    size_t offset;
    size_t base;
};

inline __device__ warp_stats get_warp_stats()
{
    warp_stats w;
    w.idx = threadIdx.x / THREADS_PER_WARP;
    w.offset = threadIdx.x % THREADS_PER_WARP;
    w.base = THREADS_PER_WARP * w.idx;
    return w;
}

constexpr inline __device__ __host__ uint64_t gen_bitmask(int bits)
{
    return ((uint64_t)~0) >> (64 - bits);
}

template <int bits>
__device__ __host__ uint64_t get_element(uint64_t* data, size_t index)
{
    size_t elems_per_slab = (64 / bits);
    size_t slab_index = index / elems_per_slab;
    size_t offset = (index - slab_index * elems_per_slab) * bits;
    return (data[slab_index] >> offset) & gen_bitmask(bits);
}
template <int bits>
__device__ __host__ uint64_t get_element_baseline(uint64_t* data, size_t index)
{
    return data[index] & gen_bitmask(bits);
}

template <int bits>
__device__ __host__ bool get_bitmask_element(uint64_t* data, size_t index)
{
    size_t slab_index = index / 64;
    size_t offset = index - slab_index * 64;
    return (data[slab_index] >> offset) & 1;
}

struct matrix_mul_rowcolcnt {
    size_t cols_l;
    size_t rows_l;
    size_t cols_r;
};

static constexpr matrix_mul_rowcolcnt matrix_mul_calc_rowcolcnt(int bits, size_t element_count)
{
    size_t elems_per_slab = 64 / bits;
    // we subtract elems_per_slab from this to prevent overflow in the output
    size_t cols_l = (size_t)std::sqrt(element_count);
    while ((cols_l + elems_per_slab - 1) / elems_per_slab * elems_per_slab * (element_count / cols_l) > element_count && cols_l > 1) {
        cols_l--;
    }
    size_t rows_l = element_count / cols_l;
    size_t cols_r = rows_l;
    // reduce cols_r in case output (rows_l * cols_r) is too big for elems
    while (rows_l * cols_r * elems_per_slab > element_count && cols_r > elems_per_slab) {
        cols_r -= elems_per_slab;
    }
    return (matrix_mul_rowcolcnt){.cols_l = cols_l, .rows_l = rows_l, .cols_r = cols_r};
}

#define WARP_SIZE 32
#define ELEMS_PER_THREAD 32

extern bool do_validation;

extern int run_count;

extern size_t grid_size_min;
extern size_t grid_size_max;
extern size_t block_size_min;
extern size_t block_size_max;

template <int bits>
uint64_t get_elem_nogaps(uint64_t* buf, size_t idx)
{
    size_t slab = (idx * bits) / 64;
    size_t offset = (idx * bits) % 64;
    uint64_t elem;
    size_t remaining_bits = (64 - offset);
    if (remaining_bits < bits) {
        elem = (buf[slab] >> offset) & gen_bitmask(remaining_bits);
        offset = bits - remaining_bits;
        elem |= (buf[slab + 1] & gen_bitmask(offset)) << remaining_bits;
    }
    else {
        elem = (buf[slab] >> offset) & gen_bitmask(bits);
    }
    return elem;
}

template <typename elem_type>
inline __host__ __device__ uint64_t get_matrix_element_noslabs(int bits, size_t rows, size_t cols, elem_type* data, size_t r, size_t c)
{
    return (uint64_t)data[r * cols + c] & gen_bitmask(bits);
}

inline uint64_t get_matrix_element(size_t bits, size_t rows, size_t cols, uint64_t* data, size_t r, size_t c)
{
    size_t elems_per_slab = 64 / bits;
    size_t row_slab_count = (cols + elems_per_slab - 1) / elems_per_slab;
    return (data[r * row_slab_count + c / elems_per_slab] >> (bits * (c % elems_per_slab))) & gen_bitmask(bits);
}

inline __host__ __device__ double get_matrix_element_noslabs_double(size_t rows, size_t cols, double* data, size_t r, size_t c)
{
    return data[r * cols + c];
}

static const size_t MAX_SHARED_MEMORY = 0xc000; // min of CC 6.5 and 7.5

inline constexpr size_t max_shared_memory_size_hop(int bits, size_t block_size)
{
    size_t slab_count = (block_size * ELEMS_PER_THREAD * bits + 63) / 64 + (1024 / bits * bits == 1024 ? 0 : 1);
    return sizeof(uint64_t) * slab_count;
}

inline constexpr size_t max_shared_memory_size_unary(int bits, size_t block_size)
{
    size_t elems_per_slab = 64 / bits;
    size_t slabs_per_warp =
        (WARP_SIZE * ELEMS_PER_THREAD + elems_per_slab - 1) / elems_per_slab + (1024 / elems_per_slab * elems_per_slab == 1024 ? 0 : 1);
    return sizeof(uint64_t) * (slabs_per_warp * block_size / WARP_SIZE);
}

inline constexpr size_t max_shared_memory_size_binary_nogap(int bits, int outbits, size_t block_size)
{
    size_t out_elems_per_iteration = (block_size * 64 + outbits - 1) / outbits;
    size_t slab_count = (out_elems_per_iteration * bits + 63) / 64 + (block_size * 64 / outbits * outbits == block_size * 64 ? 0 : 2);
    return 2 * sizeof(uint64_t) * slab_count;
}

inline constexpr size_t max_shared_memory_size_binary_slab(int bits, int outbits, size_t block_size)
{
    size_t elems_per_slab_in = 64 / bits;
    size_t elems_per_slab_out = 64 / outbits;
    size_t out_elems_per_iteration = block_size * elems_per_slab_out;
    size_t readin_slab_count =
        (out_elems_per_iteration + elems_per_slab_in - 1) / elems_per_slab_in + ((elems_per_slab_out * 1024) % elems_per_slab_in != 0 ? 1 : 0);
    return readin_slab_count * sizeof(uint64_t) * 2;
}

template <typename launcher>
inline constexpr size_t max_block_size(int bits, size_t maximum_block_size)
{
    size_t smem = launcher::smem_usage(bits, maximum_block_size);
    if (maximum_block_size == 32) return (smem > MAX_SHARED_MEMORY ? 0 : 32);
    return (smem > MAX_SHARED_MEMORY ? max_block_size<launcher>(bits, maximum_block_size / 2) : maximum_block_size);
}

extern int wantbits;

template <typename launcher, int bits>
struct bench_elementstuffing_wrapper {
    static void call(
        const char* approach_name,
        size_t element_count,
        uint64_t* h_data1,
        uint64_t* h_data2,
        uint64_t* h_output,
        uint64_t* d_data1,
        uint64_t* d_data2,
        uint64_t* d_output,
        size_t bitmask_size)
    {
        if (bits <= wantbits) {
            constexpr size_t calc_max_block_size = launcher::uses_block_size ? max_block_size<launcher>(bits, 1024) : 32;
            size_t block_size_max_present = launcher::uses_block_size ? calc_max_block_size : block_size_min;
            size_t grid_size_max_present = launcher::uses_block_size ? grid_size_max : grid_size_min;
            for (size_t block_count = grid_size_min; block_count <= grid_size_max_present; block_count *= 2) {
                for (size_t block_size = block_size_min; block_size <= block_size_max_present && block_size <= block_size_max; block_size *= 2) {
                    for (int i = 0; i < run_count; i++) {
                        float t = pow2_template_dispatch<float, launcher, 32, calc_max_block_size, std::integral_constant<int, bits>>::call(
                            launcher::uses_block_size ? block_size : 32, element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count,
                            do_validation && i == 0);
                        if (t < 0) {
                            continue;
                        }
                        assert(t != 0);
                        float throughput = (launcher::used_data_size(bits, element_count) / (t / 1000)) / (1 << 30);
                        printf(
                            "%s;%zu;%i;%zi;%zi;%f;%f;%i\n",
                            approach_name, element_count, bits,
                            launcher::uses_block_size ? (ssize_t)block_count : -1,
                            launcher::uses_block_size ? (ssize_t)block_size : -1,
                            t, throughput, i
                        );
                        fflush(stdout);
                        if (do_validation) {
                            CUDA_TRY(cudaMemset(d_output, 0x00, element_count * sizeof(uint64_t)));
                        }
                    }
                }
            }
        }
        bench_elementstuffing_wrapper<launcher, bits - 1>::call(
            approach_name, element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
    }
};

template <typename launcher>
struct bench_elementstuffing_wrapper<launcher, 0> {
    static void call(
        const char* approach_name,
        size_t element_count,
        uint64_t* h_data1,
        uint64_t* h_data2,
        uint64_t* h_output,
        uint64_t* d_data1,
        uint64_t* d_data2,
        uint64_t* d_output,
        size_t bitmask_size)
    {
    }
};
