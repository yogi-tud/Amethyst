#include "common_code.cuh"
#include <kernels/filter_slabs.cuh>

template <size_t block_size, int bits>
float bench_filter_slabs(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size,
    size_t block_count,
    bool by_warp,
    bool validate)
{
    // generate bitmask based on predacte(on gpu)
    float time;
    CUDA_QUICKTIME(&time, {
        if (by_warp) {
            kernel_filter_slabs_bywarp<block_size, bits, FILTER_PREDICATE_OP<bits>><<<block_count, block_size>>>(d_data1, element_count, d_output);
        }
        else {
            kernel_filter_slabs<block_size, bits, FILTER_PREDICATE_OP<bits>><<<block_count, block_size>>>(d_data1, element_count, d_output);
        }
    });
    CUDA_TRY(cudaDeviceSynchronize());

    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, bitmask_size, cudaMemcpyDeviceToHost));

    // validate on cpu
    for (size_t i = 0; i < element_count; i++) {
        if (FILTER_PREDICATE_OP<bits>(get_element<bits>(h_data1, i)) != get_bitmask_element<bits>(h_output, i)) {
            bit_print(get_element<bits>(h_data1, i), true, true, true, "\n");
            bit_print(h_output[i / 64], true, true, true, "\n");
            printf("validation fail on element %lu\n", i);
            assert(false);
            exit(-1);
        }
    }
    return time;
}

template <bool by_warp>
struct filter_slabs_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return max_shared_memory_size_unary(bits, block_size);
    }
    static float used_data_size(int bits, size_t element_count)
    {
        return element_count * sizeof(uint64_t);
    }
    template <typename bits, size_t block_size>
    static float call(
        size_t element_count,
        uint64_t* h_data1,
        uint64_t* h_data2,
        uint64_t* h_output,
        uint64_t* d_data1,
        uint64_t* d_data2,
        uint64_t* d_output,
        size_t bitmask_size,
        size_t block_count,
        bool validate)
    {
        return bench_filter_slabs<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, by_warp, validate);
    }
};

void bench_filter_slabs(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<filter_slabs_launcher<false>, 64>::call(
        "filter_slabs", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}

void bench_filter_slabs_by_warp(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<filter_slabs_launcher<true>, 64>::call(
        "filter_slabs_by_warp", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
