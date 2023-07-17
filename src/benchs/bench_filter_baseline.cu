#include "config.cuh"
#include <kernels/filter_baseline.cuh>

template <size_t block_size, int bits>
float bench_filter_baseline(
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
    // generate bitmask based on predacte(on gpu)
    float time;
    CUDA_QUICKTIME(&time, {
        kernel_filter_baseline<block_size, bits, FILTER_PREDICATE_OP<bits>><<<block_count, block_size>>>(d_data1, element_count, d_output);
    });
    CUDA_TRY(cudaDeviceSynchronize());

    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, bitmask_size, cudaMemcpyDeviceToHost));

    // validate on cpu
    for (size_t i = 0; i < element_count; i++) {
        bool expected = FILTER_PREDICATE_OP<bits>(h_data1[i] & (((uint64_t)~0) >> (64 - bits)));
        bool result = get_bitmask_element<bits>(h_output, i);
        if (expected != result) {
            bit_print(h_data1[i] & (((uint64_t)~0) >> (64 - bits)), true, true, true, "\n");
            bit_print(h_output[i / 64], true, true, true, "\n");
            printf("validation fail on element %lu (expected %u, got %u)\n", i, expected, result);
            assert(false);
            exit(-1);
        }
    }
    return time;
}

struct filter_baseline_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return sizeof(uint32_t) * block_size;
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
        return bench_filter_baseline<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_filter_baseline(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<filter_baseline_launcher, 64>::call(
        "filter_baseline", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
