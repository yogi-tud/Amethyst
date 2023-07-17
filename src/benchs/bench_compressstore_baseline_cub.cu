#include "common_code.cuh"
#include <kernels/compressstore_slabs.cuh>
#include <cub/device/device_select.cuh>
#include <bit_iterator.cuh>

__device__ size_t compressstore_baseline_cub_output_element_count;

template <size_t block_size, int bits>
float bench_compressstore_baseline_cub(
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

    uint64_t* d_elements = d_data1;
    uint8_t* d_mask = (uint8_t*)d_data2;
    bit_iterator d_mask_iter{d_mask};

    size_t cub_intermediate_size;
    size_t* d_output_element_count;
    CUDA_TRY(cudaGetSymbolAddress((void **)&d_output_element_count, compressstore_baseline_cub_output_element_count));
    cub::DeviceSelect::Flagged(NULL, cub_intermediate_size, d_elements, d_mask_iter, d_output, d_output_element_count, element_count);
    void* cub_intermediate_storage;
    CUDA_TRY(cudaMalloc(&cub_intermediate_storage, cub_intermediate_size));

    float time;
    CUDA_QUICKTIME(&time, {
        cub::DeviceSelect::Flagged(cub_intermediate_storage, cub_intermediate_size, d_elements, d_mask_iter, d_output, d_output_element_count, element_count);
    });
    CUDA_TRY(cudaDeviceSynchronize());
    cudaFree(cub_intermediate_storage);
    if (!validate) {
        return time; // turns off validation
    }
    // memcpy to cpu
    size_t output_element_count;
    CUDA_TRY(cudaMemcpy(&output_element_count, d_output_element_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_output, d_output, ceildiv(output_element_count * sizeof(uint64_t), 64 / bits), cudaMemcpyDeviceToHost));

    // validate on cpu
    bit_iterator mask_iter{(uint8_t*)h_data2};
    size_t selected_elem_count = 0;
    for (size_t i = 0; i < element_count; i++) {
        if (*mask_iter) {
            if (selected_elem_count == output_element_count) {
                printf("validation fail: expected %lu selected elements, got only %lu\n", selected_elem_count, output_element_count);
                fflush(stdout);
                assert(false);
                exit(-1);
            }
            auto expected = get_element_baseline<bits>(h_data1, i);
            auto got = get_element_baseline<bits>(h_output, selected_elem_count);
            if (expected != got) {
                printf("validation fail on output element %lu: expected %lu (input index %lu), got %lu\n", selected_elem_count, expected, i, got);
                fflush(stdout);
                assert(false);
                exit(-1);
            };
            selected_elem_count++;
        }
        mask_iter++;
    }
    return time;
}

struct compressstore_baseline_cub_launcher {
    static constexpr bool uses_block_size = false;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return 0;
    }
    static float used_data_size(int bits, size_t element_count)
    {
        return element_count * sizeof(uint64_t) + (element_count + 7) / 8;
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
        return bench_compressstore_baseline_cub<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_compressstore_baseline_cub(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<compressstore_baseline_cub_launcher, 64>::call(
        "compressstore_baseline_cub", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
