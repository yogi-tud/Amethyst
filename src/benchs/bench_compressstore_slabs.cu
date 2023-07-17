#include "common_code.cuh"
#include <kernels/compressstore_slabs.cuh>
#include <cub/device/device_scan.cuh>
#include <bit_iterator.cuh>

template <size_t block_size, int bits>
float bench_compressstore_slabs(
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
    size_t chunk_count = ceildiv(element_count, COMPRESSTORE_CHUNK_LENGTH);

    uint64_t* d_elements = d_data1;
    uint8_t* d_mask = (uint8_t*)d_data2;
    size_t* d_popcount = (size_t*)(d_output + 2 * element_count);
    size_t* d_prefix_sum = d_popcount + chunk_count;

    size_t cub_intermediate_size;
    cub::DeviceScan::ExclusiveSum(NULL, cub_intermediate_size, d_popcount, d_prefix_sum, chunk_count);
    void* cub_intermediate_storage;
    CUDA_TRY(cudaMalloc(&cub_intermediate_storage, cub_intermediate_size));

    // generate bitmask based on predacte(on gpu)
    float time;
    //printf("elems per slab: %lu\n", 64 / bits);
    //printf("element count: %lu\n", element_count);

    CUDA_QUICKTIME(&time, {
        //puts("elements: ");
        //gpu_buffer_print_le(d_elements, 0, element_count);
        //puts("mask: ");
        //gpu_buffer_print_le(d_mask, 0, ceildiv(element_count, 64));
        kernel_compresstore_slabs_chunk_popcount<bits, COMPRESSTORE_CHUNK_LENGTH><<<block_count, block_size>>>(d_mask, element_count, d_popcount);
        //puts("popcount: ");
        //gpu_buffer_print_le(d_popcount, 0, chunk_count);
        cub::DeviceScan::ExclusiveSum(cub_intermediate_storage, cub_intermediate_size, d_popcount, d_prefix_sum, chunk_count);
        //puts("prefixsum: ");
        //gpu_buffer_print_le(d_prefix_sum, 0, chunk_count);
        void* kargs_writeout[] = {&d_elements, &d_output, &d_mask, &d_popcount, &d_prefix_sum, &element_count};
        int launch_err = cudaLaunchKernel((void*)kernel_compresstore_slabs_write<bits, COMPRESSTORE_CHUNK_LENGTH, block_size>, dim3(block_count), dim3(block_size), kargs_writeout, 0, 0);
        if (launch_err == cudaErrorLaunchOutOfResources) {
            return -1;
        }
       /* kernel_compresstore_slabs_write<bits, COMPRESSTORE_CHUNK_LENGTH, block_size>
            <<<block_count, block_size>>>(d_elements, d_output, d_mask, d_popcount, d_prefix_sum, element_count);*/
    });
    CUDA_TRY(cudaDeviceSynchronize());
    cudaFree(cub_intermediate_storage);
    size_t last_chunk_prefix_sum, last_chunk_popcount;
    CUDA_TRY(cudaMemcpy(&last_chunk_prefix_sum, d_prefix_sum + chunk_count - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(&last_chunk_popcount, d_popcount + chunk_count - 1, sizeof(size_t), cudaMemcpyDeviceToHost));
    size_t output_element_count = last_chunk_prefix_sum + last_chunk_popcount;
    //printf("output element count: %lu\n", output_element_count);
    //puts("output: ");
    //gpu_buffer_print_le(d_output, 0, output_element_count);
    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
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
            auto expected = get_element<bits>(h_data1, i);
            auto got = get_element<bits>(h_output, selected_elem_count);
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

struct compressstore_slabs_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        size_t elems_per_slab = 64 / bits;
        uint32_t warps_per_block = block_size / WARP_SIZE;
        uint32_t elems_per_flush = WARP_SIZE * elems_per_slab;
        return (block_size + warps_per_block * elems_per_flush + warps_per_block) * sizeof(uint32_t);
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
        return bench_compressstore_slabs<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_compressstore_slabs(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<compressstore_slabs_launcher, 64>::call(
        "compressstore_slabs", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
