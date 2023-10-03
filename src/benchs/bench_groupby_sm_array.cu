#include "config.cuh"
#include "slab_iterator.cuh"
#include <cmath>
#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <unordered_map>
#include <kernels/groupby_sm_array.cuh>

template <size_t block_size, int bits>
float bench_groupby_sm_array(
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
    constexpr size_t elems_per_slab = 64 / bits;
    typedef typename tight_type<bits>::cuda_type cuda_type;
    constexpr size_t max_group_count = (size_t)1 << bits;  
    cuda_type* d_aggregates_temp;
    size_t aggregates_temp_size = (sizeof(cuda_type) + sizeof(unsigned int)) * max_group_count;
    CUDA_TRY(cudaMalloc(&d_aggregates_temp, aggregates_temp_size));
    uint64_t* d_keys_unique = d_output;
    uint64_t* d_aggregates = d_output + element_count;

    unsigned int* d_occurences_temp = (unsigned int*)(d_aggregates_temp + max_group_count);

    size_t h_unique_key_count = 0;
    size_t* d_unique_key_count;
    CUDA_TRY(cudaMalloc(&d_unique_key_count, sizeof(size_t)));

    float time;
    dim3 griddim(block_count);
    dim3 blockdim(block_size);
    void* kargs[] = {&d_data1, &d_data2, &d_aggregates_temp, &d_occurences_temp, &element_count, &d_unique_key_count};
    void* kargs_writeout[] = {&d_aggregates_temp, &d_occurences_temp, &d_keys_unique, &d_aggregates, &element_count, &d_unique_key_count};
    cudaError_t launch_err;
    CUDA_QUICKTIME(&time, {
        // init unique key count on device to 0
        CUDA_TRY(cudaMemcpy(d_unique_key_count, &h_unique_key_count, sizeof(size_t), cudaMemcpyHostToDevice));
        // initialize the aggregate values
        thrust::fill(
            thrust::device_ptr<cuda_type>(d_aggregates_temp), thrust::device_ptr<cuda_type>(d_aggregates_temp + max_group_count),
            (cuda_type)GROUP_BY_INIT_VALUE);
        // we use atomicOr to concurrently set the aggregate values of slabs, therefore this needs to be zeroed initially
        cudaMemset(d_aggregates, 0, ((element_count + elems_per_slab - 1) / elems_per_slab) * sizeof(uint64_t));
        // chevron launch does not return launch errors for out of registers (i.e. cudaErrorLaunchOutOfResources)
        launch_err = cudaLaunchKernel((void*)kernel_groupby_sm_array_gather<block_size, bits>, griddim, blockdim, kargs, 0, 0);
        if (launch_err == cudaErrorLaunchOutOfResources) {
            return -1;
        }
        launch_err = cudaLaunchKernel((void*)kernel_groupby_sm_array_writeout<block_size, bits>, griddim, blockdim, kargs_writeout, 0, 0);
        if (launch_err == cudaErrorLaunchOutOfResources) {
            return -1;
        }
    });
    CUDA_TRY(launch_err);
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(&h_unique_key_count, d_unique_key_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(d_unique_key_count));

    if (!validate) {
        return time; // turns off validation
    }
    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, sizeof(uint64_t) * element_count * 2, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(d_aggregates_temp));

    // validate on cpu
    uint64_t* h_data_keys = h_data1;
    uint64_t* h_data_values = h_data2;
    uint64_t* h_output_keys_unqiue = h_output; // keys
    uint64_t* h_output_aggregates = h_output + element_count; // aggregates

    // for (size_t i = 0; i < h_unique_key_count; i++) {
    //     printf("reduced key [%lu]: %lu = %lu\n", i, (uint64_t)(sorttype)h_output_keys_unqiue[i], (uint64_t)(sorttype)h_output_aggregates[i]);
    // }

    std::unordered_map<uint64_t, uint64_t> aggregates{};
    for (size_t i = 0; i < element_count; i++) {
        typename std::unordered_map<uint64_t, uint64_t>::iterator elem = aggregates.find(get_element<bits>(h_data_keys, i));
        if (elem == aggregates.end()) {
            elem = aggregates.emplace(get_element<bits>(h_data_keys, i), 0).first;
        }
        elem->second = GROUP_BY_OP<bits>(elem->second, get_element<bits>(h_data_values, i));
    }

    for (size_t i = 0; i < std::min(h_unique_key_count, aggregates.size()); i++) {
        uint64_t key = get_element<bits>(h_output_keys_unqiue, i);
        uint64_t out_agg = get_element<bits>(h_output_aggregates, i);
        auto map_agg_iter = aggregates.find(key);
        if (map_agg_iter == aggregates.end()) {
            printf("validation failed, unexpected [%lu] = %lu from gpu at %lu\n", key, out_agg, i);
            printf("validation setup: bits:%i grid_size:%lu block_size:%lu\n", bits, block_size, block_count);
            assert(false);
            exit(-1);
        }
        uint64_t map_agg = map_agg_iter->second & gen_bitmask(bits);
        if (out_agg != map_agg) {
            printf("validation failed, expected [%lu]=%lu, got [%lu]=%lu, offset %lu\n", key, map_agg, key, out_agg, i % elems_per_slab);
            printf("key%p value%p\n", d_keys_unique + i / elems_per_slab, d_aggregates + i / elems_per_slab);
            printf("validation setup: bits:%i grid_size:%lu block_size:%lu\n", bits, block_size, block_count);
            assert(false);
            exit(-1);
        }
    }
    if (h_unique_key_count != aggregates.size()) {
        printf("validation failed, expected %lu keys, got %lu\n", aggregates.size(), h_unique_key_count);
        printf("validation setup: bits:%i grid_size:%lu block_size:%lu\n", bits, block_size, block_count);
        assert(false);
        exit(-1);
    }
    return time;
}

struct groupby_sm_array_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        if (bits > 32) {
            return MAX_SHARED_MEMORY + 1;
        }
        size_t max_group_count = (size_t)1 << bits;
        size_t cuda_elem_width = bits > 32 ? 8 : 4;
        return (sizeof(uint64_t) * 2) * block_size + (sizeof(unsigned int) + cuda_elem_width) * max_group_count;
    }
    static float used_data_size(int bits, size_t element_count)
    {
        return 2 * element_count * sizeof(uint64_t);
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
        return bench_groupby_sm_array<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_groupby_sm_array(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<groupby_sm_array_launcher, 64>::call(
        "groupby_sm_array", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
