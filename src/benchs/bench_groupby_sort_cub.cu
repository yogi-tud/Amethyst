#include <unordered_map>
#include <iterator>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include "config.cuh"
#include "slab_iterator.cuh"

template <int bits>
struct group_by_op_functor {
    template <typename DataType>
    __host__ __device__ bool operator()(const DataType& lhs, const DataType& rhs)
    {
        return GROUP_BY_OP<bits>(lhs, rhs);
    }
};

template <size_t block_size, int bits>
float bench_groupby_sort_cub(
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
    if (block_count > grid_size_min || block_size > block_size_min) {
        return -1;
    }

    typedef typename tight_type<bits>::type sorttype;
    sorttype* d_keys = (sorttype*)d_data1;
    sorttype* d_values = (sorttype*)d_data2;
    sorttype* d_sorted_keys = (sorttype*)d_output;
    sorttype* d_sorted_values = (sorttype*)(d_output + element_count);
    sorttype* d_keys_unqiue = (sorttype*)(d_output + element_count * 2);
    sorttype* d_aggregates = (sorttype*)(d_output + element_count * 3);

    slab_iterator<bits> d_sorted_keys_iter{(uint64_t*)d_sorted_keys, 0};
    slab_iterator<bits> d_sorted_values_iter{(uint64_t*)d_sorted_values, 0};
    size_t h_unique_key_count;
    size_t* d_unique_key_count;
    CUDA_TRY(cudaMalloc(&d_unique_key_count, sizeof(size_t)));

    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_sorted_keys, d_values, d_sorted_values, element_count, 0, bits);
    size_t temp_storage_bytes2 = 0;
    cub::DeviceReduce::ReduceByKey(
        NULL, temp_storage_bytes2, d_sorted_keys_iter, d_keys_unqiue, d_sorted_values_iter, d_aggregates, d_unique_key_count,
        group_by_op_functor<bits>{}, element_count);
    void* temp_storage;
    CUDA_TRY(cudaMalloc(&temp_storage, std::max(temp_storage_bytes, temp_storage_bytes2)));

    float time;
    CUDA_QUICKTIME(&time, {
        cub::DeviceRadixSort::SortPairs(temp_storage, temp_storage_bytes, d_keys, d_sorted_keys, d_values, d_sorted_values, element_count, 0, bits);
        cub::DeviceReduce::ReduceByKey(
            temp_storage, temp_storage_bytes2, d_sorted_keys_iter, d_keys_unqiue, d_sorted_values_iter, d_aggregates, d_unique_key_count,
            group_by_op_functor<bits>{}, element_count);
    });
    CUDA_TRY(cudaDeviceSynchronize());
    CUDA_TRY(cudaMemcpy(&h_unique_key_count, d_unique_key_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(temp_storage));
    CUDA_TRY(cudaFree(d_unique_key_count));

    if (!validate) {
        return time; // turns off validation
    }

    sorttype* h_keys = (sorttype*)h_data1;
    sorttype* h_values = (sorttype*)h_data2;
    sorttype* h_keys_unqiue = (sorttype*)(h_output);
    sorttype* h_aggregates = h_keys_unqiue + h_unique_key_count;

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_keys_unqiue, d_keys_unqiue, h_unique_key_count * sizeof(sorttype), cudaMemcpyDeviceToHost)); // copy both output bufs
    CUDA_TRY(cudaMemcpy(h_aggregates, d_aggregates, h_unique_key_count * sizeof(sorttype), cudaMemcpyDeviceToHost)); // copy both output bufs

    // validate on cpu
    sorttype bitmask = (sorttype)gen_bitmask(bits);
    std::unordered_map<sorttype, sorttype> aggregates{};
    for (size_t i = 0; i < element_count; i++) {
        sorttype key = h_keys[i] & bitmask;
        sorttype value = h_values[i] & bitmask;
        typename std::unordered_map<sorttype, sorttype>::iterator elem = aggregates.find(key);
        if (elem == aggregates.end()) {
            elem = aggregates.emplace(key, GROUP_BY_INIT_VALUE).first;
        }
        elem->second = GROUP_BY_OP<bits>(elem->second, value);
    }
    if (h_unique_key_count != aggregates.size()) {
        printf("validation failed, expected %lu keys, got %lu\n", aggregates.size(), h_unique_key_count);
        assert(false);
        exit(-1);
    }
    for (size_t i = 0; i < h_unique_key_count; i++) {
        if (h_aggregates[i] != aggregates[h_keys_unqiue[i]]) {
            printf(
                "validation failed, expected [%lu]=%lu, got %lu\n", (uint64_t)h_keys_unqiue[i], (uint64_t)aggregates[h_keys_unqiue[i]],
                (uint64_t)h_aggregates[i]);
            assert(false);
            exit(-1);
        }
    }
    return time;
}

struct groupby_sort_cub_launcher {
    static constexpr bool uses_block_size = false;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return 0;
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
        return bench_groupby_sort_cub<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_groupby_sort_cub(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<groupby_sort_cub_launcher, 2>::call( //NOCHECKIN //this thing is VERY slow
        "groupby_sort", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
