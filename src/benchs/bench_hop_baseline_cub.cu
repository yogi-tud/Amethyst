#include <cub/device/device_reduce.cuh>
#include "config.cuh"

struct hop_op_functor {
    __device__ __forceinline__ uint64_t operator()(const uint64_t a, const uint64_t b) const
    {
        return HOP_OP(a, b);
    }
};

template <size_t block_size, int bits>
float bench_hop_baseline(
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
    hop_op_functor op_functor{};
    if (block_count > grid_size_min || block_size > block_size_min) {
        return -1;
    }
    CUDA_TRY(cudaMemcpy(d_output, &HOP_INIT, sizeof(uint64_t), cudaMemcpyHostToDevice));

    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, d_data1, d_output, element_count, op_functor, HOP_INIT);
    if (temp_storage_bytes > element_count * sizeof(uint64_t)) {
        assert(0);
    }

    float time;
    CUDA_QUICKTIME(&time, { cub::DeviceReduce::Reduce(d_data2, temp_storage_bytes, d_data1, d_output, element_count, op_functor, HOP_INIT); });
    CUDA_TRY(cudaDeviceSynchronize());

    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // validate on cpu
    uint64_t sum = HOP_INIT;
    for (size_t i = 0; i < element_count; i++) {
        sum = HOP_OP(sum, h_data1[i]);
    }
    if (sum != h_output[0]) {
        printf("validation failed, expected 0x%lx, got 0x%lx\n", sum, h_output[0]);
        assert(false);
        exit(-1);
    }
    return time;
}

struct hop_baseline_launcher {
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
        return bench_hop_baseline<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_hop_baseline(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<hop_baseline_launcher, 64>::call(
        "hop_baseline", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
