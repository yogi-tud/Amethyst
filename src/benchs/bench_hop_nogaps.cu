#include "common_code.cuh"

#include <kernels/hop.cuh>

template <size_t block_size, int bits, bool slabs>
float bench_hop_nogaps(
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

    CUDA_TRY(cudaMemcpy(d_output, &HOP_INIT, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // generate bitmask based on predacte(on gpu)
    float time;
    CUDA_QUICKTIME(&time, {
        std::conditional<
            slabs, slabs_call<block_size, bits, HOP_INIT, HOP_OP, HOP_OP_ATOMIC>,
            nogaps_call<block_size, bits, HOP_INIT, HOP_OP, HOP_OP_ATOMIC>>::type::call(block_count, d_data1, element_count, d_output);
    });
    CUDA_TRY(cudaDeviceSynchronize());

    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // validate on cpu
    uint64_t sum = HOP_INIT;
    for (size_t i = 0; i < element_count; i++) {
        if (slabs) {
            sum = HOP_OP(sum, get_element<bits>(h_data1, i));
        }
        else {
            sum = HOP_OP(sum, get_elem_nogaps<bits>(h_data1, i));
        }
    }
    if (sum != h_output[0]) {
        printf("validation failed, expected 0x%lx, got 0x%lx\n", sum, h_output[0]);
        assert(false);
        exit(-1);
    }
    return time;
}

template <bool slabs> // instead of nogaps
struct hop_nogaps_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return slabs ? 0 : max_shared_memory_size_hop(bits, block_size);
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
        return bench_hop_nogaps<block_size, bits::value, slabs>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size, block_count, validate);
    }
};

void bench_hop_nogaps(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<hop_nogaps_launcher<false>, 64>::call(
        "hop_nogaps", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
