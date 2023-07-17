#include "common_code.cuh"
#include <kernels/binary_op_slabs.cuh>

template <size_t block_size, int bits>
float bench_binary_op_slabs(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t block_count,
    bool validate)
{
    constexpr size_t outbits = BINARY_OP_OUTBITS(bits);
    // generate bitmask based on predacte(on gpu)
    float time;
    CUDA_QUICKTIME(&time, {
        kernel_binary_op_slabs<block_size, bits, outbits, BINARY_OP<bits>><<<block_count, block_size>>>(d_data1, d_data2, d_output, element_count);
    });
    CUDA_TRY(cudaDeviceSynchronize());

    if (!validate) {
        return time; // turns off validation
    }

    // memcpy to cpu
    CUDA_TRY(cudaMemcpy(h_output, d_output, element_count * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // validate on cpu
    for (size_t i = 0; i < element_count; i++) {
        if (BINARY_OP<bits>(get_element<bits>(h_data1, i), get_element<bits>(h_data2, i)) != get_element<outbits>(h_output, i)) {
            printf(
                "expected %lu, got %lu\n", BINARY_OP<bits>(get_element<bits>(h_data1, i), get_element<bits>(h_data2, i)),
                get_element<outbits>(h_output, i));
            printf("validation fail on element %lu\n", i);
            assert(false);
            exit(-1);
        }
    }
    return time;
}

struct binary_op_slabs_launcher {
    static constexpr bool uses_block_size = true;
    static constexpr size_t smem_usage(int bits, size_t block_size)
    {
        return max_shared_memory_size_binary_slab(bits, BINARY_OP_OUTBITS(bits), block_size);
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
        return bench_binary_op_slabs<block_size, bits::value>(
            element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, block_count, validate);
    }
};

void bench_binary_op_slabs(
    size_t element_count,
    uint64_t* h_data1,
    uint64_t* h_data2,
    uint64_t* h_output,
    uint64_t* d_data1,
    uint64_t* d_data2,
    uint64_t* d_output,
    size_t bitmask_size)
{
    bench_elementstuffing_wrapper<binary_op_slabs_launcher, 64>::call(
        "binary_op_slabs", element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);
}
