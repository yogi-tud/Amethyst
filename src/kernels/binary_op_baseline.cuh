#pragma once
#include <config.cuh>

template <size_t bits, uint64_t (*op)(uint64_t, uint64_t)>
__global__ void kernel_binary_op_baseline(uint64_t* lhs, uint64_t* rhs, uint64_t* out, size_t element_count)
{
    // simple grid striding
    size_t pos = threadIdx.x + blockIdx.x * blockDim.x;
    size_t stride = gridDim.x * blockDim.x;
    for (; pos < element_count; pos += stride) {
        out[pos] = op(lhs[pos] & gen_bitmask(bits), rhs[pos] & gen_bitmask(bits));
    }
}
