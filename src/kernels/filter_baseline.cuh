#pragma once
#include <config.cuh>

// 32 elements horizontal array: read-in access pattern like this (numbers are warp tids, rows are consecutive 32 elements)
// 0  1 .. 31
// 0  1 .. 31
// ..
// 0  1 .. 31
template <size_t block_size, size_t bits, bool (*predicate)(uint64_t)>
__global__ void kernel_filter_baseline(uint64_t* data, size_t element_count, uint64_t* bitmask)
{
    __shared__ uint32_t results[block_size]; // collect one writeout 32bit value for each thread, pattern not final (see above)
    size_t tid = threadIdx.x;
    size_t groupOffset = (tid / THREADS_PER_WARP) * THREADS_PER_WARP;
    size_t gid = tid % THREADS_PER_WARP;
    size_t pos = blockIdx.x * blockDim.x * THREADS_PER_WARP;
    size_t stride = gridDim.x * block_size * THREADS_PER_WARP;
    for (; pos < element_count; pos += stride) {
        results[tid] = 0;
        for (int i = 0; i < THREADS_PER_WARP; i++) {
            size_t inIdx = pos + (groupOffset + i) * THREADS_PER_WARP + gid; // warp accesses consecutive memory, all at once
            if (inIdx >= element_count) break;
            uint64_t inVal = data[inIdx] & (((uint64_t)~0) >> (64 - bits));
            results[tid] |= predicate(inVal) ? 1 << i : 0;
        }
        __syncwarp();
        uint32_t res = 0; // 32bits for writeout of this thread
        for (int i = 0; i < THREADS_PER_WARP; i++) {
            res |= ((results[groupOffset + i] >> gid) & 0x1) << i; // extract correct bits from shared memory, smem read broadcasts
        }
        size_t outIdx = pos / THREADS_PER_WARP + tid; // writeout consecutive memory, all at once per warp
        if (outIdx * THREADS_PER_WARP < element_count) {
            ((uint32_t*)bitmask)[outIdx] = res;
        }
    }
}
