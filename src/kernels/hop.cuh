#pragma once
#include <config.cuh>

// same as kernel_elementstuffing_nogaps with minor corrections for the collection/writeout
// instead of collecting to a thread mask, threads collect values into the hop aggregator (agg) with a varying initial value (0 for OR, 1 for AND..)
// instead of writing out to a bitmask, threads use a global atomic op (automatic reduction on hardware) to collect their value into the global agg
template <size_t block_size, size_t bits, uint64_t hop_init, uint64_t (*op)(uint64_t, uint64_t), uint64_t (*atomic_op)(uint64_t*, uint64_t)>
__global__ void kernel_hop_nogaps(uint64_t* data, size_t element_count, uint64_t* out)
{
    size_t total_slab_count = (element_count * bits + 63) / 64;
    constexpr size_t slab_count = (block_size * ELEMS_PER_THREAD * bits + 63) / 64 + (1024 / bits * bits == 1024 ? 0 : 1);
    constexpr size_t slabs_per_thread = (slab_count + block_size - 1) / block_size;
    warp_stats ws = get_warp_stats();
    __shared__ uint64_t slabs[slab_count];
    size_t elems_per_iteration = ELEMS_PER_THREAD * block_size;
    size_t grid_stride = elems_per_iteration * gridDim.x;
    uint64_t agg = hop_init;
    for (size_t block_offset = blockIdx.x * elems_per_iteration; block_offset < element_count; block_offset += grid_stride) {
        size_t base_slab = (bits * block_offset) / 64;
        __syncthreads();
        for (size_t i = 0; i < slabs_per_thread; i++) {
            size_t slab_idx = ws.base * slabs_per_thread + i * WARP_SIZE + ws.offset;
            size_t data_idx = base_slab + slab_idx;
            if (slab_idx < slab_count && data_idx < total_slab_count) {
                slabs[slab_idx] = data[data_idx];
            }
        }
        __syncthreads();

        size_t elem_index = threadIdx.x * ELEMS_PER_THREAD;
        if (block_offset + elem_index >= element_count) continue;
        size_t slab = (elem_index * bits) / 64;
        size_t offset = (elem_index * bits) % 64;

        for (size_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if (block_offset + elem_index + i >= element_count) break;
            uint64_t elem;
            size_t remaining_bits = (64 - offset);
            if (remaining_bits < bits) {
                elem = ((slabs[slab] >> offset) & gen_bitmask(remaining_bits));
                slab++;
                offset = bits - remaining_bits;
                elem |= (slabs[slab] & gen_bitmask(offset)) << remaining_bits;
            }
            else {
                elem = (slabs[slab] >> offset) & gen_bitmask(bits);
                offset += bits;
                if (offset == 64) {
                    offset = 0;
                    slab++;
                }
            }
            agg = op(agg, elem);
        }
    }
    atomic_op(out, agg);
}

// simple grid striding kernel, all threads of a warp load consecutive slabs at once and process the contained elems into a logal aggregator (agg) for
// later writeout to the global agg by cuda device atomic
template <size_t block_size, size_t bits, uint64_t hop_init, uint64_t (*op)(uint64_t, uint64_t), uint64_t (*atomic_op)(uint64_t*, uint64_t)>
__global__ void kernel_hop_slabs(uint64_t* data, size_t element_count, uint64_t* out)
{
    constexpr size_t elems_per_slab = 64 / bits;
    size_t total_slab_count = (element_count + elems_per_slab - 1) / elems_per_slab;
    uint64_t agg = hop_init;
    for (size_t idx = (size_t)threadIdx.x + (size_t)blockIdx.x * (size_t)blockDim.x; idx < total_slab_count; idx += blockDim.x * gridDim.x) {
        uint64_t slab = data[idx];
        for (size_t i = 0; i < elems_per_slab; i++) {
            if (idx * elems_per_slab + i >= element_count) break;
            uint64_t val = (slab >> (i * bits)) & gen_bitmask(bits);
            agg = op(agg, val);
        }
    }
    atomic_op(out, agg);
}

template <size_t block_size, int bits, uint64_t hop_init, uint64_t (*op)(uint64_t, uint64_t), uint64_t (*atomic_op)(uint64_t*, uint64_t)>
struct slabs_call {
    static void call(size_t block_count, uint64_t* d_data1, size_t element_count, uint64_t* d_output)
    {
        kernel_hop_slabs<block_size, bits, hop_init, op, atomic_op><<<block_count, block_size>>>(d_data1, element_count, d_output);
    }
};
template <size_t block_size, int bits, uint64_t hop_init, uint64_t (*op)(uint64_t, uint64_t), uint64_t (*atomic_op)(uint64_t*, uint64_t)>
struct nogaps_call {
    static void call(size_t block_count, uint64_t* d_data1, size_t element_count, uint64_t* d_output)
    {
        kernel_hop_nogaps<block_size, bits, hop_init, op, atomic_op><<<block_count, block_size>>>(d_data1, element_count, d_output);
    }
};
