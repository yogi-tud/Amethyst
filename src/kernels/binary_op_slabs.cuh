#pragma once
#include <config.cuh>

// readin same as kernel_elementstuffing_slabs but read from two input arrays into two smem slab regions
// still writeout oriented striding, but accomodating full length output (size of outbits per elem)
// writeout still very similar to that of kernel_elementstuffing
template <size_t block_size, size_t bits, size_t outbits, uint64_t (*op)(uint64_t, uint64_t)>
__global__ void kernel_binary_op_slabs(uint64_t* lhs, uint64_t* rhs, uint64_t* out, size_t element_count)
{
    static_assert(outbits <= 64, "outbits > 64");
    // calculate some fitting properties for the input vs output to help in the writeout
    constexpr size_t elems_per_slab_in = 64 / bits;
    constexpr size_t elems_per_slab_out = 64 / outbits;
    constexpr size_t livespace_in = bits * elems_per_slab_in;
    constexpr size_t livespace_out = outbits * elems_per_slab_out;
    size_t total_in_slab_count = (element_count + elems_per_slab_in - 1) / elems_per_slab_in;
    size_t total_out_slab_count = (element_count + elems_per_slab_out - 1) / elems_per_slab_out;
    constexpr size_t out_elems_per_iteration = block_size * elems_per_slab_out;
    // same readin as unary version, but for two input at the same time
    constexpr size_t readin_slab_count =
        (out_elems_per_iteration + elems_per_slab_in - 1) / elems_per_slab_in + ((elems_per_slab_out * 1024) % elems_per_slab_in != 0 ? 1 : 0);
    constexpr size_t readin_slabs_per_thread = (readin_slab_count + block_size - 1) / block_size;
    warp_stats ws = get_warp_stats();
    __shared__ uint64_t slabs_lhs[readin_slab_count];
    __shared__ uint64_t slabs_rhs[readin_slab_count];
    size_t grid_stride = block_size * gridDim.x;
    for (size_t block_offset = blockIdx.x * block_size; block_offset < total_out_slab_count; block_offset += grid_stride) {
        size_t base_out_slab = block_offset;
        size_t base_in_slab = (base_out_slab * elems_per_slab_out) / elems_per_slab_in;
        __syncthreads();
        for (size_t i = 0; i < readin_slabs_per_thread; i++) {
            size_t slab_idx = ws.base * readin_slabs_per_thread + i * WARP_SIZE + ws.offset;
            size_t data_idx = base_in_slab + slab_idx;
            if (slab_idx < readin_slab_count && data_idx < total_in_slab_count) {
                slabs_lhs[slab_idx] = lhs[data_idx];
                slabs_rhs[slab_idx] = rhs[data_idx];
            }
        }
        __syncthreads();

        size_t out_slab_idx = threadIdx.x; // threads write out consecutive slabs (at once)
        // idx of elem, idx of slab in smem and offset in slab to start working from for this thread
        size_t in_elem_idx = (base_out_slab + out_slab_idx) * elems_per_slab_out;
        size_t in_slab_idx = in_elem_idx / elems_per_slab_in - base_in_slab;
        size_t in_slab_offset = (in_elem_idx % elems_per_slab_in) * bits;
        if (base_out_slab + out_slab_idx >= total_out_slab_count) continue; // skip if whole slab out of bounds

        uint64_t thread_slab = 0;
        for (size_t out_slab_offset = 0; out_slab_offset < livespace_out; out_slab_offset += outbits) {
            if (base_in_slab + in_slab_idx >= total_in_slab_count) break; // stop if next elem to add to slab out of bounds
            uint64_t elem_lhs, elem_rhs;
            // using slabs with deadspace at the back -> easy element retrieval from smem, no overlaps
            elem_lhs = (slabs_lhs[in_slab_idx] >> in_slab_offset) & gen_bitmask(bits);
            elem_rhs = (slabs_rhs[in_slab_idx] >> in_slab_offset) & gen_bitmask(bits);
            // adjust input idx and offset for next elem
            in_slab_offset += bits;
            if (in_slab_offset == livespace_in) {
                in_slab_offset = 0;
                in_slab_idx++;
            }
            // append processed elem to thread output slab
            uint64_t elem_sum = op(elem_lhs, elem_rhs) & gen_bitmask(outbits);
            thread_slab |= elem_sum << out_slab_offset;
        }
        out[base_out_slab + out_slab_idx] = thread_slab;
    }
}
