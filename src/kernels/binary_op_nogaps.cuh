#pragma once
#include <config.cuh>

// readin very similar to kernel_elementstuffing_nogaps
template <size_t block_size, size_t bits, size_t outbits, uint64_t (*op)(uint64_t, uint64_t)>
__global__ void kernel_binary_op_nogaps(uint64_t* lhs, uint64_t* rhs, uint64_t* out, size_t element_count)
{
    static_assert(outbits <= 64, "outbits > 64");
    size_t total_in_slab_count = (element_count * bits + 63) / 64;
    size_t total_out_slab_count = (element_count * outbits + 63) / 64;
    constexpr size_t out_elems_per_iteration = (block_size * 64 + outbits - 1) / outbits;
    // needs to adjust by up to two slabs b/c tail extends this by up to one element, which needs a whole extra slab "in front"
    constexpr size_t slab_count = (out_elems_per_iteration * bits + 63) / 64 + (block_size * 64 / outbits * outbits == block_size * 64 ? 0 : 2);
    constexpr size_t slabs_per_thread = (slab_count + block_size - 1) / block_size;
    warp_stats ws = get_warp_stats();
    __shared__ uint64_t slabs_lhs[slab_count];
    __shared__ uint64_t slabs_rhs[slab_count];
    size_t grid_stride = block_size * gridDim.x;
    for (size_t block_offset = blockIdx.x * block_size; block_offset < total_out_slab_count; block_offset += grid_stride) {
        size_t base_out_slab = block_offset;
        size_t base_in_slab = ((base_out_slab * 64) / outbits * bits) / 64; // readin follows from writeout location, determine by bits used
        __syncthreads();
        // same readin as unary nogaps kernel
        for (size_t i = 0; i < slabs_per_thread; i++) {
            size_t slab_idx = ws.base * slabs_per_thread + i * WARP_SIZE + ws.offset;
            size_t data_idx = base_in_slab + slab_idx;
            if (slab_idx < slab_count && data_idx < total_in_slab_count) {
                slabs_lhs[slab_idx] = lhs[data_idx];
                slabs_rhs[slab_idx] = rhs[data_idx];
            }
        }
        __syncthreads();

        size_t out_index = threadIdx.x;
        size_t out_offset = 0; // thread always write *a* slab, it may start with a new output elem, or be the tail of some previously written elem
        size_t in_elem = ((base_out_slab + out_index) * 64) / outbits;
        size_t in_index = (in_elem * bits) / 64 - base_in_slab;
        size_t in_offset = (in_elem * bits) % 64;
        size_t sum_offset =
            ((base_out_slab + out_index) * 64) % outbits; // offset bits for the previous element which may now be reaching into this threads slab
        bool is_tail = (sum_offset > 0);
        if (base_out_slab + out_index >= total_out_slab_count) continue; // skip if slab out of bounds

        uint64_t thread_slab = 0;
        while (true) {
            if (base_in_slab + in_index >= total_in_slab_count) break; // stop if next elem out of bounds
            uint64_t elem_lhs, elem_rhs;
            size_t remaining_bits = (64 - in_offset); // remaining unprocessed bits in the input slab
            // same two cases for element retrieval as kernel_elementstuffing_nogaps, all in one slab vs. spread over two slabs
            if (remaining_bits < bits) {
                elem_lhs = (slabs_lhs[in_index] >> in_offset) & gen_bitmask(remaining_bits);
                elem_rhs = (slabs_rhs[in_index] >> in_offset) & gen_bitmask(remaining_bits);
                in_index++;
                in_offset = bits - remaining_bits;
                elem_lhs |= (slabs_lhs[in_index] & gen_bitmask(in_offset)) << remaining_bits;
                elem_rhs |= (slabs_rhs[in_index] & gen_bitmask(in_offset)) << remaining_bits;
            }
            else {
                elem_lhs = (slabs_lhs[in_index] >> in_offset) & gen_bitmask(bits);
                elem_rhs = (slabs_rhs[in_index] >> in_offset) & gen_bitmask(bits);
                in_offset += bits;
                if (in_offset == 64) {
                    in_offset = 0;
                    in_index++;
                }
            }
            uint64_t elem_sum = op(elem_lhs, elem_rhs);
            size_t remaining_out_bits = 64 - out_offset; // remaining bits for writing in the output slab
            if (is_tail) {
                elem_sum >>= sum_offset; // only read the tail end of the elem data if we are the tail
            }
            // two writeout cases, like the input
            if (out_offset <= 64 - outbits) {
                // elem fits into the output slab
                thread_slab |= elem_sum << out_offset;
                out_offset +=
                    is_tail ? (outbits - sum_offset) : outbits; // if this was a tail elem, it does fit, but wrote less bits, inc offset accordingly
            }
            else {
                // elem exceeds beyond the output slab, cut off bits we can't fit into this slab
                thread_slab |= (elem_sum & gen_bitmask(remaining_out_bits)) << out_offset;
                out_offset += remaining_out_bits;
            }
            if (out_offset >= 64) break;
            is_tail = false;
        }
        out[base_out_slab + out_index] = thread_slab;
    }
}
