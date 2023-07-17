#pragma once
#include <config.cuh>

template <size_t block_size, size_t bits, bool (*predicate)(uint64_t)>
__global__ void kernel_filter_nogaps(uint64_t* data, size_t element_count, uint64_t* bitmask)
{
    size_t total_slab_count = (element_count * bits + 63) / 64;
    // slab count req'd in smem for a block with 32 elems per thread (i.e. one 32bit writeout mask elem per thread) per writeout cycle
    // +1 slab if the fit for one 1024bit mask part is not perfect, b/c for a block_offset not starting on the beginning of a base_slab the readin
    // requires one more slab to compensate for the few useless bits of the first slab
    constexpr size_t slab_count = (block_size * ELEMS_PER_THREAD * bits + 63) / 64 + (1024 / bits * bits == 1024 ? 0 : 1);
    constexpr size_t slabs_per_thread = (slab_count + block_size - 1) / block_size;
    warp_stats ws = get_warp_stats();
    __shared__ uint64_t slabs[slab_count];
    size_t elems_per_iteration = ELEMS_PER_THREAD * block_size;
    size_t grid_stride = elems_per_iteration * gridDim.x;
    // writeout oriented stride, i.e. 32 threads will writeout consecutive 32bit bitmask elems all at once
    for (size_t block_offset = blockIdx.x * elems_per_iteration; block_offset < element_count; block_offset += grid_stride) {
        size_t base_slab = (bits * block_offset) / 64; // round block_offset (block starting element) to slab
        __syncthreads(); // catch writing stragglers from last loop
        for (size_t i = 0; i < slabs_per_thread; i++) {
            // readin 32 consecutive slabs all at once for a warp: uses this blocks base slab and offsets by the amount of slabs previous warps have
            // read already + offset to work this warp is doing + idx in warp
            size_t slab_idx = ws.base * slabs_per_thread + i * WARP_SIZE + ws.offset;
            size_t data_idx = base_slab + slab_idx;
            if (slab_idx < slab_count && data_idx < total_slab_count) {
                slabs[slab_idx] = data[data_idx];
            }
        }
        __syncthreads(); // catch reading stragglers

        size_t elem_index = threadIdx.x * ELEMS_PER_THREAD; // elem idx for thread to start at, determined from idx in block
        if (block_offset + elem_index >= element_count) continue; // skip if whole 32bit elem is out of bounds
        size_t slab = (elem_index * bits) / 64; // slab for this thread to start reading its inputs from (indexing into smem slabs)
        size_t offset = (elem_index * bits) % 64; // offset in bits within that slab to start reading from

        // every thread builds its 32bits of writeout by reading slabs from smem
        uint32_t thread_output = 0;
        for (size_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if (block_offset + elem_index + i >= element_count) break; // stop if next elem added to thread mask is out of bounds
            uint64_t elem;
            size_t remaining_bits = (64 - offset);
            if (remaining_bits < bits) {
                // non-trivial case: elem is stretched over two slabs, extract remaining bits from first slab and rest from second, inc slab and
                // correct offset to account for already read bits
                elem = ((slabs[slab] >> offset) & gen_bitmask(remaining_bits));
                slab++;
                offset = bits - remaining_bits;
                elem |= (slabs[slab] & gen_bitmask(offset)) << remaining_bits;
            }
            else {
                // easy case: elem is completely inside of the current slab, extract it and increment the offset
                elem = (slabs[slab] >> offset) & gen_bitmask(bits);
                offset += bits;
                if (offset == 64) {
                    // inc slab idx if trivial
                    offset = 0;
                    slab++;
                }
            }
            thread_output |= ((uint32_t)predicate(elem)) << i; // append (OR) the pred(elem) value into the thread mask
        }
        ((uint32_t*)bitmask)[block_offset / WARP_SIZE + threadIdx.x] = thread_output;
    }
}
