#pragma once
#include <config.cuh>

template <size_t block_size, size_t bits, bool (*predicate)(uint64_t)>
__global__ void kernel_filter_slabs(uint64_t* data, size_t element_count, uint64_t* bitmask)
{
    constexpr size_t elems_per_slab = 64 / bits;
    size_t total_slab_count = (element_count + elems_per_slab - 1) / elems_per_slab;
    // slab count req'd in smem for a block with 32 elems per thread (i.e. one 32bit writeout mask elem per thread) per writeout cycle
    // +1 slab if the fit for one 1024bit mask part is not perfect, b/c for a block_offset not starting on the beginning of a base_slab the readin
    // requires one more slab to compensate for the few useless bits of the first slab
    constexpr size_t slab_count =
        (block_size * ELEMS_PER_THREAD + elems_per_slab - 1) / elems_per_slab + (1024 / elems_per_slab * elems_per_slab == 1024 ? 0 : 1);
    constexpr size_t slabs_per_thread = (slab_count + block_size - 1) / block_size;
    warp_stats ws = get_warp_stats();
    __shared__ uint64_t slabs[slab_count];
    size_t elems_per_iteration = ELEMS_PER_THREAD * block_size;
    size_t grid_stride = elems_per_iteration * gridDim.x;
    // writeout oriented stride, i.e. 32 threads will writeout consecutive 32bit bitmask elems all at once
    for (size_t block_offset = blockIdx.x * elems_per_iteration; block_offset < element_count; block_offset += grid_stride) {
        size_t base_slab = block_offset / elems_per_slab; // round block_offset (block starting element) to slab
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
        size_t slab_index = (block_offset + elem_index) / elems_per_slab -
                            base_slab; // slab for this thread to start reading its inputs from (indexing into smem slabs)
        size_t slab_offset = (block_offset + elem_index) % elems_per_slab; // offset in elems within that slab

        // every thread builds its 32bits of writeout by reading slabs from smem
        uint32_t thread_output = 0;
        for (size_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if (block_offset + elem_index + i >= element_count) break; // stop if next elem added to thread mask is out of bounds
            // getting the value is always easy by slab idx and offset b/c all values are completely within their containing slabs
            uint64_t value = (uint64_t)((slabs[slab_index] >> (slab_offset * bits)) & gen_bitmask(bits));
            // add pred(elem) to thread bitmask
            thread_output |= ((uint32_t)predicate(value)) << i;
            slab_offset++;
            // if end of slab inc slab idx and reset offset
            if (slab_offset == elems_per_slab) {
                slab_index++;
                slab_offset = 0;
            }
        }
        ((uint32_t*)bitmask)[block_offset / WARP_SIZE + threadIdx.x] = thread_output;
    }
}

// similar to kernel_elementstuffing but does not require syncthreads, slabs regions in smem used per warp now do not overlap (small overhead for a
// few slabs loaded multiple times)
template <size_t block_size, size_t bits, bool (*predicate)(uint64_t)>
__global__ void kernel_filter_slabs_bywarp(uint64_t* data, size_t element_count, uint64_t* bitmask)
{
    constexpr size_t elems_per_slab = 64 / bits;
    warp_stats ws = get_warp_stats();
    constexpr size_t slabs_per_warp =
        (WARP_SIZE * ELEMS_PER_THREAD + elems_per_slab - 1) / elems_per_slab + (1024 / elems_per_slab * elems_per_slab == 1024 ? 0 : 1);
    constexpr size_t slabs_per_thread = (slabs_per_warp + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ uint64_t slabs[slabs_per_warp * block_size / WARP_SIZE];
    size_t elems_per_iteration = ELEMS_PER_THREAD * block_size;
    size_t grid_stride = elems_per_iteration * gridDim.x;
    for (size_t block_offset = blockIdx.x * elems_per_iteration; block_offset < element_count; block_offset += grid_stride) {
        size_t warp_offset = block_offset + ws.base * WARP_SIZE; // warp offset (elem idx) different per warp
        size_t base_slab = warp_offset / elems_per_slab; // round to base slab (slab idx) for the warp
        __syncwarp();
        for (size_t i = 0; i < slabs_per_thread; i++) {
            // slab_idx indexes into this warp, for access into smem slabs: offset by regions of slabs other warps are responsible for
            size_t slab_idx = i * WARP_SIZE + ws.offset;
            size_t data_idx = base_slab + i * WARP_SIZE + ws.offset;
            if (slab_idx < slabs_per_warp && data_idx < element_count) {
                slabs[ws.idx * slabs_per_warp + slab_idx] = data[data_idx];
            }
        }
        __syncwarp();
        size_t elem_index = ws.offset * ELEMS_PER_THREAD; // elem idx for thread to start at, now determined on per warp basis
        if (warp_offset + elem_index >= element_count) continue;
        size_t slab_index = (warp_offset + elem_index) / elems_per_slab - base_slab;
        size_t slab_offset = (warp_offset + elem_index) % elems_per_slab;
        uint32_t thread_output = 0;
        for (size_t i = 0; i < ELEMS_PER_THREAD; i++) {
            if (warp_offset + elem_index + i >= element_count) break;
            // retrieval of value has to adjust for regions of slabs handled by other warps, then idx by slab_index
            uint64_t value = (uint64_t)((slabs[ws.idx * slabs_per_warp + slab_index] >> (slab_offset * bits)) & gen_bitmask(bits));
            thread_output |= ((uint32_t)predicate(value)) << i;
            slab_offset++;
            if (slab_offset == elems_per_slab) {
                slab_index++;
                slab_offset = 0;
            }
        }
        ((uint32_t*)bitmask)[block_offset / WARP_SIZE + threadIdx.x] = thread_output;
    }
}
