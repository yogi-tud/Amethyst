#pragma once
#include <config.cuh>

template <int bits, size_t CHUNK_LENGTH>
__global__ void kernel_compresstore_slabs_chunk_popcount(uint8_t* mask, size_t element_count, size_t* chunk_popcount)
{
    // we perform the popcount on 32 bit elements, so for simplicity we expect CHUNK_LENGTH to be a multiple of 32
    static_assert(CHUNK_LENGTH % 32 == 0);
    size_t chunk_count = ceildiv(element_count, CHUNK_LENGTH);
    size_t bytes_to_process = CHUNK_LENGTH / 8;
    size_t remainder_bits_to_process = 8;
    for (size_t chunk_index = (blockIdx.x * blockDim.x) + threadIdx.x; chunk_index < chunk_count; chunk_index += blockDim.x * gridDim.x) {
        uint32_t chunk_mask_offset = chunk_index * (CHUNK_LENGTH / 8); // index of first mask byte for this chunk
        if (chunk_index + 1 == chunk_count) {
            bytes_to_process = ceildiv(element_count - chunk_index * CHUNK_LENGTH, 8);
            remainder_bits_to_process = element_count % 8 ? element_count % 8 : 8;
        }
        // assuming CHUNK_LENGTH to be multiple of 32
        uint32_t popcount = 0;
        int i = 0;
        for (; i < bytes_to_process / 4 * 4; i += 4) {
            popcount += __popc(*reinterpret_cast<uint32_t*>(mask + chunk_mask_offset + i));
        }
        if (i < bytes_to_process / 2 * 2) {
            popcount += __popc(*reinterpret_cast<uint16_t*>(mask + chunk_mask_offset + i));
            i += 2;
        }
        if (i < bytes_to_process) {
            popcount += __popc(*reinterpret_cast<uint8_t*>(mask + chunk_mask_offset + i) & gen_bitmask(remainder_bits_to_process));
        }
        chunk_popcount[chunk_index] = popcount;
    }
}

template <int BITS, size_t CHUNK_LENGTH, uint32_t BLOCK_DIM>
__global__ void kernel_compresstore_slabs_write(
    uint64_t* input, uint64_t* output, uint8_t* mask, size_t* chunk_popcount, size_t* chunk_prefix_sum, size_t element_count)
{
    constexpr size_t ELEMS_PER_SLAB = 64 / BITS;
    constexpr uint32_t WARPS_PER_BLOCK = BLOCK_DIM / WARP_SIZE;
    constexpr uint32_t ELEMS_PER_MASK_LOAD = WARP_SIZE * WARP_SIZE;
    constexpr uint32_t ELEMS_PER_FLUSH = WARP_SIZE * ELEMS_PER_SLAB;
    //we want to be able to load aligned, 32 bit mask strips with each thread
    static_assert(CHUNK_LENGTH % 32 == 0);

    const uint32_t offset_in_warp = threadIdx.x % WARP_SIZE;
    const uint32_t warp_index = threadIdx.x / WARP_SIZE;
    const size_t warp_base_tid = threadIdx.x - offset_in_warp;

    const uint32_t chunk_count = ceildiv(element_count, CHUNK_LENGTH);

    __shared__ uint32_t mask_bits[BLOCK_DIM];
    __shared__ uint32_t selected_input_indices[WARPS_PER_BLOCK * ELEMS_PER_FLUSH];
    __shared__ uint32_t warp_max_out_indices[WARPS_PER_BLOCK];

    // each warp must start at a chunk boundary to be able
    // to aquire a output start index from the prefix sum array
    const size_t elements_per_warp = ceildiv(chunk_count, gridDim.x) * CHUNK_LENGTH;

    size_t warp_input_idx = (blockIdx.x * WARPS_PER_BLOCK + warp_index) * elements_per_warp;
    if (warp_input_idx >= element_count) return;

    size_t warp_output_idx = chunk_prefix_sum[warp_input_idx / CHUNK_LENGTH];
    size_t warp_input_idx_end = warp_input_idx + elements_per_warp;
    if (warp_input_idx_end > element_count) warp_input_idx_end = element_count;
    size_t warp_mask_idx_end = ceildiv(warp_input_idx_end, 8);

    size_t overhang_slab_elems_present = warp_output_idx % ELEMS_PER_SLAB;
    size_t overhang_slab_elems_missing = overhang_slab_elems_present ? ELEMS_PER_SLAB - overhang_slab_elems_present : 0;
    uint32_t elements_aquired = overhang_slab_elems_present;
    if (offset_in_warp == 0) {
        //printf("foo %i %i: warp_output_idx %lu, eps: %lu, overhang present: %lu\n", blockIdx.x, threadIdx.x, warp_output_idx, ELEMS_PER_SLAB, overhang_slab_elems_present);
    }
    warp_output_idx -= overhang_slab_elems_present;
    while (warp_input_idx < warp_input_idx_end) {
        // check if we can skip this stride (all chunks have popc==0)
        bool empty_stride = true;
        for (uint32_t cid = warp_input_idx / CHUNK_LENGTH; cid <= (warp_input_idx + ELEMS_PER_FLUSH - 1) / CHUNK_LENGTH; cid++) {
            if (chunk_popcount[cid] != 0) {
                empty_stride = false;
                break;
            }
        }
        if (empty_stride) {
            warp_input_idx += ELEMS_PER_MASK_LOAD;
            continue;
        }
        // load mask
        uint32_t mask_idx = warp_input_idx / 8 + offset_in_warp * 4;
        uchar4 ucx = {0, 0, 0, 0};
        if (mask_idx + 4 > warp_mask_idx_end) {
            switch ((ssize_t)warp_mask_idx_end - (ssize_t)mask_idx) {
                case 3: ucx.z = *(mask + mask_idx + 2); // fallthrough
                case 2: ucx.y = *(mask + mask_idx + 1); // fallthrough
                case 1: ucx.x = *(mask + mask_idx) & gen_bitmask(element_count % 8 ? element_count % 8 : 8); // fallthrough
                default: break;
            }
        }
        else {
            ucx = *reinterpret_cast<uchar4*>(mask + mask_idx);
        }
        mask_bits[threadIdx.x] = *reinterpret_cast<uint32_t*>(&ucx);
        __syncwarp();
        // collect output indices
        uint32_t input_index = warp_input_idx + offset_in_warp;
        for (int i = 0; i < WARP_SIZE; i++) {
            uint32_t mask = mask_bits[warp_base_tid + i];
            uint32_t mask_popc_before = (offset_in_warp == 0) ? 0 : __popc(mask << (WARP_SIZE - offset_in_warp));
            bool v = (mask >> offset_in_warp) & 0b1;
            if (offset_in_warp == WARP_SIZE - 1) {
                warp_max_out_indices[warp_index] = mask_popc_before + v;
            }
            __syncwarp();
            uint32_t warp_max_out_idx = warp_max_out_indices[warp_index];
            if (elements_aquired + warp_max_out_idx >= ELEMS_PER_FLUSH) {
                uint32_t nth_out_index = elements_aquired + mask_popc_before;
                if (v && nth_out_index < ELEMS_PER_FLUSH) {
                    selected_input_indices[ELEMS_PER_SLAB * warp_base_tid + nth_out_index] = input_index;
                }
                __syncwarp();
                uint64_t slab = 0;
                bool handle_overhang = (offset_in_warp == 0 && overhang_slab_elems_missing);
                size_t e = handle_overhang ? overhang_slab_elems_present : 0;
                for(; e < ELEMS_PER_SLAB; e++) {
                    size_t in_index = selected_input_indices[ELEMS_PER_SLAB * (warp_base_tid + offset_in_warp) + e];
                    size_t in_slab_offset = in_index % ELEMS_PER_SLAB;
                    uint64_t input_slab = input[in_index / ELEMS_PER_SLAB];
                    slab |= ((input_slab >> in_slab_offset * BITS) & gen_bitmask(BITS)) << (BITS * e);
                    /*printf(
                        "handling chunked elements: (%i %i / %li): [%li(%li ~ %li)] = 0x%lx -> [%li(%li ~ %li)]\n",
                        blockIdx.x, threadIdx.x, e,
                        in_index, in_index / ELEMS_PER_SLAB, in_slab_offset * BITS,
                        ((input_slab >> (in_slab_offset * BITS)) & gen_bitmask(BITS)),
                        warp_output_idx + offset_in_warp * ELEMS_PER_SLAB + e,
                        warp_output_idx / ELEMS_PER_SLAB + offset_in_warp,
                        e * BITS
                    );*/
                }
                uint64_t* out_ptr = &output[warp_output_idx / ELEMS_PER_SLAB + offset_in_warp];
                if (handle_overhang) {
                    atomicAnd((unsigned long long*)out_ptr, gen_bitmask(overhang_slab_elems_present * BITS));
                    atomicOr((unsigned long long*)out_ptr, slab);
                    overhang_slab_elems_missing = 0;
                    overhang_slab_elems_present = 0;
                }
                else{
                    *out_ptr = slab;
                }
                __syncwarp();
                if (v && nth_out_index >= ELEMS_PER_FLUSH) {
                    //printf("selected for output index %lu: input index %u\n", warp_output_idx + ELEMS_PER_SLAB * warp_base_tid + elements_aquired + mask_popc_before, input_index);
                    selected_input_indices[ELEMS_PER_SLAB * warp_base_tid + nth_out_index - ELEMS_PER_FLUSH] = input_index;
                }
                elements_aquired += warp_max_out_idx;
                elements_aquired -= ELEMS_PER_FLUSH;
                warp_output_idx += ELEMS_PER_FLUSH;
            }
            else {
                if (v) {
                    //printf("selected for output index %lu: input index %u\n", warp_output_idx + ELEMS_PER_SLAB * warp_base_tid + elements_aquired + mask_popc_before, input_index);
                    selected_input_indices[ELEMS_PER_SLAB * warp_base_tid + elements_aquired + mask_popc_before] = input_index;
                }
                elements_aquired += warp_max_out_idx;
            }
            input_index += WARP_SIZE;
        }
        warp_input_idx += ELEMS_PER_MASK_LOAD;
    }
    __syncwarp();
    if (offset_in_warp * ELEMS_PER_SLAB < elements_aquired) {
        size_t elem_count = elements_aquired - offset_in_warp * ELEMS_PER_SLAB;
        if (elem_count > ELEMS_PER_SLAB) elem_count = ELEMS_PER_SLAB;
        uint64_t slab = 0;
        bool handle_overhang = (offset_in_warp == 0 && overhang_slab_elems_present);
        bool handle_underfill = elem_count != ELEMS_PER_SLAB;
        size_t e = handle_overhang ? overhang_slab_elems_present : 0;
        for(; e < elem_count; e++) {
            size_t in_index = selected_input_indices[ELEMS_PER_SLAB * (warp_base_tid + offset_in_warp) + e];
            size_t in_slab_offset = in_index % ELEMS_PER_SLAB;
            uint64_t input_slab = input[in_index / ELEMS_PER_SLAB];
            slab |= ((input_slab >> (in_slab_offset * BITS)) & gen_bitmask(BITS)) << (BITS * e);
            /*printf(
                "handling trailing elements: (%i %i / %li): (ec: %lu, ea: %lu, op: %lu, oiw: %lu) [%li(%li ~ %li)] = 0x%lx -> [%li(%li ~ %li)]\n",
                blockIdx.x, threadIdx.x, e,
                elem_count, elements_aquired,  overhang_slab_elems_present, offset_in_warp * ELEMS_PER_SLAB,
                in_index, in_index / ELEMS_PER_SLAB, in_slab_offset * BITS,
                ((input_slab >> (in_slab_offset * BITS)) & gen_bitmask(BITS)),
                warp_output_idx + offset_in_warp * ELEMS_PER_SLAB + e,
                warp_output_idx / ELEMS_PER_SLAB + offset_in_warp,
                e * BITS
            );*/
        }
        uint64_t* out_ptr = &output[warp_output_idx / ELEMS_PER_SLAB + offset_in_warp];
        if(handle_underfill || handle_overhang) {
            //printf("AND: oi: %lu, ec: %lu, p: %lu, mask: %lx\n",  warp_output_idx + offset_in_warp * ELEMS_PER_SLAB, elem_count, overhang_slab_elems_present, ~(gen_bitmask(BITS * (elem_count - overhang_slab_elems_present)) << (BITS * overhang_slab_elems_present)));
            atomicAnd(
                (unsigned long long*)out_ptr,
                ~(gen_bitmask(BITS * (elem_count - overhang_slab_elems_present)) << (BITS * overhang_slab_elems_present))
            );
            atomicOr((unsigned long long*)out_ptr, slab);
        }
        else {
            *out_ptr = slab;
        }
    }
}
