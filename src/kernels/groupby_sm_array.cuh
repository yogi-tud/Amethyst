#pragma once
#include <config.cuh>

template <size_t block_size, size_t bits>
__global__ void kernel_groupby_sm_array_gather(
    uint64_t* keys,
    uint64_t* values,
    typename tight_type<bits>::cuda_type* aggregates_temp,
    unsigned int* occurences_temp,
    size_t element_count,
    uint64_t* reduce_key_count)
{
    typedef typename tight_type<bits>::cuda_type cuda_elem_type;
    constexpr size_t max_group_count = (size_t)1 << bits;
    constexpr size_t elems_per_slab = 64 / bits;
    size_t slab_count = (element_count + elems_per_slab - 1) / elems_per_slab;
    __shared__ uint64_t slabs_keys[block_size];
    __shared__ uint64_t slabs_values[block_size];
    __shared__ cuda_elem_type table_aggregates[max_group_count];
    __shared__ unsigned int occurence_flag[max_group_count]; // TODO bitmask
    size_t grid_stride = gridDim.x * block_size;
    for (size_t i = threadIdx.x; i < max_group_count; i += block_size) {
        table_aggregates[i] = 0;
        occurence_flag[i] = 0;
    }
    __syncthreads();
    for (size_t block_offset = blockIdx.x * block_size; block_offset < slab_count; block_offset += grid_stride) {
        __syncthreads(); // catch writing stragglers from last loop

        size_t readin_slab_index = block_offset + threadIdx.x;
        if (readin_slab_index < slab_count) {
            slabs_keys[threadIdx.x] = keys[readin_slab_index];
            slabs_values[threadIdx.x] = values[readin_slab_index];
        }

        __syncthreads(); // catch reading stragglers

        cuda_elem_type key;
        cuda_elem_type value;
        for (size_t i = threadIdx.x; i < block_size * elems_per_slab; i += block_size) {
            size_t elem_index = block_offset * elems_per_slab + i;
            if (elem_index >= element_count) break;

            size_t smem_slab_index = elem_index / elems_per_slab - block_offset;
            size_t smem_slab_offset = elem_index % elems_per_slab;

            key = (cuda_elem_type)((slabs_keys[smem_slab_index] >> (smem_slab_offset * bits)) & gen_bitmask(bits));
            value = (cuda_elem_type)((slabs_values[smem_slab_index] >> (smem_slab_offset * bits)) & gen_bitmask(bits));
            GROUP_BY_OP_ATOMIC<bits>(table_aggregates + key, value);
            atomicAdd((unsigned int*)(occurence_flag + key), 1);
        }
    }
    __syncthreads();
    for (size_t i = threadIdx.x; i < max_group_count; i += block_size) {
        if (occurence_flag[i] == 0) continue;
        atomicOr((unsigned int*)(occurences_temp + i), 1);
        GROUP_BY_OP_ATOMIC<bits>(aggregates_temp + i, table_aggregates[i]);
    }
}

template <size_t block_size, size_t bits>
__global__ void kernel_groupby_sm_array_writeout(
    typename tight_type<bits>::cuda_type* aggregates_temp,
    unsigned int* occurences_temp,
    uint64_t* keys_out,
    uint64_t* values_out,
    size_t element_count,
    uint64_t* reduce_key_count)
{
    constexpr size_t elems_per_slab = 64 / bits;
    constexpr size_t max_group_count = 1 << bits;
    for (size_t i = blockIdx.x * block_size + threadIdx.x; i < max_group_count; i += block_size * gridDim.x) {
        if (occurences_temp[i] == 0) continue;
        size_t elem_idx = atomicAdd((unsigned long long*)reduce_key_count, 1);
        size_t slab_idx = elem_idx / elems_per_slab;
        size_t offset = bits * (elem_idx % elems_per_slab);
        uint64_t value = (uint64_t)aggregates_temp[i] & gen_bitmask(bits);
        atomicOr((unsigned long long*)keys_out + slab_idx, (uint64_t)i << offset);
        atomicOr((unsigned long long*)values_out + slab_idx, value << offset);
    }
}
