#include <config.h>

#include <kernels/binary_op_baseline.cuh>
template <size_t bits, uint64_t (*op)(uint64_t, uint64_t)>
void binary_op_baseline(uint64_t* d_lhs, uint64_t* d_rhs, uint64_t* d_out, size_t element_count) {
    kernel_binary_op_baseline<bits, BINARY_OP<bits>><<<block_count, block_size>>>(d_lhs, d_rhs, d_out, element_count);
}


#include <kernels/binary_op_nogaps.cuh>
template <size_t bits, size_t outbits, uint64_t (*op)(uint64_t, uint64_t), size_t block_size>
void binary_op_nogaps(uint64_t* d_lhs, uint64_t* d_rhs, uint64_t* d_out, size_t element_count, size_t grid_size) {
  kernel_binary_op_nogaps<block_size, bits, outbits, BINARY_OP<bits>><<<grid_size, block_size>>>(d_lhs, d_rhs, d_out, element_count);
}

#include <kernels/binary_op_slabs.cuh>
template <size_t bits, size_t outbits, uint64_t (*op)(uint64_t, uint64_t), size_t block_size>
void binary_op_slabs(uint64_t* d_lhs, uint64_t* d_rhs, uint64_t* d_out, size_t element_count, size_t grid_size) {
    kernel_binary_op_slabs<block_size, bits, outbits, BINARY_OP<bits>><<<grid_size, block_size>>>(d_lhs, d_rhs, d_out, element_count);
}

#include <kernels/compressstore_baseline_cub.cuh>
#include <bit_iterator.cuh>
template <size_t bits>
size_t compressstore_baseline_cub(uint8_t* d_mask, uint64_t* d_elements, uint64_t* d_out, size_t element_count) {
    bit_iterator d_mask_iter{d_mask};
    size_t cub_intermediate_size;
    size_t* d_output_element_count;
    CUDA_TRY(cudaMalloc(&d_output_element_count, sizeof(size_t)));
    cub::DeviceSelect::Flagged(NULL, cub_intermediate_size, d_elements, d_mask_iter, d_output, d_output_element_count, element_count);
    void* cub_intermediate_storage;
    CUDA_TRY(cudaMalloc(&cub_intermediate_storage, cub_intermediate_size));
    cub::DeviceSelect::Flagged(cub_intermediate_storage, cub_intermediate_size, d_elements, d_mask_iter, d_output, d_output_element_count, element_count);
    size_t output_element_count;
    CUDA_TRY(cudaMemcpy(&output_element_count, d_output_element_count, sizeof(size_t), cudaMemcpyDeviceToHost));
    cudaFree(cub_intermediate_storage);
    cudaFree(d_output_element_count);
    return output_element_count;
}


#include <kernels/filter_baseline.cuh>
template <size_t bits, bool (*predicate)(uint64_t), size_t block_size>
void filter_baseline(uint64_t* d_elements, uint64_t* d_output_bitmask, size_t element_count, size_t grid_size) {
    kernel_filter_baseline<block_size, bits, predicate<bits>><<<block_count, block_size>>>(d_elements, element_count, d_output_bitmask);
}
