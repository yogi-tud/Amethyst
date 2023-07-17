#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "utils.cuh"
#include "fast_prng.cuh"
#include <cstring>

#include "common_code.cuh"

bool do_validation = true;

int run_count = 3;

int wantbits = 64;

size_t grid_size_min = 32;
size_t grid_size_max = 8192;
size_t block_size_min = 32;
size_t block_size_max = 1024;

#define DECLARE_AND_CALL(name)                                                                                                                       \
    void name(                                                                                                                                       \
        size_t element_count, uint64_t* h_data1, uint64_t* h_data2, uint64_t* h_output, uint64_t* d_data1, uint64_t* d_data2, uint64_t* d_output,    \
        size_t bitmask_size);                                                                                                                        \
    name(element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);

int main(int argc, char** argv)
{
    size_t element_count = 1 << 27;
    bool use_rng = true;

    int w_argc = argc - 1; // remaining arg count
    while (w_argc > 0) {
        char* w_arg = argv[argc - (w_argc--)]; // working arg
        char* n_arg = (w_argc > 0) ? argv[argc - w_argc] : NULL; // next arg
        if (strcmp(w_arg, "--nv") == 0) {
            do_validation = false;
        }
        else if (strcmp(w_arg, "-n") == 0) {
            w_argc--;
            element_count = atoi(n_arg);
            if (element_count == 0) {
                element_count = 1024;
            }
        }
        else if (strcmp(w_arg, "-N") == 0) {
            w_argc--;
            element_count = 1 << atoi(n_arg);
            if (element_count == 1) {
                element_count = 1 << 10;
            }
        }
        else if (strcmp(w_arg, "-r") == 0) {
            w_argc--;
            run_count = atoi(n_arg);
            if (run_count == 0) {
                run_count = 4;
            }
        }
        else if (strcmp(w_arg, "-b") == 0) {
            w_argc--;
            wantbits = atoi(n_arg);
            if (wantbits == 0) {
                exit(0);
            }
        }
        else if (strcmp(w_arg, "--gsm") == 0) {
            w_argc--;
            grid_size_min = atoi(n_arg);
            if (grid_size_min == 0) {
                exit(0);
            }
        }
        else if (strcmp(w_arg, "--gsM") == 0) {
            w_argc--;
            grid_size_max = atoi(n_arg);
            if (grid_size_max == 0) {
                exit(0);
            }
        }
        else if (strcmp(w_arg, "--bsm") == 0) {
            w_argc--;
            block_size_min = atoi(n_arg);
            if (block_size_min == 0) {
                exit(0);
            }
        }
        else if (strcmp(w_arg, "--bsM") == 0) {
            w_argc--;
            block_size_max = atoi(n_arg);
            if (block_size_max == 0) {
                exit(0);
            }
        }
        else if (strcmp(w_arg, "--ff") == 0) {
            use_rng = false;
        }
        else {
            printf("ignoring unknown argument: \"%s\"\n", w_arg);
        }
    }

    size_t data_size = element_count * sizeof(uint64_t);
    size_t bitmask_size = ((element_count + 64 - 1) / 64) * sizeof(uint64_t);
    size_t output_data_size = data_size * 4; // group by need 2 outputs and reduce by key need 2 more

    // generate data (compressed)
    uint64_t* h_data1 = (uint64_t*)malloc(data_size);
    uint64_t* h_data2 = (uint64_t*)malloc(data_size);
    uint64_t* h_output = (uint64_t*)malloc(output_data_size);
    fast_prng rng(42);
    if (use_rng) {
        for (size_t i = 0; i < data_size / sizeof(uint64_t); i++) {
            h_data1[i] = ((uint64_t)rng.rand()) << 32 | (uint64_t)rng.rand();
            h_data2[i] = ((uint64_t)rng.rand()) << 32 | (uint64_t)rng.rand();
        }
    }
    else {
        memset(h_data1, 0xFF, data_size);
        memset(h_data2, 0xFF, data_size);
    }

    // memcpy to gpu
    uint64_t* d_data1;
    uint64_t* d_data2;
    uint64_t* d_output;
    CUDA_TRY(cudaMalloc(&d_data1, data_size));
    CUDA_TRY(cudaMalloc(&d_data2, data_size));
    CUDA_TRY(cudaMalloc(&d_output, output_data_size));
    CUDA_TRY(cudaMemcpy(d_data1, h_data1, data_size, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_data2, h_data2, data_size, cudaMemcpyHostToDevice));

    printf("approach;element_count;bits;block_count;thread_count;time_ms;throughput;run\n");
    fflush(stdout);

    DECLARE_AND_CALL(bench_binary_op_baseline);
    DECLARE_AND_CALL(bench_binary_op_nogaps);
    DECLARE_AND_CALL(bench_binary_op_slabs);

    DECLARE_AND_CALL(bench_compressstore_slabs);

    DECLARE_AND_CALL(bench_filter_baseline);
    DECLARE_AND_CALL(bench_filter_nogaps);
    DECLARE_AND_CALL(bench_filter_slabs);

    DECLARE_AND_CALL(bench_groupby_sm_array);
    // DECLARE_AND_CALL(bench_groupby_sort_cub); // FIXME

    DECLARE_AND_CALL(bench_hop_baseline);
    DECLARE_AND_CALL(bench_hop_nogaps);
    DECLARE_AND_CALL(bench_hop_slabs);

    free(h_data1);
    free(h_data2);
    free(h_output);
    cudaFree(d_output);
    cudaFree(d_data1);
    cudaFree(d_data2);
}
