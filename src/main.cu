#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "utils.cuh"
#include "fast_prng.cuh"
#include <cstring>
#include "common_code.cuh"
#include <bit_iterator.cuh>
bool do_validation = true;

int run_count = 2;
int device_id =0;
int bench_part =2;
float selectivity = -0.0f;
int mask_distribution= -1;
int wantbits = 64;
int cluster_count = 4;

size_t grid_size_min = 32;
size_t grid_size_max = 8192;
size_t block_size_min = 32;
size_t block_size_max = 1024;


#define DECLARE_AND_CALL(name)                                                                                                                       \
    void name(                                                                                                                                       \
        size_t element_count, uint64_t* h_data1, uint64_t* h_data2, uint64_t* h_output, uint64_t* d_data1, uint64_t* d_data2, uint64_t* d_output,    \
        size_t bitmask_size);                                                                                                                        \
    name(element_count, h_data1, h_data2, h_output, d_data1, d_data2, d_output, bitmask_size);


void generate_mask_uniform(uint8_t* h_buffer, uint64_t offset, uint64_t length, double selectivity)
{
    uint64_t p_adjusted = selectivity * UINT32_MAX;
    fast_prng rng(42);
    for (uint64_t i = offset; i < offset + length; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (rng.rand() < p_adjusted) {
                acc |= (1 << j);
            }
        }
        h_buffer[i] = acc;
    }
}
/***
 *
 *
 * @param total_elements number of bits in mask
 * @param selectivity percentage of elements that are set to 1 in bitmask
 * @param cluster_count on how many clusters should the elements distributed
 * bitmask 0 -> all algos
 * bitmask 1 -> space 8 + cub
 * @return a bitmask as uint8_t which is used for compressstore operations
 *
 *
 */
std::vector<uint8_t> create_bitmask(float selectivity, size_t cluster_count, size_t total_elements)
{
        std::vector<bool> bitset;
        bitset.resize(total_elements);
    size_t total_set_one = selectivity * total_elements;
    size_t cluster_size = total_set_one / cluster_count;
    size_t slice = bitset.size() / cluster_count;

        //start by setting all to zero
        for(int i=0; i<bitset.size();i++)
        {
            bitset[i]=0;
        }

    for (int i = 0 ; i< cluster_count;i++)
    {
        for(int k =0;k< cluster_size;k++)
        {
            size_t cluster_offset = i*slice;
            bitset[k+cluster_offset]=1;
        }
    }


    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements/8);

    for(int i=0; i<total_elements/8;i++)
    {
        final_bitmask_cpu[i]=0;
    }

    for(int i =0;i< bitset.size();i++)
    {
        //set bit of uint8
        if(bitset[i])
        {
            uint8_t current =final_bitmask_cpu[i/8];
            int location = i % 8;
            current = 1 << (7-location);
            uint8_t add_res = final_bitmask_cpu[i/8];
            add_res = add_res | current;
            final_bitmask_cpu[i/8] = add_res;
        }

    }


    return final_bitmask_cpu;
}


int main(int argc, char** argv)
{
    bit_iterator foo{NULL};
    foo++;
    size_t element_count = 1 << 27; //27 - 1 gib per array
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
        else if (strcmp(w_arg, "-p") == 0) {
            w_argc--;
            bench_part = atoi(n_arg);
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
        else if (strcmp(w_arg, "-d") == 0) {
            w_argc--;
            device_id = atoi(n_arg);
            
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
         else if (strcmp(w_arg, "-s") == 0) {
            w_argc--;
            selectivity = atof(n_arg);
            
        }
        else if (strcmp(w_arg, "-m") == 0) {
            w_argc--;
            mask_distribution = atoi(n_arg);
            
        }



     
        else if (!strcmp(w_arg, "--help") || !strcmp(w_arg, "-h")) {
            printf(
                "%s [OPTIONS]\n"
                "    -h | --help: print this help\n"
                "    -m <int>:    distribution of bits in bitmask -1. Random mask 0-unifirm 1-single cluster 2-multi cluster \n"
                "    -s <float>:  selectivity from                \n "
                "    --nv:        disable validation (on by default)\n"
                "    -n <int>:    number of elements (default %lu)\n"
                "    -N <int>:    like -n, but as power of two, given x will lead to 2^x elements\n"
                "    -r <int>:    number of repeat runs for each benchmark configuration (default %u)\n"
                "    -b <int>:    maximum number of bits to test for each element, minimum is always 1 (default %u)\n"
                "    -p <int>:    Bench is split into 2 parts. 0 is first, 1 is second, 2 is all "
                "    -gsm <int>:  minimum grid size to test\n"
                "    -gsM <int>:  maximum grid size to test\n"
                "    -bsm <int>:  minimum block size to test\n"
                "    -bsM <int>:  maximum block size to test\n"
                "    --ff <int>:  instead of using random input data, fill all input bytes with 0xff\n",
                argv[0], element_count, run_count, wantbits
            );
            exit(0);
        }
        else {
            printf("ignoring unknown argument: \"%s\"\n", w_arg);
        }
    }

    size_t data_size = element_count * sizeof(uint64_t);
    size_t bitmask_size = ((element_count + 64 - 1) / 64) * sizeof(uint64_t);
    size_t output_data_size = data_size * 4; // group by need 2 outputs and reduce by key need 2 more
    CUDA_TRY(cudaSetDevice(device_id));  // pick GPU device
    // generate data (compressed)
    uint64_t* h_data1 = (uint64_t*)malloc(data_size);
    uint64_t* h_data2 = (uint64_t*)malloc(data_size);
    uint64_t* h_output = (uint64_t*)malloc(output_data_size);
    fast_prng rng(42);
    if (use_rng) {
        for (size_t i = 0; i < data_size / sizeof(uint64_t); i++) {
            h_data1[i] = ((uint64_t)rng.rand()) << 32 | (uint64_t)rng.rand();
            if(h_data1[i]==0)h_data1[i]=1;; //minimum value for random data is 1 so mod and div always work

            h_data2[i] = ((uint64_t)rng.rand()) << 32 | (uint64_t)rng.rand();
            if(h_data2[i]==0)h_data2[i]=1;
        }
    }
    else {
        memset(h_data1, 0xFF, data_size);
        memset(h_data2, 0xFF, data_size);
    }
   
   
    std::vector<uint8_t> im; //intermediate mask
    im.reserve(data_size);
     

   //SET MASK Distributions for Compressstore benchmarks:
   if(mask_distribution !=-1)
   {
    switch (mask_distribution)
    {
       case 0:  
            cluster_count =1;
            generate_mask_uniform((uint8_t*)h_data2, 0, bitmask_size, selectivity);
            break;

        case 1: //generate 1 big cluster
            cluster_count =1;
            im = create_bitmask(selectivity,1,element_count);
            memcpy ( h_data2, im.data(),im.size()*sizeof(uint8_t));
            break;

        case 2: //multi cluster
            cluster_count =8;
            im = create_bitmask(selectivity,8,element_count);
            memcpy ( h_data2, im.data(),im.size()*sizeof(uint8_t));
        break;
    }
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

    //write a loop that iterates over d_data1 and summarizes it
     
 // wantbits=30;
//  bench_part=1;

    if(bench_part == 0 || bench_part == 2)
    {

    
    DECLARE_AND_CALL(bench_compressstore_baseline_cub);
    DECLARE_AND_CALL(bench_compressstore_slabs);

      DECLARE_AND_CALL(bench_groupby_sm_array);
   //  DECLARE_AND_CALL(bench_groupby_sort_cub); // FIXME

    DECLARE_AND_CALL(bench_hop_baseline);
   DECLARE_AND_CALL(bench_hop_nogaps);
    DECLARE_AND_CALL(bench_hop_slabs);
   
    }

    if(bench_part == 1 || bench_part == 2)
    {
    
    DECLARE_AND_CALL(bench_binary_op_nogaps);
    DECLARE_AND_CALL(bench_binary_op_slabs);
    DECLARE_AND_CALL(bench_binary_op_baseline);

    //DECLARE_AND_CALL(bench_filter_baseline);
    //DECLARE_AND_CALL(bench_filter_nogaps);
  //  DECLARE_AND_CALL(bench_filter_slabs);
    }



    free(h_data1);
    free(h_data2);
    free(h_output);
    cudaFree(d_output);
    cudaFree(d_data1);
    cudaFree(d_data2);
}
