#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <cstddef>
#include <cassert>
#include <bitset>
#include "fast_prng.cuh"
#include <string>
#define UNUSED(VAR) (void)(true ? (void)0 : ((void)(VAR)))

// TODO: improve this
constexpr size_t clog2(size_t v)
{
    size_t res = 0;
    while (v >>= 1) res++;
    return res;
}


inline void error(const char* error)
{
    fputs(error, stderr);
    fputs("\n", stderr);
    assert(false);
    exit(EXIT_FAILURE);
}
inline void alloc_failure()
{
    error("memory allocation failed");
}

template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (uint32_t i = offset; i < offset + length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer + offset, length * sizeof(T), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template<std::size_t N>
void bitset_reverse(std::bitset<N> &b) {
    for(std::size_t i = 0; i < N/2; ++i) {
        bool t = b[i];
        b[i] = b[N-i-1];
        b[N-i-1] = t;
    }
}

template <typename T>
void gpu_buffer_print_le(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length * sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer + offset, length * sizeof(T), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < length; i++) {
        std::bitset<sizeof(T) * 8> bits(h_buffer[i]);
        bitset_reverse(bits);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

template <typename T>
T* vector_to_gpu(const std::vector<T>& vec)
{
    T* buff;
    const auto size = vec.size() * sizeof(T);
    CUDA_TRY(cudaMalloc(&buff, size));
    CUDA_TRY(cudaMemcpy(buff, &vec[0], size, cudaMemcpyHostToDevice));
    return buff;
}

template <typename T>
std::vector<T> gpu_to_vector(T* buff, size_t length)
{
    std::vector<T> vec;
    vec.resize(length);
    CUDA_TRY(cudaMemcpy(&vec[0], buff, length * sizeof(T), cudaMemcpyDeviceToHost));
    return vec;
}

template <class T>
struct dont_deduce_t {
    using type = T;
};

template <typename T>
T gpu_to_val(T* d_val)
{
    T val;
    CUDA_TRY(cudaMemcpy(&val, d_val, sizeof(T), cudaMemcpyDeviceToHost));
    return val;
}

template <typename T>
void val_to_gpu(T* d_val, typename dont_deduce_t<T>::type val)
{
    CUDA_TRY(cudaMemcpy(d_val, &val, sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
T* alloc_gpu(size_t length)
{
    T* buff;
    CUDA_TRY(cudaMalloc(&buff, length * sizeof(T)));
    return buff;
}

template <typename T>
__device__ __host__ T ceil2mult(T val, typename dont_deduce_t<T>::type mult)
{
    T rem = val % mult;
    if (rem) return val + mult - rem;
    return val;
}

template <typename T>
__device__ __host__ T ceildiv(T div, typename dont_deduce_t<T>::type divisor)
{
    T rem = div / divisor;
    if (rem * divisor == div) return rem;
    return rem + 1;
}

template <typename T>
__device__ __host__ T overlap(T value, typename dont_deduce_t<T>::type align)
{
    T rem = value % align;
    if (rem) return align - rem;
    return 0;
}

inline __host__ __device__ void byte_print(uint8_t byte, bool little_endian_bits)
{
    for (int j = 0; j < 8; j++) {
        printf("%c", (byte >> (little_endian_bits ? j : (7 - j))) & 0b1 ? '1' : '0');
    }
}

template <typename T>
__host__ __device__ void bit_print(T data, bool little_endian = false, bool little_endian_bits = false, bool spacing = true, const char* trailer = "")
{
    uint8_t* b = (uint8_t*)&data;
    for (size_t i = 0; i < sizeof(T); i++) {
        byte_print(b[little_endian ? i : sizeof(T) - i - 1], little_endian_bits);
        if (spacing) printf(" ");
    }
    printf("%s", trailer);
}

template <typename T>
__host__ __device__
T gcd(T a, T b)
{
    for (;;)
    {
        if (a == 0) return b;
        b %= a;
        if (b == 0) return a;
        a %= b;
    }
}

template <typename T>
__host__ __device__
T lcm(T a, T b)
{
    T temp = gcd(a, b);
    return temp ? (a / temp * b) : 0;
}

template <typename T>
std::vector<uint8_t> gen_predicate(const std::vector<T>& col, bool (*predicate)(T value), size_t* one_count = NULL)
{
    std::vector<uint8_t> predicate_bitmask{};
    size_t mask_bytes = ceildiv(col.size(), 8);
    predicate_bitmask.reserve(mask_bytes);
    auto it = col.begin();
    size_t one_count_loc = 0;
    for (size_t i = 0; i < mask_bytes; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (it == col.end()) break;
            if (predicate(*it++)) {
                acc |= (1 << j);
                one_count_loc++;
            }
        }
        predicate_bitmask.push_back(acc);
    }
    if (one_count) *one_count = one_count_loc;
    return predicate_bitmask;
}

template <typename RET_TYPE, typename DISPATCHER, size_t MIN, size_t MAX, typename... ARGS>
struct pow2_template_dispatch {

    template <
        size_t MIN_MAYBE,
        size_t MAX_MAYBE,
        size_t P2,
        typename... RT_ARGS,
        typename ENABLED = typename std::enable_if<(P2 >= MIN_MAYBE && P2 <= MAX_MAYBE)>::type>
    static RET_TYPE call_me_maybe(int dummy, RT_ARGS... rt_args)
    {
        return DISPATCHER::template call<ARGS..., P2>(rt_args...);
    }

    template <
        size_t MIN_MAYBE,
        size_t MAX_MAYBE,
        size_t P2,
        typename... RT_ARGS,
        typename ENABLED = typename std::enable_if<(P2<MIN_MAYBE || P2> MAX_MAYBE)>::type>
    static RET_TYPE call_me_maybe(float dummy, RT_ARGS... rt_args)
    {
        assert(0);
        exit(1);
        return call_me_maybe<MIN_MAYBE, MAX_MAYBE, MIN_MAYBE>(0, rt_args...);
    }

    template <typename... RT_ARGS>
    static RET_TYPE call(size_t p2, RT_ARGS... args)
    {
        switch (p2) {
            case 1: return call_me_maybe<MIN, MAX, 1>(0, args...);
            case 2: return call_me_maybe<MIN, MAX, 2>(0, args...);
            case 4: return call_me_maybe<MIN, MAX, 4>(0, args...);
            case 8: return call_me_maybe<MIN, MAX, 8>(0, args...);
            case 16: return call_me_maybe<MIN, MAX, 16>(0, args...);
            case 32: return call_me_maybe<MIN, MAX, 32>(0, args...);
            case 64: return call_me_maybe<MIN, MAX, 64>(0, args...);
            case 128: return call_me_maybe<MIN, MAX, 128>(0, args...);
            case 256: return call_me_maybe<MIN, MAX, 256>(0, args...);
            case 512: return call_me_maybe<MIN, MAX, 512>(0, args...);
            case 1024: return call_me_maybe<MIN, MAX, 1024>(0, args...);
            case 2048: return call_me_maybe<MIN, MAX, 2048>(0, args...);
            case 4096: return call_me_maybe<MIN, MAX, 4096>(0, args...);
            case 8192: return call_me_maybe<MIN, MAX, 8192>(0, args...);
            default: assert(0); exit(1);
        }
    }
};
