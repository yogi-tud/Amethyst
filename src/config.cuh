#pragma once

#include "common_code.cuh"

template <int bits>
constexpr __host__ __device__ uint64_t op_add(uint64_t l, uint64_t r)
{
    return (l + r) & gen_bitmask(bits);
}

template <int bits>
constexpr __host__ __device__ uint64_t op_math(uint64_t l, uint64_t r)
{
    return ((l*l + r*r + 4*l) / (l*l + r*r +5*l)) & gen_bitmask(bits);
}

template <int bits>
constexpr __host__ __device__ uint64_t op_mod(uint64_t l, uint64_t r)
{
    return (l % r) & gen_bitmask(bits);
}
template <int bits>
constexpr __host__ __device__ uint64_t op_div(uint64_t l, uint64_t r)
{
    return (l / r) & gen_bitmask(bits);
}
template <int bits>
constexpr __host__ __device__ uint64_t op_mul(uint64_t l, uint64_t r)
{
    return (l * r) & gen_bitmask(bits);
}
template <int bits>
constexpr __host__ __device__ uint64_t op_bool_simple(uint64_t l, uint64_t r)
{
    return (l  ||  r) & gen_bitmask(bits);
}

template <int bits>
constexpr __host__ __device__ uint64_t op_bool_complex(uint64_t l, uint64_t r)
{
    return ((l && !r) || (!l && r)) & gen_bitmask(bits);
}

constexpr __host__ __device__ int op_add_outbits(int bits)
{
    return bits == 64 ? 64 : bits + 1;
}
constexpr __host__ __device__ int op_mul_outbits(int bits)
{
    return bits * 2 >= 64 ? 64 : bits * 2;
}
constexpr __host__ __device__ int op_mod_outbits(int bits)
{
    return bits == 64 ? 64 : bits;
}

inline __device__ uint64_t cuda_atomic_add(uint64_t* p, uint64_t v)
{
    return atomicAdd((unsigned long long*)p, (unsigned long long)v);
}

// hop
constexpr uint64_t HOP_INIT = 0;
constexpr __host__ __device__ uint64_t HOP_OP(uint64_t l, uint64_t r)
{
    return l + r;
}
inline __device__ uint64_t HOP_OP_ATOMIC(uint64_t* p, uint64_t v)
{
    return cuda_atomic_add(p, v);
}

// binary op
template <int bits>
constexpr __host__ __device__ uint64_t BINARY_OP(uint64_t a, uint64_t b)
{
    //return op_add<op_add_outbits(bits)>(a, b);
   // return op_div<op_mod_outbits(bits)>(a, b);
   //return op_mod<op_mod_outbits(bits)>(a, b);
   // return op_bool_simple<op_mod_outbits(bits)>(a, b);

 return op_mul<op_mul_outbits(bits)>(a, b);
  
   
}

constexpr __host__ __device__ int BINARY_OP_OUTBITS(int bits)
{
    return op_add_outbits(bits);
}

// group by
constexpr uint64_t GROUP_BY_INIT_VALUE = 0;
template <int bits>
constexpr __host__ __device__ uint64_t GROUP_BY_OP(uint64_t a, uint64_t b)
{
    return op_add<bits>(a, b);
}
template <int bits>
__host__ __device__ typename tight_type<bits>::cuda_type
GROUP_BY_OP_ATOMIC(typename tight_type<bits>::cuda_type* ptr, typename tight_type<bits>::cuda_type val)
{
    return atomicAdd(ptr, val);
}

// filter
template <int bits>
constexpr __host__ __device__ bool FILTER_PREDICATE_OP(uint64_t value)
{
    constexpr uint64_t half = ((uint64_t)1) << (bits - 1);
    bool res = value >= half;
    return res;
}

// compresstore
constexpr size_t COMPRESSTORE_CHUNK_LENGTH = 1024; // TODO: make this dynamic / templated
