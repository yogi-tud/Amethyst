#pragma once
#include <iterator>
#include <cstdint>

struct bit_iterator {
    uint8_t* byte_ptr;
    size_t idx;

    __host__ __device__ bit_iterator(uint8_t* byte_ptr, size_t idx = 0) : byte_ptr(byte_ptr), idx(idx)
    {
    }

    __host__ __device__ bool operator[](size_t i) const
    {
        return *bit_iterator(byte_ptr, idx + i);
    }

    __host__ __device__ bool operator*() const
    {
        return (byte_ptr[idx / 8] >> (idx % 8)) & 0b1;
    }

    __host__ __device__ bit_iterator operator+(size_t i) const
    {
        return bit_iterator{byte_ptr, idx + i};
    }

    __host__ __device__ bit_iterator operator-(size_t i) const
    {
        return bit_iterator{byte_ptr, idx - i};
    }

    __host__ __device__ bit_iterator operator++()
    {
        idx++;
        return *this;
    }
    __host__ __device__ bit_iterator operator--()
    {
        idx--;
        return *this;
    }

    __host__ __device__ bit_iterator operator++(int)
    {
        bit_iterator copy = *this;
        idx++;
        return copy;
    }
    __host__ __device__ bit_iterator operator--(int)
    {
        bit_iterator copy = *this;
        idx--;
        return copy;
    }
};
template <>
struct std::iterator_traits<bit_iterator> {
    typedef bool value_type;
};
