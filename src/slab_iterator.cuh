#pragma once

#include "common_code.cuh"

template <int bits>
struct sentinel {
    static constexpr size_t bytecount = (bits + 7) / 8;
    typedef typename std::conditional<
        (bytecount > 4),
        uint64_t,
        typename std::conditional<(bytecount > 2), uint32_t, typename std::conditional<(bytecount > 1), uint16_t, uint8_t>::type>::type>::type
        sorttype_raw;
    static constexpr sorttype_raw bitmask = gen_bitmask(bits);
    static constexpr size_t elems_per_slab = 64 / bits;
    uint64_t* data;
    size_t index;

    struct value_t {
        sorttype_raw v;
        __host__ __device__ value_t() = default;
        __host__ __device__ value_t(sorttype_raw v) : v(v & bitmask)
        {
        }
        __host__ __device__ value_t operator+(value_t v2) const
        {
            return value_t{(sorttype_raw)((v + v2.v) & bitmask)};
        }
        __host__ __device__ value_t operator+=(value_t v2)
        {
            v = (v + v2.v) & bitmask;
            return v;
        }
        __host__ __device__ value_t operator&=(sorttype_raw bm)
        {
            v &= bm;
        }
        __host__ __device__ bool operator==(value_t v2) const
        {
            return (v & bitmask) == (v2.v & bitmask);
        }
        __host__ __device__ bool operator<(value_t v2) const
        {
            return (v & bitmask) < (v2.v & bitmask);
        }
        __host__ __device__ operator sorttype_raw() const
        {
            return v & bitmask;
        }
    };
    typedef value_t sorttype;

    __host__ __device__ sentinel(uint64_t* data, size_t index = 0) : data(data), index(index)
    {
    }

    __host__ __device__ sentinel(const sentinel&& s) : data(s.data), index(s.index)
    {
    }

    __host__ __device__ sentinel(const sentinel& s) : data(s.data), index(s.index)
    {
    }

    __host__ __device__ sorttype operator=(sentinel&& v)
    {
        *this = (sorttype)v;
        return (sorttype)v;
    }

    __host__ __device__ sorttype operator=(sentinel v)
    {
        *this = (sorttype)v;
        return (sorttype)v;
    }

    __device__ sorttype operator=(sorttype v)
    {
        v &= bitmask;
        ((sorttype*)data)[index] = v;
        return v;
    }

    __host__ __device__ operator sorttype() const
    {
        return ((sorttype*)data)[index] & bitmask;
    }

    __host__ __device__ operator sorttype_raw() const
    {
        return ((sorttype_raw*)data)[index] & bitmask;
    }

    __host__ __device__ bool operator==(const sentinel<bits> other)
    {
        return ((sorttype)(*this) == (sorttype)other);
    }
};

template <int bits>
struct slab_iterator {
    typedef typename sentinel<bits>::sorttype sorttype;
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename sentinel<bits>::value_t;
    using difference_type = size_t;
    using pointer = sentinel<bits>*;
    using reference = sentinel<bits>;

    uint64_t* data;
    size_t index;

    __host__ __device__ slab_iterator(uint64_t* data, size_t index = 0) : data(data), index(index)
    {
    }

    __host__ __device__ sentinel<bits> operator*() const
    {
        return sentinel<bits>{data, index};
    }

    __host__ __device__ sentinel<bits> operator[](size_t i)
    {
        return sentinel<bits>(data, index + i);
    }

    __host__ __device__ slab_iterator<bits> operator+(size_t i)
    {
        return slab_iterator{data, index + i};
    }

    __host__ __device__ slab_iterator<bits> operator-(size_t i)
    {
        return slab_iterator{data, index - i};
    }

    __host__ __device__ size_t operator-(const sentinel<bits>& s)
    {
        return this->index - s.index;
    }

    __host__ __device__ sorttype operator++()
    {
        index++;
        return **this;
    }

    __host__ __device__ sorttype operator--()
    {
        index--;
        return **this;
    }
};
