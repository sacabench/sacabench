/*******************************************************************************
 * Copyright (C) 2018 Christopher
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <util/assertions.hpp>
#include <util/bits.hpp>
#include <util/span.hpp>

namespace sacabench::util {

using uchar = unsigned char;

/// \brief represents a number type based on sa_index, but with special
/// operations for the first bits.
template <typename sa_index, uchar extra_bits>
class tagged_number {
private:
    sa_index memory;

    // Bitmask for 0011 1111 1111 1111 (if two extra bits are used)
    constexpr static sa_index MAX = static_cast<sa_index>(-1) >> extra_bits;

    template <uchar i>
    constexpr static sa_index BITMASK = static_cast<sa_index>(1)
                                        << (bits_of<sa_index> - 1 - i);

public:
    constexpr inline tagged_number() : memory(0) {
        DCHECK_LE(extra_bits, bits_of<sa_index>());
    }

    constexpr inline tagged_number(const sa_index m) : memory(m) {
        DCHECK_LE(extra_bits, bits_of<sa_index>);
    }

    constexpr inline sa_index number() const { return memory & MAX; }

    constexpr inline operator sa_index() const { return number(); }

    template <uchar i>
    constexpr inline bool get() const {
        DCHECK_LT(i, extra_bits);
        return memory & BITMASK<i>;
    }

    template <uchar i>
    constexpr inline void set(bool v) {
        // This is all 1s.
        constexpr sa_index ones = static_cast<sa_index>(-1);

        // This is 0000100000, with a 1 only at the correct position.
        constexpr sa_index mask = BITMASK<i>;

        //                         |-------------------| This sets the bit to 1,
        //                                               if v is true.
        memory = (memory & ~mask) | ((v * ones) & mask);
        //       |--------------|   This part sets the bit in `memory` to 0.
    }

    constexpr inline bool operator<(const sa_index& rhs) const {
        return number() < rhs;
    }

    constexpr inline bool operator==(const sa_index& rhs) const {
        return number() == rhs;
    }

    constexpr inline bool operator>(const sa_index& rhs) const {
        return number() > rhs;
    }

    constexpr inline void operator++() {
        DCHECK_LT(memory, MAX);
        ++memory;
    }

    constexpr inline void operator--() {
        DCHECK_GT(memory, 0);
        --memory;
    }

    constexpr inline tagged_number operator+(const sa_index& rhs) {
        return tagged_number(number() + rhs);
    }

    constexpr inline tagged_number operator-(const sa_index& rhs) {
        DCHECK_GE(number(), rhs);
        return tagged_number(number() - rhs);
    }

    constexpr inline tagged_number operator*(const sa_index& rhs) {
        return tagged_number(number() * rhs);
    }

    constexpr inline tagged_number operator/(const sa_index& rhs) {
        return tagged_number(number() / rhs);
    }
};

template<typename T, unsigned char N>
inline span<tagged_number<T, N>> cast_to_tagged_numbers(span<T> array) {
    // Since tagged_number is basically a T, it can be safely cast.
    auto* ptr = reinterpret_cast<tagged_number<T, N>*>(array.data());

    // Since tagged_number<T> is the same size as T, the size is equal.
    return span(ptr, array.size());
}

} // namespace sacabench::util
