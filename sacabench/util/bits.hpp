/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <climits>
#include <cmath>
#include <cstdint>

namespace sacabench::util {
/// Generic constant equal to the size of a type in bits.
///
/// Example:
/// ```cpp
/// ASSERT_EQ(util::bits_of<uint64_t>, 64);
/// ```
template <typename type>
constexpr size_t bits_of = sizeof(type) * CHAR_BIT;

// Source:
// https://stackoverflow.com/questions/3272424/compute-fast-log-base-2-ceiling
/// Calculates `ceil(log2(x))` efficiently.
inline uint64_t ceil_log2(uint64_t x) {
    static const uint64_t t[6] = {  0xFFFFFFFF00000000ull,
                                    0x00000000FFFF0000ull,
                                    0x000000000000FF00ull,
                                    0x00000000000000F0ull,
                                    0x000000000000000Cull,
                                    0x0000000000000002ull};

    uint64_t y = (((x & (x - 1)) == 0) ? 0 : 1);
    uint64_t j = 32;
    uint64_t i;

    for (i = 0; i < 6; i++) {
        uint64_t k = (((x & t[i]) == 0) ? 0 : j);
        y += k;
        x >>= k;
        j >>= 1;
    }

    return y;
}

/*
inline uint64_t ceil_log_base(uint64_t x, uint64_t base) {
    double loga = std::log(double(x)) / std::log(double(base));
    return std::ceil(loga);
}
*/

inline uint64_t powi(uint64_t base, uint64_t exp) {
    return std::pow(base, exp);
}

/// Calculates 'floor(log2(x))'. Placeholder for possibly efficient
/// implementation.
inline uint64_t floor_log2(uint64_t x) { return floor(std::log2(x)); }

/// Calculates next power of 2.
inline uint64_t next_pow2(uint64_t x) {
    if (x < 2)
        return 1;
    return 1ull << (64 - __builtin_clzll(x - 1));
}

// https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
template <typename sa_index>
inline sa_index min(sa_index x, sa_index y) {
    return y ^ ((x ^ y) & -(x < y));
}

// https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
template <typename sa_index>
inline sa_index max(sa_index x, sa_index y) {
    return x ^ ((x ^ y) & -(x < y));
}

} // namespace sacabench::util
