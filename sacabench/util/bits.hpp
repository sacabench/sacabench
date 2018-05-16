/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <climits>
#include <cstdint>

namespace sacabench::util {
    /// Generic constant equal to the size of a type in bits.
    ///
    /// Example:
    /// ```cpp
    /// ASSERT_EQ(util::bits_of<uint64_t>, 64);
    /// ```
    template<typename type>
    constexpr size_t bits_of = sizeof(type) * CHAR_BIT;

    // Source: https://stackoverflow.com/questions/3272424/compute-fast-log-base-2-ceiling
    /// Calculates `ceil(log2(x))` efficiently.
    inline uint64_t ceil_log2(uint64_t x) {
        static const uint64_t t[6] = {
            0xFFFFFFFF00000000ull,
            0x00000000FFFF0000ull,
            0x000000000000FF00ull,
            0x00000000000000F0ull,
            0x000000000000000Cull,
            0x0000000000000002ull
        };

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
}
