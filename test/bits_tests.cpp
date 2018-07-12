/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/bits.hpp>

using namespace sacabench::util;

TEST(Bits, next_pow2) {
    ASSERT_EQ(next_pow2(0), 1);
    ASSERT_EQ(next_pow2(1), 1);
    ASSERT_EQ(next_pow2(2), 2);
    ASSERT_EQ(next_pow2(3), 4);
    ASSERT_EQ(next_pow2(4), 4);
    ASSERT_EQ(next_pow2(5), 8);
    ASSERT_EQ(next_pow2(6), 8);
    ASSERT_EQ(next_pow2(7), 8);
    ASSERT_EQ(next_pow2(8), 8);
    ASSERT_EQ(next_pow2(9), 16);
}

TEST(Bits, bits_of) {
    ASSERT_EQ(bits_of<uint64_t>, 64u);
    ASSERT_EQ(bits_of<uint32_t>, 32u);
    ASSERT_EQ(bits_of<uint16_t>, 16u);
    ASSERT_EQ(bits_of<uint8_t>, 8u);

    ASSERT_EQ(bits_of<int64_t>, 64u);
    ASSERT_EQ(bits_of<int32_t>, 32u);
    ASSERT_EQ(bits_of<int16_t>, 16u);
    ASSERT_EQ(bits_of<int8_t>, 8u);

    struct Foo {
        uint8_t a;
        // Due to alignment, there will be 7 bytes of padding here
        uint64_t b;
    };
    ASSERT_EQ(bits_of<Foo>, 128u);
}

TEST(Bits, ceil_log2) {
    ASSERT_EQ(ceil_log2(0), 0u);
    ASSERT_EQ(ceil_log2(1), 0u);

    ASSERT_EQ(ceil_log2(2), 1u);

    ASSERT_EQ(ceil_log2(3), 2u);
    ASSERT_EQ(ceil_log2(4), 2u);

    ASSERT_EQ(ceil_log2(5), 3u);
    ASSERT_EQ(ceil_log2(6), 3u);
    ASSERT_EQ(ceil_log2(7), 3u);
    ASSERT_EQ(ceil_log2(8), 3u);

    ASSERT_EQ(ceil_log2(9), 4u);
    ASSERT_EQ(ceil_log2(10), 4u);
    ASSERT_EQ(ceil_log2(11), 4u);
    ASSERT_EQ(ceil_log2(12), 4u);
    ASSERT_EQ(ceil_log2(13), 4u);
    ASSERT_EQ(ceil_log2(14), 4u);
    ASSERT_EQ(ceil_log2(15), 4u);
    ASSERT_EQ(ceil_log2(16), 4u);

    ASSERT_EQ(ceil_log2(17), 5u);
    // ...
    ASSERT_EQ(ceil_log2(32), 5u);

    ASSERT_EQ(ceil_log2(33), 6u);
    // ...
    ASSERT_EQ(ceil_log2(64), 6u);

    ASSERT_EQ(ceil_log2(65), 7u);
    // ...
    ASSERT_EQ(ceil_log2(128), 7u);

    for(size_t i = 8; i < 64; i++) {
        ASSERT_EQ(ceil_log2((1ull << (i - 1)) + 1), i);
        // ...
        ASSERT_EQ(ceil_log2(1ull << i), i);
    }
}
