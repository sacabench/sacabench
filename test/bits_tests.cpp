/*******************************************************************************
 * Copyright (C) 2018 Marvin Löbel <loebel.marvin@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/bits.hpp>

using namespace sacabench::util;

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
