/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <util/assertions.hpp>

using namespace sacabench;

TEST(assert_text_length, simple) {
    util::string_span text = "hallo, test!"_s;
    ASSERT_TRUE(util::assert_text_length<uint8_t>(text.size(), 2u));
    ASSERT_TRUE(util::assert_text_length<uint16_t>(text.size(), 2u));
    ASSERT_TRUE(util::assert_text_length<uint32_t>(text.size(), 2u));
    ASSERT_TRUE(util::assert_text_length<uint64_t>(text.size(), 2u));
    ASSERT_TRUE(util::assert_text_length<size_t>(text.size(), 2u));
}
TEST(assert_text_length, limits) {

    size_t limit_size = std::numeric_limits<size_t>::max();
    ASSERT_FALSE(util::assert_text_length<size_t>(limit_size,2u));
    ASSERT_FALSE(util::assert_text_length<size_t>((limit_size>>1),2u));
    ASSERT_TRUE(util::assert_text_length<size_t>((limit_size>>2),2u));
    
    uint8_t limit_8 = std::numeric_limits<uint8_t>::max();
    ASSERT_FALSE(util::assert_text_length<uint8_t>(limit_8,2u));
    ASSERT_FALSE(util::assert_text_length<uint8_t>((limit_8>>1),2u));
    ASSERT_TRUE(util::assert_text_length<uint8_t>((limit_8>>2),2u));
}
