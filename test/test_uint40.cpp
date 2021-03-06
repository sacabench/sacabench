/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@udo.edu>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/uint_types.hpp>

TEST(uint40, simple_test) {
    using sacabench::util::uint40;
    uint40 a = 0;
    a += 1;
    ASSERT_EQ(a, uint40(1));
}
