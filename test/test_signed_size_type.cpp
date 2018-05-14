/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/signed_size_type.hpp>

using namespace sacabench;

TEST(signed_size_type, simple_test) {
    util::ssize test_number = -1;
    ASSERT_EQ(test_number, (signed int)-1);
}
