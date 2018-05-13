/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <saca/deep_shallow.hpp>

using namespace sacabench;
using deep_shallow = sacabench::deep_shallow::saca;

TEST(deep_shallow, simple) {
    saca::construct_sa(nullptr, 0, nullptr);
    ASSERT_TRUE(true);
}
