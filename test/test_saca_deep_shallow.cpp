/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/deep_shallow.hpp>

using namespace sacabench;
using ds = sacabench::deep_shallow::saca;

TEST(deep_shallow, simple) {
    auto input = "hallo"_s;
    auto sa = util::make_container<size_t>(input.size());

    ds::construct_sa<size_t>(input, 0, sa);
    ASSERT_TRUE(true);
}
