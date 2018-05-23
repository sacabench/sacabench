/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/m_suf_sort.hpp>

using namespace sacabench::m_suf_sort;

TEST(prefix_doubling, test) {
    test::saca_corner_cases<m_suf_sort<>>();
}
