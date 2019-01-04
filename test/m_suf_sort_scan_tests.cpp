/*******************************************************************************
 * Copyright (C) 2018 Rosa Pink
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <util/container.hpp>
#include <util/span.hpp>
#include <util/string.hpp>
#include <saca/m_suf_sort_scan.hpp>
#include <util/alphabet.hpp>
#include <stack>
#include <utility>

using namespace sacabench;
using namespace sacabench::util;
using namespace sacabench::m_suf_sort_scan;

TEST(m_suf_sort_scan, test) {
    test::saca_corner_cases<m_suf_sort_scan2>();
}
