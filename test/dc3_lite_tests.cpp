/*******************************************************************************
 * Copyright (C) 2018 Nico Bertram <nico.bertram@gmail.com>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"

#include <saca/dc3_lite.hpp>

using namespace sacabench::dc3_lite;

TEST(dc3_lite, test) {
    test::saca_corner_cases<dc3_lite>();
}
