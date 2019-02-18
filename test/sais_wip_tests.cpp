/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/sais_with_parallel_induce.hpp>

TEST(sais_wip, construct_sa) {
	test::saca_corner_cases<sacabench::sais_wip::sais_wip>();
}
