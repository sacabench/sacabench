/*******************************************************************************
 * Copyright (C) 2018 Christopher Poeplau <christopher.poeplau@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <gtest/gtest.h>
#include "test/saca.hpp"
#include <saca/external/sads/sads.hpp>

TEST(sads_reference, construct_sa) {
	test::saca_corner_cases<sacabench::reference_sacas::sads>();
}
