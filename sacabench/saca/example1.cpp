/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include "util/saca.hpp"
#include "saca/example1.hpp"

std::int32_t main(std::int32_t /*argc*/, char const** /*argv*/) {
    sacabench::example1::example1::run_example();
    return 0;
}
