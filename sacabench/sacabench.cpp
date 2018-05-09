/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#include <cstdint>
#include "util/bucket_size.hpp"
#include "util/container.hpp"

#include "util/saca.hpp"

std::int32_t main(std::int32_t /*argc*/, char const** /*argv*/) {

    auto& saca_list = saca_list::get();
    for (const auto& a : saca_list) {
        a->run_example();
    }

    return 0;
}

/******************************************************************************/
