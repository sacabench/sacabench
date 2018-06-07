/*******************************************************************************
 * Copyright (C) 2018 Florian Kurpicz <florian.kurpicz@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/


#include <cstdint>
#include "util/bucket_size.hpp"
#include "util/container.hpp"

#include "util/saca.hpp"
#include "saca/sacak.hpp"

#include "util/type_extraction.hpp"
#include "util/insert_lms.hpp"
#include "util/string.hpp"

using namespace sacabench::util;

std::int32_t main(std::int32_t /*argc*/, char const** /*argv*/) {

    string_span test_text = "caabaccaabacaa"_s;
    container<size_t> sa = make_container<size_t>(14);

    for (size_t i = 0; i < sa.size(); i++)
    {
        sa[i] = 0;
    }

    std::cout << "\n Wert von bla ist: " << 42 << "\n" << std::endl;

    insert_lms_rtl(test_text, sa);

    for (size_t i = 0; i < sa.size(); i++)
    {
        std::cout << "Wert von sa[" << i << "] ist: " << sa[i] << "\n" << std::endl;
    }

    return 0;

}

/******************************************************************************/
