/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include "string.hpp"
#include "util/type_extraction.hpp"

#pragma once


namespace sacabench::util {

    bool current_type = 0;

    void initialise_lms_checker_onfly(string_span t_0, string_span alph) {

        initialize_type_extraction_rtl_onfly(t_0);
        current_type = get_next_type_onfly();
    }


    bool get_next_is_lms_inplace() {

        if(symbols_left() > 1)
        {
            bool next_type = get_next_type_onfly();
            bool is_lms = (next_type && !current_type);
            current_type = next_type;

            return is_lms;
        }

        return false;
    }
}

/******************************************************************************/

