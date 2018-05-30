/*******************************************************************************
 * util/type_extraction.hpp
 *
 * Copyright (C) 2018 Jonas Bode <jonas.bode@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/
#include <array>
#include <tuple>
#include "util/string.hpp"
#include "util/signed_size_type.hpp"
#include "util/span.hpp"
#include "util/type_extraction.hpp"
#include "util/insert_lms.hpp"

#pragma once


namespace sacabench::saca {

    void calculate_sa(util::string_span t_0, util::container<size_t> sa) {
           
    // Berechne zuerst Bucket Arraylist bkt
    // Füge danach LMS Substrings ein

    // Erzeuge neuen String T_1 aus den Substrings
    // Falls neue Alphabetgröße von T_1 = |T_1|, dann ist die Ordnung der LMS-Substrings gleichzeitig auch die der LMS-Suffixe
    // Falls nicht, rufe Rekursion auf

    // Aus Ordnung der Suffixe benutze bkt um das endgültige SA zu berechnen

    }

    void calculate_deep_sa(util::string_span t_0, util::container<size_t> sa) {



    }

}

/******************************************************************************/

