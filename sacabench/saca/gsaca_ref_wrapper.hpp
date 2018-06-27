//
//  gsaca.hpp
//  gsaca
//
//  Created by David Piper on 09.05.18.
//  Copyright © 2018 David Piper. All rights reserved.
//

#pragma once

#include <util/span.hpp>
#include <util/string.hpp>
#include <limits.h>
#include <stdlib.h>

#include <gsaca.h>

namespace sacabench::gsaca {

    class gsaca_ref_wrapper {

    public:
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const *NAME = "REFERENCE GSACA";
        static constexpr char const *DESCRIPTION =
                "Computes a suffix array with the algorithm gsaca by Uwe Baier.";

        template<typename sa_index>
        inline static void construct_sa(sacabench::util::string_span text_with_sentinels,
                                        util::alphabet const /* &alphabet */,
                                        sacabench::util::span<sa_index> out_sa) {

            const unsigned char *chars = text_with_sentinels.data();
            int n = int(text_with_sentinels.size());
            int *SA = new int[n];

            /* int resultCode = */ ::gsaca(chars, SA, n);

            for (int index = 0; index < n; index++) {
                out_sa[index] = static_cast<sa_index>(SA[index]);
            }
        }
    }; // class gsaca_ref_wrapper
} // namespace sacabench::gsaca
