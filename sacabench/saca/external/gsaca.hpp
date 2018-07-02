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

namespace sacabench::reference_sacas {

    class gsaca {

    public:
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const *NAME = "REFERENCE GSACA";
        static constexpr char const *DESCRIPTION =
                "Computes a suffix array with the algorithm gsaca by Uwe Baier.";

        template<typename sa_index>
        inline static void construct_sa(sacabench::util::string_span text_with_sentinels,
                                        util::alphabet const& /* alphabet */,
                                        sacabench::util::span<sa_index> out_sa) {

            size_t n = text_with_sentinels.size();
            DCHECK_MSG(int(n) >= 0 && n == size_t(int(n)), "reference gsaca can only handle Textsizes that fit in a `int`.");

            const unsigned char *chars = text_with_sentinels.data();
            auto SA = std::make_unique<int[]>(n);

            /* int resultCode = */ ::gsaca(chars, SA.get(), n);

            for (size_t index = 0; index < n; index++) {
                out_sa[index] = static_cast<sa_index>(SA[index]);
            }
        }
    }; // class gsaca
} // namespace sacabench::reference_sacas
