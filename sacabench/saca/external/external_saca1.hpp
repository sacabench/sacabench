/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace sacabench::reference_sacas {
    struct external_saca1 {
        template <typename sa_index, typename Fn>
        inline static void run_external(util::string_span text,
                                 util::span<sa_index> out_sa,
                                 Fn saca_fn) {

            size_t n = text.size();
            auto sa_correct_size = util::make_container<int32_t>(n);
            auto* text_ptr = const_cast<uint8_t*>(text.data());

            if(n < 2) {
                return;
            }

            saca_fn(text_ptr, sa_correct_size.data(), n);

            for(size_t i = 0; i < n; ++i) {
                out_sa[i] = sa_correct_size[i];
            }
        }
    };
}
