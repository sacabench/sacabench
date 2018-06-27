/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

extern "C" void ds_ssort(unsigned char* x, int32_t* p, int32_t n);

namespace sacabench::reference_sacas {
    struct deep_shallow {
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const* NAME = "Reference-Deep-Shallow";
        static constexpr char const* DESCRIPTION =
            "Reference implementation of Deep-Shallow";
            
        template <typename sa_index>
        static void construct_sa(util::string_span text,
                                 const util::alphabet&,
                                 util::span<sa_index> out_sa) {

            static_assert(sizeof(sa_index) >= sizeof(int32_t));

            auto* text_ptr = const_cast<uint8_t*>(text.data());
            auto* sa_ptr = out_sa.data();
            auto* sa_ptr32 = reinterpret_cast<int32_t*>(sa_ptr);
            size_t n = text.size();
            ::ds_ssort(text_ptr, sa_ptr32, n);
        }
    };
}
