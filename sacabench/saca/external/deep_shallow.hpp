/*******************************************************************************
 * Copyright (C) 2018 Marvin Böcker <marvin.boecker@tu-dortmund.de>
 *
 * All rights reserved. Published under the BSD-3 license in the LICENSE file.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include "external_saca1.hpp"

extern int32_t _ds_Word_size;
extern "C" void ds_ssort(unsigned char* x, int32_t* p, int32_t n);

namespace sacabench::reference_sacas {
    struct deep_shallow {
        static constexpr size_t EXTRA_SENTINELS = 1;
        static constexpr char const* NAME = "Reference-Deep-Shallow";
        static constexpr char const* DESCRIPTION =
            "Reference implementation of Deep-Shallow";

        template <typename sa_index>
        inline static void construct_sa(util::string_span text,
                                 const util::alphabet&,
                                 util::span<sa_index> out_sa) {
            _ds_Word_size = 1;
            external_saca1::run_external<sa_index>(text, out_sa, ds_ssort);
        }
    };
}
